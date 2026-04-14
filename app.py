from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import psycopg2
import psycopg2.extras
from functools import wraps
from sentence_transformers import SentenceTransformer
import os
from werkzeug.utils import secure_filename
from flask import make_response # For cache control

app = Flask(__name__)
app.secret_key = "dblp_secret_key_change_this"
app.jinja_env.globals.update(enumerate=enumerate)

DB_PARAMS = {
    "dbname": "dblp_project",
    "user": "siddharthkonnur",
    "password": "",   # Your Postgres password
    "host": "localhost",
    "port": "5432"
}

# --- HARDCODED ADMIN CREDENTIALS ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# --- FILE UPLOAD SETUP ---
UPLOAD_FOLDER = 'static/uploads/papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- FRONTEND DEV MODE: DUMMY MODEL ---
# print("Loading DUMMY model for fast UI testing...")
# class DummyEmbedder:
#     def encode(self, text, *args, **kwargs):
#         return [0.0] * 384 
# embedder = DummyEmbedder()

# --- UNCOMMENT FOR FINAL PRODUCTION ---
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

def get_db():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

# ── SECURITY ROLES ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return "Unauthorized: Admins only.", 403
        return f(*args, **kwargs)
    return decorated

# ── HOME / SEARCH ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name
        FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
        ORDER BY p.n_citations DESC LIMIT 6
    """)
    trending = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS cnt FROM Papers")
    paper_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Authors")
    author_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Citations")
    citation_count = cur.fetchone()["cnt"]

    cur.close(); conn.close()
    return render_template("index.html",
                           trending=trending,
                           paper_count=f"{paper_count:,}",
                           author_count=f"{author_count:,}",
                           citation_count=f"{citation_count:,}")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    sort  = request.args.get("sort", "relevance")
    page  = int(request.args.get("page", 1))
    per_page = 15
    offset = (page - 1) * per_page

    if not query:
        return redirect(url_for("index"))

    # Log search history (Skip if it's the admin)
    if "user_id" in session and session.get("role") != "admin":
        try:
            conn_log = get_db()
            cur_log = conn_log.cursor()
            cur_log.execute("""
                INSERT INTO Search_History (user_id, query, searched_at)
                VALUES (%s, %s, NOW())
            """, (session["user_id"], query))
            conn_log.commit()
            cur_log.close(); conn_log.close()
        except Exception:
            pass

    conn = get_db()
    cur = conn.cursor()

    order = "relevance_score DESC, p.n_citations DESC"
    if sort == "citations": order = "p.n_citations DESC"
    elif sort == "year": order = "p.publication_year DESC"

    cur.execute(f"""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name,
               ts_rank(to_tsvector('english', p.title || ' ' || p.abstract_text),
                       plainto_tsquery('english', %s)) AS relevance_score,
               LEFT(p.abstract_text, 200) AS snippet
        FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
        WHERE to_tsvector('english', p.title || ' ' || p.abstract_text)
              @@ plainto_tsquery('english', %s)
        ORDER BY {order} LIMIT %s OFFSET %s
    """, (query, query, per_page, offset))
    results = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*) AS cnt FROM Papers p
        WHERE to_tsvector('english', p.title || ' ' || p.abstract_text)
              @@ plainto_tsquery('english', %s)
    """, (query,))
    total = cur.fetchone()["cnt"]

    cur.close(); conn.close()
    return render_template("search.html", query=query, results=results, total=total,
                           page=page, per_page=per_page, sort=sort,
                           total_pages=(total + per_page - 1) // per_page)


@app.route("/paper/<int:paper_id>")
def paper_detail(paper_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT p.*, v.venue_name, v.venue_type, u.username as uploader_name, u.trust_factor
        FROM Papers p 
        JOIN Venues v ON p.venue_id = v.venue_id
        LEFT JOIN Users u ON p.uploaded_by = u.user_id
        WHERE p.paper_id = %s
    """, (paper_id,))
    paper = cur.fetchone()

    if not paper: return "Paper not found", 404

    cur.execute("""
        SELECT a.author_id, a.name FROM Authors a
        JOIN Paper_Authors pa ON a.author_id = pa.author_id
        WHERE pa.paper_id = %s
    """, (paper_id,))
    authors = cur.fetchall()

    has_rated = False
    is_bookmarked = False
    user_note = "" # NEW: Variable to hold the user's note
    if "user_id" in session and session.get("role") != "admin":
        cur.execute("SELECT 1 FROM Paper_Ratings WHERE user_id = %s AND paper_id = %s", (session["user_id"], paper_id))
        if cur.fetchone(): has_rated = True
            
        cur.execute("SELECT 1 FROM Bookmarks WHERE user_id = %s AND paper_id = %s", (session["user_id"], paper_id))
        if cur.fetchone(): is_bookmarked = True

        # NEW: Fetch the user's private note for this paper
        cur.execute("SELECT note_text FROM Paper_Notes WHERE user_id = %s AND paper_id = %s", (session["user_id"], paper_id))
        note_row = cur.fetchone()
        if note_row: user_note = note_row["note_text"]

        cur.execute("INSERT INTO Reading_History (user_id, paper_id, viewed_at) VALUES (%s, %s, NOW())", 
                    (session["user_id"], paper_id))
        conn.commit()

# --- ADVANCED DBMS: Recursive CTE + Window Functions ---
    # Finds papers citing this paper, partitioned to show:
    # Max 10 Level 1, Max 5 Level 2, Max 2 Level 3
    cur.execute("""
        WITH RECURSIVE CitationGraph AS (
            -- Base Case: Direct Citations (Depth 1)
            SELECT c.citing_paper_id AS paper_id, 1 AS depth
            FROM Citations c
            WHERE c.cited_paper_id = %s
            
            UNION ALL
            
            -- Recursive Step: Multi-hop Citations (Depth 2 & 3)
            SELECT c.citing_paper_id, cg.depth + 1
            FROM Citations c
            INNER JOIN CitationGraph cg ON c.cited_paper_id = cg.paper_id
            WHERE cg.depth < 3
        ),
        RankedCitations AS (
            -- Group to find the shortest path (MIN depth) if a paper appears multiple times
            SELECT 
                p.paper_id, p.title, p.publication_year, p.n_citations,
                MIN(cg.depth) as depth
            FROM CitationGraph cg
            JOIN Papers p ON cg.paper_id = p.paper_id
            GROUP BY p.paper_id, p.title, p.publication_year, p.n_citations
        ),
        PartitionedCitations AS (
            -- Use Window Functions to number the rows WITHIN each depth level
            SELECT *,
                   ROW_NUMBER() OVER(PARTITION BY depth ORDER BY n_citations DESC) as row_num
            FROM RankedCitations
        )
        -- Finally, filter based on your exact distribution requirements
        SELECT paper_id, title, publication_year, depth
        FROM PartitionedCitations
        WHERE (depth = 1 AND row_num <= 10)
           OR (depth = 2 AND row_num <= 5)
           OR (depth = 3 AND row_num <= 2)
        ORDER BY depth ASC, n_citations DESC;
    """, (paper_id,))
    citation_tree = cur.fetchall()

    cur.close(); conn.close()

    scholar_url = f"https://scholar.google.com/scholar?q={paper['title'].replace(' ', '+')}"
    arxiv_url   = f"https://arxiv.org/search/?query={paper['title'].replace(' ', '+')}"

    return render_template("paper.html", paper=paper, authors=authors,
                           scholar_url=scholar_url, arxiv_url=arxiv_url, 
                           has_rated=has_rated, is_bookmarked=is_bookmarked,
                           citation_tree=citation_tree)

@app.route("/for-you")
@login_required
def for_you():
    if session.get("role") == "admin": return redirect(url_for("admin_dashboard"))

    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT query, COUNT(*) AS freq FROM Search_History WHERE user_id = %s GROUP BY query ORDER BY freq DESC, MAX(searched_at) DESC LIMIT 8", (user_id,))
    top_queries = cur.fetchall()

    cur.execute("SELECT DISTINCT paper_id FROM Reading_History WHERE user_id = %s", (user_id,))
    read_ids = {r["paper_id"] for r in cur.fetchall()}

    cur.execute("SELECT paper_id FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmarked_ids = {r["paper_id"] for r in cur.fetchall()}

    already_seen = read_ids | bookmarked_ids
    recommendations = {}

    if top_queries:
        terms = [row["query"] for row in top_queries[:4]]
        combined_query = " OR ".join(terms)
        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name, LEFT(p.abstract_text, 180) AS snippet,
                   ts_rank(to_tsvector('english', p.title || ' ' || p.abstract_text), websearch_to_tsquery('english', %s)) AS score
            FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
            WHERE to_tsvector('english', p.title || ' ' || p.abstract_text) @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC, p.n_citations DESC LIMIT 30
        """, (combined_query, combined_query))

        for row in cur.fetchall():
            pid = row["paper_id"]
            if pid not in already_seen:
                recommendations[pid] = dict(row)
                recommendations[pid]["reason"] = "Matches your searches"
                recommendations[pid]["reason_type"] = "search"

    if len(recommendations) < 12:
        exclude = list(already_seen | set(recommendations.keys())) or [-1]
        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name, LEFT(p.abstract_text, 180) AS snippet, p.n_citations::float AS score
            FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
            WHERE p.paper_id != ALL(%s) ORDER BY p.n_citations DESC LIMIT %s
        """, (exclude, 24 - len(recommendations)))

        for row in cur.fetchall():
            pid = row["paper_id"]
            r = dict(row)
            r["reason"] = "Trending in the database"
            r["reason_type"] = "popular"
            recommendations[pid] = r

    priority = {"vector": 0, "search": 1, "popular": 2}
    sorted_recs = sorted(recommendations.values(), key=lambda x: (priority.get(x["reason_type"], 9), -float(x.get("score") or 0)))[:24]

    cur.execute("SELECT COUNT(DISTINCT paper_id) AS cnt FROM Reading_History WHERE user_id = %s", (user_id,))
    read_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmark_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM Search_History WHERE user_id = %s", (user_id,))
    search_count = cur.fetchone()["cnt"]

    cur.close(); conn.close()
    # Create a response object instead of returning the template directly
    response = make_response(render_template("foryou.html", 
                           recommendations=sorted_recs, 
                           top_queries=top_queries,
                           read_count=read_count, 
                           bookmark_count=bookmark_count, 
                           search_count=search_count,
                           has_history=bool(read_ids or top_queries)))
                           
    # Instruct the browser to never cache this page
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response
    


# ── AUTH & ADMIN ───────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["user_id"] = 0  
            session["username"] = "admin"
            session["role"] = "admin"
            return jsonify({"success": True}) 

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Users WHERE username = %s AND password_hash = %s", (username, password))
        user = cur.fetchone()
        cur.close(); conn.close()

        if user:
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["role"] = user.get("role", "user")
            return jsonify({"success": True})

        return jsonify({"success": False, "error": "Invalid credentials"})
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        role = "user"

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO Users (username, password_hash, gender, age, institute, role, trust_factor)
                VALUES (%s, %s, %s, %s, %s, %s, 5.0)
            """, (data["username"], data["password"], data.get("gender"), data.get("age"), data.get("institute"), role))
            conn.commit()
            return jsonify({"success": True})
        except psycopg2.IntegrityError:
            return jsonify({"success": False, "error": "Username already taken"})
        finally:
            cur.close(); conn.close()
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# --- ADMIN DASHBOARD ---
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT user_id, username, role, institute, trust_factor FROM Users ORDER BY user_id DESC")
    users = cur.fetchall()
    cur.execute("SELECT paper_id, title, publication_year FROM Papers ORDER BY uploaded_at DESC NULLS LAST LIMIT 100")
    papers = cur.fetchall()
    cur.close(); conn.close()
    return render_template("admin.html", users=users, papers=papers)

@app.route("/admin/search-papers")
@admin_required
def admin_search_papers():
    query = request.args.get("q", "").strip()
    conn = get_db()
    cur = conn.cursor()
    
    if query:
        # ILIKE is Postgres's case-insensitive search
        cur.execute("""
            SELECT paper_id, title, publication_year 
            FROM Papers 
            WHERE title ILIKE %s 
            ORDER BY paper_id DESC LIMIT 50
        """, (f"%{query}%",))
    else:
        # If search is empty, just return the 100 most recent
        cur.execute("""
            SELECT paper_id, title, publication_year 
            FROM Papers 
            ORDER BY uploaded_at DESC NULLS LAST LIMIT 100
        """)
        
    papers = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"papers": papers})

@app.route("/admin/delete-user/<int:del_uid>", methods=["POST"])
@admin_required
def admin_delete_user(del_uid):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT paper_id FROM Papers WHERE uploaded_by = %s", (del_uid,))
        user_papers = [r["paper_id"] for r in cur.fetchall()]
        if user_papers:
            cur.execute("DELETE FROM Paper_Ratings WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Embeddings WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Bookmarks WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Reading_History WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Authors WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Citations WHERE citing_paper_id = ANY(%s) OR cited_paper_id = ANY(%s)", (user_papers, user_papers))
            cur.execute("DELETE FROM Papers WHERE paper_id = ANY(%s)", (user_papers,))
        
        cur.execute("DELETE FROM Paper_Ratings WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Bookmarks WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Reading_History WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Search_History WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Users WHERE user_id = %s", (del_uid,))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()
# ── PROFILE & PAPER MANAGEMENT ─────────────────────────────────────────────────

@app.route("/profile")
@login_required
def profile():
    if session.get("role") == "admin": return redirect(url_for("admin_dashboard"))

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM Users WHERE user_id = %s", (session["user_id"],))
    user = cur.fetchone()

    cur.execute("""
        SELECT DISTINCT ON (p.paper_id) p.paper_id, p.title, p.publication_year, p.n_citations, rh.viewed_at
        FROM Reading_History rh JOIN Papers p ON rh.paper_id = p.paper_id
        WHERE rh.user_id = %s ORDER BY p.paper_id, rh.viewed_at DESC
    """, (session["user_id"],))
    history = sorted(cur.fetchall(), key=lambda x: x["viewed_at"], reverse=True)

    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, b.bookmarked_at
        FROM Bookmarks b JOIN Papers p ON b.paper_id = p.paper_id
        WHERE b.user_id = %s ORDER BY b.bookmarked_at DESC
    """, (session["user_id"],))
    bookmarks = cur.fetchall()

    cur.execute("SELECT query, COUNT(*) AS freq, MAX(searched_at) AS last_searched FROM Search_History WHERE user_id = %s GROUP BY query ORDER BY last_searched DESC LIMIT 20", (session["user_id"],))
    search_history = cur.fetchall()

    cur.execute("SELECT p.paper_id, p.title, p.publication_year, p.n_citations, p.uploaded_at, p.pdf_path FROM Papers p WHERE p.uploaded_by = %s ORDER BY p.uploaded_at DESC", (session["user_id"],))
    my_uploads = cur.fetchall()

    cur.close(); conn.close()
    return render_template("profile.html", user=user, history=history, bookmarks=bookmarks, search_history=search_history, my_uploads=my_uploads)

@app.route("/add-paper", methods=["GET", "POST"])
@login_required
def add_paper():
    if request.method == "POST":
        user_id = session["user_id"]
        title, year = request.form.get("title", "").strip(), request.form.get("year")
        venue_name, abstract = request.form.get("venue", "Unknown Venue").strip(), request.form.get("abstract", "").strip()
        pdf_file = request.files.get("pdf_file")
        
        conn = get_db()
        cur = conn.cursor()

        try:
            if session.get("role") != "admin":
                cur.execute("SELECT COUNT(*) as cnt FROM Papers WHERE uploaded_by = %s AND uploaded_at::date = CURRENT_DATE", (user_id,))
                if cur.fetchone()["cnt"] >= 5: return jsonify({"success": False, "error": "Daily limit of 5 uploads reached."})

            cur.execute("SELECT venue_id FROM Venues WHERE venue_name = %s", (venue_name,))
            v_row = cur.fetchone()
            if v_row: venue_id = v_row["venue_id"]
            else:
                cur.execute("SELECT COALESCE(MAX(venue_id), 0) + 1 AS next_vid FROM Venues")
                venue_id = cur.fetchone()["next_vid"]
                cur.execute("INSERT INTO Venues (venue_id, venue_name, venue_type) VALUES (%s, %s, 'C')", (venue_id, venue_name))

            cur.execute("SELECT COALESCE(MAX(paper_id), 0) + 1 AS next_id FROM Papers")
            new_paper_id = cur.fetchone()["next_id"]

            pdf_path = None
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                filename = secure_filename(f"paper_{new_paper_id}.pdf")
                pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pdf_path = f"/static/uploads/papers/{filename}"

            cur.execute("""
                INSERT INTO Papers (paper_id, title, publication_year, abstract_text, n_citations, venue_id, uploaded_by, pdf_path)
                VALUES (%s, %s, %s, %s, 0, %s, %s, %s)
            """, (new_paper_id, title, year, abstract, venue_id, user_id, pdf_path))

            cur.execute("INSERT INTO Paper_Embeddings (paper_id, embedding) VALUES (%s, %s::vector)", (new_paper_id, embedder.encode(abstract).tolist()))

            conn.commit()
            return jsonify({"success": True, "paper_id": new_paper_id})
        except Exception as e:
            conn.rollback(); return jsonify({"success": False, "error": str(e)})
        finally: cur.close(); conn.close()

    return render_template("add_paper.html")


@app.route("/edit-paper/<int:paper_id>", methods=["GET", "POST"])
@login_required
def edit_paper(paper_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT p.*, v.venue_name FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id WHERE p.paper_id = %s", (paper_id,))
    paper = cur.fetchone()

    if not paper or (paper["uploaded_by"] != session["user_id"] and session.get("role") != "admin"):
        cur.close(); conn.close(); return "Unauthorized or paper not found", 403

    if request.method == "POST":
        title, year = request.form.get("title", "").strip(), request.form.get("year")
        venue_name, abstract = request.form.get("venue", "Unknown Venue").strip(), request.form.get("abstract", "").strip()
        pdf_file = request.files.get("pdf_file")
        
        try:
            cur.execute("SELECT venue_id FROM Venues WHERE venue_name = %s", (venue_name,))
            v_row = cur.fetchone()
            if v_row: venue_id = v_row["venue_id"]
            else:
                cur.execute("SELECT COALESCE(MAX(venue_id), 0) + 1 AS next_vid FROM Venues")
                venue_id = cur.fetchone()["next_vid"]
                cur.execute("INSERT INTO Venues (venue_id, venue_name, venue_type) VALUES (%s, %s, 'C')", (venue_id, venue_name))

            pdf_path = paper["pdf_path"]
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                if pdf_path and os.path.exists(os.path.join(app.root_path, pdf_path.lstrip('/'))): os.remove(os.path.join(app.root_path, pdf_path.lstrip('/')))
                filename = secure_filename(f"paper_{paper_id}.pdf")
                pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pdf_path = f"/static/uploads/papers/{filename}"

            cur.execute("UPDATE Papers SET title = %s, publication_year = %s, abstract_text = %s, venue_id = %s, pdf_path = %s WHERE paper_id = %s", (title, year, abstract, venue_id, pdf_path, paper_id))
            if paper["abstract_text"] != abstract:
                cur.execute("UPDATE Paper_Embeddings SET embedding = %s::vector WHERE paper_id = %s", (embedder.encode(abstract).tolist(), paper_id))

            conn.commit(); return jsonify({"success": True})
        except Exception as e:
            conn.rollback(); return jsonify({"success": False, "error": str(e)})
        finally: cur.close(); conn.close()

    cur.close(); conn.close()
    return render_template("edit_paper.html", paper=paper)

@app.route("/delete-papers", methods=["POST"])
@login_required
def delete_papers():
    data = request.get_json()
    paper_ids = data.get("paper_ids", [])
    if not paper_ids: return jsonify({"success": False, "error": "No papers selected."})

    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    try:
        if session.get("role") == "admin": cur.execute("SELECT paper_id, pdf_path FROM Papers WHERE paper_id = ANY(%s)", (paper_ids,))
        else: cur.execute("SELECT paper_id, pdf_path FROM Papers WHERE uploaded_by = %s AND paper_id = ANY(%s)", (user_id, paper_ids))
            
        rows = cur.fetchall()
        valid_ids = [row["paper_id"] for row in rows]
        if not valid_ids: return jsonify({"success": False, "error": "Unauthorized or papers not found."})

        paths_to_delete = [row["pdf_path"] for row in rows if row["pdf_path"]]
        for path in paths_to_delete:
            full_path = os.path.join(app.root_path, path.lstrip('/'))
            if os.path.exists(full_path): os.remove(full_path)

        cur.execute("DELETE FROM Paper_Ratings WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Paper_Embeddings WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Bookmarks WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Reading_History WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Paper_Authors WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Citations WHERE citing_paper_id = ANY(%s) OR cited_paper_id = ANY(%s)", (valid_ids, valid_ids))
        cur.execute("DELETE FROM Papers WHERE paper_id = ANY(%s)", (valid_ids,))

        conn.commit(); return jsonify({"success": True, "deleted_count": len(valid_ids)})
    except Exception as e:
        conn.rollback(); return jsonify({"success": False, "error": str(e)})
    finally: cur.close(); conn.close()


@app.route("/rate/<int:paper_id>", methods=["POST"])
@login_required
def rate_paper(paper_id):
    data = request.get_json()
    rating = data.get("rating")
    
    if not rating or not (1 <= int(rating) <= 10):
        return jsonify({"success": False, "error": "Rating must be between 1 and 10."})
        
    rating = int(rating)
    user_id = session["user_id"]
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT uploaded_by FROM Papers WHERE paper_id = %s", (paper_id,))
        paper = cur.fetchone()
        
        if not paper or not paper["uploaded_by"]:
            return jsonify({"success": False, "error": "This paper cannot be rated."})
            
        uploader_id = paper["uploaded_by"]
        if uploader_id == user_id:
            return jsonify({"success": False, "error": "You cannot rate your own paper."})
            
        # STRICT CHECK: See if the user has already rated this paper
        cur.execute("SELECT 1 FROM Paper_Ratings WHERE user_id = %s AND paper_id = %s", (user_id, paper_id))
        if cur.fetchone():
            return jsonify({"success": False, "error": "You have already rated this paper."})
            
        # Secure INSERT without ON CONFLICT to ensure only one rating goes through
        cur.execute("""
            INSERT INTO Paper_Ratings (user_id, paper_id, rating) 
            VALUES (%s, %s, %s)
        """, (user_id, paper_id, rating))
        
        # Calculate EMA
        cur.execute("SELECT trust_factor FROM Users WHERE user_id = %s", (uploader_id,))
        old_tf = float(cur.fetchone()["trust_factor"])
        new_tf = round((old_tf + rating) / 2.0, 1)
        
        cur.execute("UPDATE Users SET trust_factor = %s WHERE user_id = %s", (new_tf, uploader_id))
        conn.commit()
        
        return jsonify({"success": True, "new_tf": new_tf})
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"success": False, "error": "You have already rated this paper."})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close()
        conn.close()


@app.route("/bookmark/<int:paper_id>", methods=["POST"])
def bookmark(paper_id):
    if "user_id" not in session or session.get("role") == "admin": return jsonify({"error": "Unauthorized"}), 401
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT 1 FROM Bookmarks WHERE user_id=%s AND paper_id=%s", (session["user_id"], paper_id))
    if cur.fetchone():
        cur.execute("DELETE FROM Bookmarks WHERE user_id=%s AND paper_id=%s", (session["user_id"], paper_id))
        conn.commit(); cur.close(); conn.close(); return jsonify({"bookmarked": False})
    else:
        cur.execute("INSERT INTO Bookmarks (user_id, paper_id, bookmarked_at) VALUES (%s, %s, NOW())", (session["user_id"], paper_id))
        conn.commit(); cur.close(); conn.close(); return jsonify({"bookmarked": True})


@app.route("/analytics")
def analytics():
    conn = get_db()
    cur = conn.cursor()

    # 1. Papers per year
    cur.execute("""
        SELECT publication_year AS year, COUNT(*) AS count
        FROM Papers
        GROUP BY publication_year
        ORDER BY publication_year
    """)
    papers_per_year = cur.fetchall()

# 2. Query the Materialized View directly (Instantaneous!)
    cur.execute("SELECT * FROM mv_top_venues")
    top_venues = cur.fetchall()

    # 3. Query the Materialized View directly
    cur.execute("SELECT * FROM mv_top_authors")
    top_authors = cur.fetchall()

    # 4. Top Papers
    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name
        FROM Papers p
        LEFT JOIN Venues v ON p.venue_id = v.venue_id
        ORDER BY p.n_citations DESC NULLS LAST
        LIMIT 10
    """)
    top_papers = cur.fetchall()

    # Global Counts
    cur.execute("SELECT COUNT(*) AS cnt FROM Papers")
    total_papers = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Authors")
    total_authors = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Citations")
    total_citations = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Venues")
    total_venues = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Users")
    total_users = cur.fetchone()["cnt"]

    cur.close()
    conn.close()

    return render_template("analytics.html",
        papers_per_year=papers_per_year,
        top_venues=top_venues,
        top_authors=top_authors,
        top_papers=top_papers,
        total_papers=f"{total_papers:,}",
        total_authors=f"{total_authors:,}",
        total_citations=f"{total_citations:,}",
        total_venues=f"{total_venues:,}",
        total_users=f"{total_users:,}",
    )

# author part
@app.route("/search_authors")
def search_authors():
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()
    
    # Advanced DB Concept: ILIKE for case-insensitive substring matching, 
    # joined with aggregations to show author stats in search results.
    cur.execute("""
        SELECT a.author_id, a.name, 
               COUNT(pa.paper_id) as total_papers,
               COALESCE(SUM(p.n_citations), 0) as total_citations
        FROM Authors a
        LEFT JOIN Paper_Authors pa ON a.author_id = pa.author_id
        LEFT JOIN Papers p ON pa.paper_id = p.paper_id
        WHERE a.name ILIKE %s
        GROUP BY a.author_id, a.name
        ORDER BY total_citations DESC, total_papers DESC
        LIMIT 20
    """, (f"%{query}%",))
    
    results = cur.fetchall()
    cur.close(); conn.close()
    
    # You can reuse search.html or render a simple template. 
    # Assuming we pass it to a new template or handle it in UI.
    return render_template("author_search_results.html", query=query, results=results)


@app.route("/author/<int:author_id>")
def author_profile(author_id):
    conn = get_db()
    cur = conn.cursor()

    # 1. Get basic author info
    cur.execute("SELECT * FROM Authors WHERE author_id = %s", (author_id,))
    author = cur.fetchone()
    if not author: return "Author not found", 404

    # 2. Get author's papers
    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name
        FROM Papers p
        JOIN Paper_Authors pa ON p.paper_id = pa.paper_id
        LEFT JOIN Venues v ON p.venue_id = v.venue_id
        WHERE pa.author_id = %s
        ORDER BY p.publication_year DESC, p.n_citations DESC
    """, (author_id,))
    papers = cur.fetchall()

    # 3. Calculate total citations and top venue
    total_citations = sum(p["n_citations"] for p in papers)
    
    # 4. ADVANCED DBMS: Author Network Reach (Recursive CTE)
    # This finds how many papers cited THIS author's papers (Depth 1)
    # How many papers cited THOSE papers (Depth 2), and so on up to Depth 3.
    cur.execute("""
        WITH RECURSIVE AuthorPapers AS (
            SELECT paper_id FROM Paper_Authors WHERE author_id = %s
        ),
        CitationGraph AS (
            -- Base Case: Papers directly citing the author's papers
            SELECT c.citing_paper_id AS paper_id, 1 AS depth
            FROM Citations c
            INNER JOIN AuthorPapers ap ON c.cited_paper_id = ap.paper_id
            
            UNION ALL
            
            -- Recursive Step: Multi-hop citations (Papers citing the citing papers)
            SELECT c.citing_paper_id, cg.depth + 1
            FROM Citations c
            INNER JOIN CitationGraph cg ON c.cited_paper_id = cg.paper_id
            WHERE cg.depth < 3
        )
        -- Aggregate unique papers at each depth level
        SELECT depth, COUNT(DISTINCT paper_id) as hop_count
        FROM CitationGraph
        GROUP BY depth
        ORDER BY depth;
    """, (author_id,))
    network_reach = {row["depth"]: row["hop_count"] for row in cur.fetchall()}

    cur.close(); conn.close()

    return render_template("author.html", 
                           author=author, 
                           papers=papers, 
                           total_citations=total_citations,
                           network_reach=network_reach)

@app.route("/save_note/<int:paper_id>", methods=["POST"])
@login_required
def save_note(paper_id):
    data = request.get_json()
    note_text = data.get("note_text", "").strip()
    user_id = session["user_id"]

    conn = get_db()
    cur = conn.cursor()

    try:
        if not note_text:
            # If the note is empty, we delete the record to save space
            cur.execute("DELETE FROM Paper_Notes WHERE user_id = %s AND paper_id = %s", (user_id, paper_id))
        else:
            # ADVANCED DBMS: The UPSERT (Insert or Update)
            # If the user_id + paper_id combination already exists, it updates the text.
            cur.execute("""
                INSERT INTO Paper_Notes (user_id, paper_id, note_text, last_updated)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id, paper_id) 
                DO UPDATE SET note_text = EXCLUDED.note_text, last_updated = NOW()
            """, (user_id, paper_id, note_text))
            
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    app.run(debug=True, port=5000)