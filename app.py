from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import psycopg2
import psycopg2.extras
from functools import wraps
from sentence_transformers import SentenceTransformer
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "dblp_secret_key_change_this"
app.jinja_env.globals.update(enumerate=enumerate)

DB_PARAMS = {
    "dbname": "dblp_project",
    "user": "postgres",
    "password": "praty",   # change this gng🥀
    "host": "localhost",
    "port": "5432"
}

# --- FILE UPLOAD SETUP ---
UPLOAD_FOLDER = 'static/uploads/papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Load ML model once on startup
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# --- FRONTEND DEV MODE: DUMMY MODEL ---
# print("Loading DUMMY model for fast UI testing...")
# class DummyEmbedder:
#     def encode(self, text, *args, **kwargs):
#         # Returns a fake array of 384 zeros so the DB doesn't crash if you add a paper
#         return [0.0] * 384 

# embedder = DummyEmbedder()

def get_db():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
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

    cur.close()
    conn.close()
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

    # ── LOG SEARCH HISTORY ─────────────────────────────────
    if "user_id" in session:
        try:
            conn_log = get_db()
            cur_log = conn_log.cursor()
            cur_log.execute("""
                INSERT INTO Search_History (user_id, query, searched_at)
                VALUES (%s, %s, NOW())
            """, (session["user_id"], query))
            conn_log.commit()
            cur_log.close()
            conn_log.close()
        except Exception:
            pass
    # ───────────────────────────────────────────────────────

    conn = get_db()
    cur = conn.cursor()

    order = "relevance_score DESC, p.n_citations DESC"
    if sort == "citations":
        order = "p.n_citations DESC"
    elif sort == "year":
        order = "p.publication_year DESC"

    cur.execute(f"""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name,
               ts_rank(to_tsvector('english', p.title || ' ' || p.abstract_text),
                       plainto_tsquery('english', %s)) AS relevance_score,
               LEFT(p.abstract_text, 200) AS snippet
        FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
        WHERE to_tsvector('english', p.title || ' ' || p.abstract_text)
              @@ plainto_tsquery('english', %s)
        ORDER BY {order}
        LIMIT %s OFFSET %s
    """, (query, query, per_page, offset))
    results = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*) AS cnt FROM Papers p
        WHERE to_tsvector('english', p.title || ' ' || p.abstract_text)
              @@ plainto_tsquery('english', %s)
    """, (query,))
    total = cur.fetchone()["cnt"]

    cur.close()
    conn.close()

    return render_template("search.html",
                           query=query,
                           results=results,
                           total=total,
                           page=page,
                           per_page=per_page,
                           sort=sort,
                           total_pages=(total + per_page - 1) // per_page)


# ── PAPER DETAIL ───────────────────────────────────────────────────────────────

@app.route("/paper/<int:paper_id>")
def paper_detail(paper_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT p.*, v.venue_name, v.venue_type
        FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
        WHERE p.paper_id = %s
    """, (paper_id,))
    paper = cur.fetchone()

    if not paper:
        return "Paper not found", 404

    cur.execute("""
        SELECT a.author_id, a.name FROM Authors a
        JOIN Paper_Authors pa ON a.author_id = pa.author_id
        WHERE pa.paper_id = %s
    """, (paper_id,))
    authors = cur.fetchall()

    cur.execute("""
        SELECT p.paper_id, p.title, p.n_citations,
               pe.embedding <=> ref.embedding AS distance
        FROM Paper_Embeddings pe
        JOIN Papers p ON pe.paper_id = p.paper_id
        CROSS JOIN (SELECT embedding FROM Paper_Embeddings WHERE paper_id = %s) AS ref
        WHERE pe.paper_id != %s
        ORDER BY distance ASC LIMIT 5
    """, (paper_id, paper_id))
    similar = cur.fetchall()

    cur.execute("""
        SELECT p.paper_id, p.title, p.n_citations
        FROM Citations c JOIN Papers p ON c.cited_paper_id = p.paper_id
        WHERE c.citing_paper_id = %s LIMIT 10
    """, (paper_id,))
    references = cur.fetchall()

    cur.execute("""
        SELECT p.paper_id, p.title, p.n_citations
        FROM Citations c JOIN Papers p ON c.citing_paper_id = p.paper_id
        WHERE c.cited_paper_id = %s LIMIT 10
    """, (paper_id,))
    cited_by = cur.fetchall()

    if "user_id" in session:
        cur.execute("""
            INSERT INTO Reading_History (user_id, paper_id, viewed_at)
            VALUES (%s, %s, NOW())
        """, (session["user_id"], paper_id))
        conn.commit()

    cur.close()
    conn.close()

    scholar_url = f"https://scholar.google.com/scholar?q={paper['title'].replace(' ', '+')}"
    arxiv_url   = f"https://arxiv.org/search/?query={paper['title'].replace(' ', '+')}"

    return render_template("paper.html", paper=paper, authors=authors,
                           similar=similar, references=references,
                           cited_by=cited_by, scholar_url=scholar_url,
                           arxiv_url=arxiv_url)


# ── FOR YOU PAGE ───────────────────────────────────────────────────────────────

@app.route("/for-you")
@login_required
def for_you():
    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT query, COUNT(*) AS freq
        FROM Search_History
        WHERE user_id = %s
        GROUP BY query
        ORDER BY freq DESC, MAX(searched_at) DESC
        LIMIT 8
    """, (user_id,))
    top_queries = cur.fetchall()

    cur.execute("SELECT DISTINCT paper_id FROM Reading_History WHERE user_id = %s", (user_id,))
    read_ids = {r["paper_id"] for r in cur.fetchall()}

    cur.execute("SELECT paper_id FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmarked_ids = {r["paper_id"] for r in cur.fetchall()}

    already_seen = read_ids | bookmarked_ids
    recommendations = {}

    if top_queries:
        terms = [row["query"] for row in top_queries[:4]]
        # Use OR for websearch_to_tsquery
        combined_query = " OR ".join(terms)

        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations,
                   v.venue_name,
                   LEFT(p.abstract_text, 180) AS snippet,
                   ts_rank(
                       to_tsvector('english', p.title || ' ' || p.abstract_text),
                       websearch_to_tsquery('english', %s)
                   ) AS score
            FROM Papers p
            JOIN Venues v ON p.venue_id = v.venue_id
            WHERE to_tsvector('english', p.title || ' ' || p.abstract_text)
                  @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC, p.n_citations DESC
            LIMIT 30
        """, (combined_query, combined_query))

        for row in cur.fetchall():
            pid = row["paper_id"]
            if pid not in already_seen:
                recommendations[pid] = dict(row)
                recommendations[pid]["reason"] = "Matches your searches"
                recommendations[pid]["reason_type"] = "search"

    cur.execute("""
        SELECT rh.paper_id
        FROM Reading_History rh
        JOIN Paper_Embeddings pe ON rh.paper_id = pe.paper_id
        WHERE rh.user_id = %s
        ORDER BY rh.viewed_at DESC
        LIMIT 3
    """, (user_id,))
    seed_papers = [r["paper_id"] for r in cur.fetchall()]

    for seed_id in seed_papers:
        # Added missing JOIN Venues v ON p.venue_id = v.venue_id
        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations,
                   v.venue_name,
                   LEFT(p.abstract_text, 180) AS snippet,
                   pe.embedding <=> ref.embedding AS distance
            FROM Paper_Embeddings pe
            JOIN Papers p ON pe.paper_id = p.paper_id
            JOIN Venues v ON p.venue_id = v.venue_id
            CROSS JOIN (SELECT embedding FROM Paper_Embeddings WHERE paper_id = %s) AS ref
            WHERE pe.paper_id != %s
            ORDER BY distance ASC
            LIMIT 10
        """, (seed_id, seed_id))

        for row in cur.fetchall():
            pid = row["paper_id"]
            if pid not in already_seen and pid not in recommendations:
                r = dict(row)
                r["score"] = 1.0 - float(row["distance"])
                r["reason"] = "Similar to papers you read"
                r["reason_type"] = "vector"
                recommendations[pid] = r

    if len(recommendations) < 12:
        exclude = list(already_seen | set(recommendations.keys()))
        if not exclude:
            exclude = [-1]

        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations,
                   v.venue_name,
                   LEFT(p.abstract_text, 180) AS snippet,
                   p.n_citations::float AS score
            FROM Papers p
            JOIN Venues v ON p.venue_id = v.venue_id
            WHERE p.paper_id != ALL(%s)
            ORDER BY p.n_citations DESC
            LIMIT %s
        """, (exclude, 24 - len(recommendations)))

        for row in cur.fetchall():
            pid = row["paper_id"]
            r = dict(row)
            r["reason"] = "Trending in the database"
            r["reason_type"] = "popular"
            recommendations[pid] = r

    priority = {"vector": 0, "search": 1, "popular": 2}
    sorted_recs = sorted(
        recommendations.values(),
        key=lambda x: (priority.get(x["reason_type"], 9), -float(x.get("score") or 0))
    )[:24]

    cur.execute("SELECT COUNT(*) AS cnt FROM Reading_History WHERE user_id = %s", (user_id,))
    read_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmark_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM Search_History WHERE user_id = %s", (user_id,))
    search_count = cur.fetchone()["cnt"]

    cur.close()
    conn.close()

    return render_template("foryou.html",
                           recommendations=sorted_recs,
                           top_queries=top_queries,
                           read_count=read_count,
                           bookmark_count=bookmark_count,
                           search_count=search_count,
                           has_history=bool(read_ids or top_queries))


# ── AUTH ───────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Users WHERE username = %s AND password_hash = %s",
                    (data["username"], data["password"]))
        user = cur.fetchone()
        cur.close(); conn.close()
        if user:
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Invalid credentials"})
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO Users (username, password_hash, gender, age, institute)
                VALUES (%s, %s, %s, %s, %s)
            """, (data["username"], data["password"],
                  data.get("gender"), data.get("age"), data.get("institute")))
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


@app.route("/bookmark/<int:paper_id>", methods=["POST"])
def bookmark(paper_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM Bookmarks WHERE user_id=%s AND paper_id=%s",
                (session["user_id"], paper_id))
    exists = cur.fetchone()
    if exists:
        cur.execute("DELETE FROM Bookmarks WHERE user_id=%s AND paper_id=%s",
                    (session["user_id"], paper_id))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"bookmarked": False})
    else:
        cur.execute("INSERT INTO Bookmarks (user_id, paper_id, bookmarked_at) VALUES (%s, %s, NOW())",
                    (session["user_id"], paper_id))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"bookmarked": True})


@app.route("/analytics")
def analytics():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT publication_year AS year, COUNT(*) AS count
        FROM Papers
        GROUP BY publication_year
        ORDER BY publication_year
    """)
    papers_per_year = cur.fetchall()

    try:
        cur.execute("""
            SELECT venue_name, total_papers, avg_citations, total_citations
            FROM mv_venue_stats
            WHERE total_papers >= 5
            ORDER BY avg_citations DESC
            LIMIT 10
        """)
        top_venues = cur.fetchall()

        cur.execute("""
            SELECT name, total_papers, total_citations
            FROM mv_author_stats
            ORDER BY total_citations DESC
            LIMIT 10
        """)
        top_authors = cur.fetchall()

        cur.execute("""
            SELECT paper_id, title, publication_year, n_citations, venue_name
            FROM mv_top_cited_papers
            LIMIT 10
        """)
        top_papers = cur.fetchall()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        top_venues, top_authors, top_papers = [], [], []

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


# ── PROFILE & PAPER MANAGEMENT ─────────────────────────────────────────────────

@app.route("/profile")
@login_required
def profile():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM Users WHERE user_id = %s", (session["user_id"],))
    user = cur.fetchone()

    cur.execute("""
        SELECT DISTINCT ON (p.paper_id)
            p.paper_id, p.title, p.publication_year, p.n_citations,
            rh.viewed_at
        FROM Reading_History rh
        JOIN Papers p ON rh.paper_id = p.paper_id
        WHERE rh.user_id = %s
        ORDER BY p.paper_id, rh.viewed_at DESC
    """, (session["user_id"],))
    history_raw = cur.fetchall()
    history = sorted(history_raw, key=lambda x: x["viewed_at"], reverse=True)

    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations,
               b.bookmarked_at
        FROM Bookmarks b
        JOIN Papers p ON b.paper_id = p.paper_id
        WHERE b.user_id = %s
        ORDER BY b.bookmarked_at DESC
    """, (session["user_id"],))
    bookmarks = cur.fetchall()

    cur.execute("""
        SELECT query, COUNT(*) AS freq, MAX(searched_at) AS last_searched
        FROM Search_History
        WHERE user_id = %s
        GROUP BY query
        ORDER BY last_searched DESC
        LIMIT 20
    """, (session["user_id"],))
    search_history = cur.fetchall()

    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, p.n_citations, p.uploaded_at, p.pdf_path 
        FROM Papers p 
        WHERE p.uploaded_by = %s 
        ORDER BY p.uploaded_at DESC
    """, (session["user_id"],))
    my_uploads = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("profile.html",
                           user=user,
                           history=history,
                           bookmarks=bookmarks,
                           search_history=search_history,
                           my_uploads=my_uploads)


@app.route("/add-paper", methods=["GET", "POST"])
@login_required
def add_paper():
    if request.method == "POST":
        user_id = session["user_id"]
        
        # Using request.form for FormData
        title = request.form.get("title", "").strip()
        year = request.form.get("year")
        venue_name = request.form.get("venue", "Unknown Venue").strip()
        abstract = request.form.get("abstract", "").strip()
        pdf_file = request.files.get("pdf_file")
        
        conn = get_db()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT COUNT(*) as cnt FROM Papers 
                WHERE uploaded_by = %s AND uploaded_at::date = CURRENT_DATE
            """, (user_id,))
            
            if cur.fetchone()["cnt"] >= 5:
                return jsonify({"success": False, "error": "Daily limit of 5 uploads reached."})

            cur.execute("SELECT venue_id FROM Venues WHERE venue_name = %s", (venue_name,))
            v_row = cur.fetchone()
            
            if v_row:
                venue_id = v_row["venue_id"]
            else:
                cur.execute("SELECT COALESCE(MAX(venue_id), 0) + 1 AS next_vid FROM Venues")
                venue_id = cur.fetchone()["next_vid"]
                cur.execute("INSERT INTO Venues (venue_id, venue_name, venue_type) VALUES (%s, %s, 'C')", (venue_id, venue_name))

            cur.execute("SELECT COALESCE(MAX(paper_id), 0) + 1 AS next_id FROM Papers")
            new_paper_id = cur.fetchone()["next_id"]

            # Handle PDF
            pdf_path = None
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                filename = secure_filename(f"paper_{new_paper_id}.pdf")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf_file.save(filepath)
                pdf_path = f"/static/uploads/papers/{filename}"

            cur.execute("""
                INSERT INTO Papers 
                (paper_id, title, publication_year, abstract_text, n_citations, venue_id, uploaded_by, pdf_path)
                VALUES (%s, %s, %s, %s, 0, %s, %s, %s)
            """, (new_paper_id, title, year, abstract, venue_id, user_id, pdf_path))

            embedding = embedder.encode(abstract).tolist()
            cur.execute("""
                INSERT INTO Paper_Embeddings (paper_id, embedding) 
                VALUES (%s, %s::vector)
            """, (new_paper_id, embedding))

            conn.commit()
            return jsonify({"success": True, "paper_id": new_paper_id})
            
        except Exception as e:
            conn.rollback()
            return jsonify({"success": False, "error": str(e)})
        finally:
            cur.close()
            conn.close()

    return render_template("add_paper.html")


@app.route("/edit-paper/<int:paper_id>", methods=["GET", "POST"])
@login_required
def edit_paper(paper_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT p.*, v.venue_name 
        FROM Papers p 
        JOIN Venues v ON p.venue_id = v.venue_id 
        WHERE p.paper_id = %s AND p.uploaded_by = %s
    """, (paper_id, session["user_id"]))
    paper = cur.fetchone()

    if not paper:
        cur.close(); conn.close()
        return "Unauthorized or paper not found", 403

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        year = request.form.get("year")
        venue_name = request.form.get("venue", "Unknown Venue").strip()
        abstract = request.form.get("abstract", "").strip()
        pdf_file = request.files.get("pdf_file")
        
        try:
            cur.execute("SELECT venue_id FROM Venues WHERE venue_name = %s", (venue_name,))
            v_row = cur.fetchone()
            
            if v_row:
                venue_id = v_row["venue_id"]
            else:
                cur.execute("SELECT COALESCE(MAX(venue_id), 0) + 1 AS next_vid FROM Venues")
                venue_id = cur.fetchone()["next_vid"]
                cur.execute("INSERT INTO Venues (venue_id, venue_name, venue_type) VALUES (%s, %s, 'C')", (venue_id, venue_name))

            # Handle PDF update
            pdf_path = paper["pdf_path"]
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                # Delete old if exists
                if pdf_path:
                    old_path = os.path.join(app.root_path, pdf_path.lstrip('/'))
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                filename = secure_filename(f"paper_{paper_id}.pdf")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf_file.save(filepath)
                pdf_path = f"/static/uploads/papers/{filename}"

            cur.execute("""
                UPDATE Papers 
                SET title = %s, publication_year = %s, abstract_text = %s, venue_id = %s, pdf_path = %s
                WHERE paper_id = %s
            """, (title, year, abstract, venue_id, pdf_path, paper_id))

            if paper["abstract_text"] != abstract:
                embedding = embedder.encode(abstract).tolist()
                cur.execute("""
                    UPDATE Paper_Embeddings 
                    SET embedding = %s::vector 
                    WHERE paper_id = %s
                """, (embedding, paper_id))

            conn.commit()
            return jsonify({"success": True})
        except Exception as e:
            conn.rollback()
            return jsonify({"success": False, "error": str(e)})
        finally:
            cur.close(); conn.close()

    cur.close()
    conn.close()
    return render_template("edit_paper.html", paper=paper)


@app.route("/delete-papers", methods=["POST"])
@login_required
def delete_papers():
    data = request.get_json()
    paper_ids = data.get("paper_ids", [])
    
    if not paper_ids:
        return jsonify({"success": False, "error": "No papers selected."})

    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT paper_id FROM Papers 
            WHERE uploaded_by = %s AND paper_id = ANY(%s)
        """, (user_id, paper_ids))
        
        valid_ids = [row["paper_id"] for row in cur.fetchall()]

        if not valid_ids:
            return jsonify({"success": False, "error": "Unauthorized or papers not found."})

        # Delete physical PDF files from disk
        cur.execute("SELECT pdf_path FROM Papers WHERE paper_id = ANY(%s) AND pdf_path IS NOT NULL", (valid_ids,))
        paths_to_delete = [row["pdf_path"] for row in cur.fetchall()]
        
        for path in paths_to_delete:
            full_path = os.path.join(app.root_path, path.lstrip('/'))
            if os.path.exists(full_path):
                os.remove(full_path)

        # Delete DB entries
        cur.execute("DELETE FROM Paper_Embeddings WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Bookmarks WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Reading_History WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Paper_Authors WHERE paper_id = ANY(%s)", (valid_ids,))
        cur.execute("DELETE FROM Citations WHERE citing_paper_id = ANY(%s) OR cited_paper_id = ANY(%s)", (valid_ids, valid_ids))

        cur.execute("DELETE FROM Papers WHERE paper_id = ANY(%s)", (valid_ids,))

        conn.commit()
        return jsonify({"success": True, "deleted_count": len(valid_ids)})

    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close() 
        conn.close()


if __name__ == "__main__":
    app.run(debug=True, port=5000)