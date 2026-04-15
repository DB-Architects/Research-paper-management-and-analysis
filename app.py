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
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='mps')
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

    # Log search history
    if "user_id" in session and session.get("role") != "admin":
        try:
            conn_log = get_db()
            cur_log = conn_log.cursor()
            cur_log.execute("INSERT INTO Search_History (user_id, query, searched_at) VALUES (%s, %s, NOW())", (session["user_id"], query))
            conn_log.commit()
            cur_log.close(); conn_log.close()
        except Exception: pass

    conn = get_db()
    cur = conn.cursor()

    if sort == "relevance" or sort == "semantic":
        # --- ADVANCED DBMS+ML: Hybrid Search with Reciprocal Rank Fusion (RRF) ---
        query_vector = embedder.encode(query).tolist()
        
        cur.execute("""
            WITH semantic_search AS (
                -- 1. Get top 100 semantic matches
                SELECT p.paper_id,
                       ROW_NUMBER() OVER (ORDER BY pe.embedding <=> %s::vector) AS rank
                FROM Papers p
                JOIN Paper_Embeddings pe ON p.paper_id = pe.paper_id
                ORDER BY pe.embedding <=> %s::vector
                LIMIT 100
            ),
            keyword_search AS (
                -- 2. Get top 100 keyword matches
                SELECT p.paper_id,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', p.title || ' ' || p.abstract_text), plainto_tsquery('english', %s)) DESC) AS rank
                FROM Papers p
                WHERE to_tsvector('english', p.title || ' ' || p.abstract_text) @@ plainto_tsquery('english', %s)
                ORDER BY rank
                LIMIT 100
            )
            -- 3. Fuse the ranks using the RRF Formula: 1 / (k + rank) where k=60
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name,
                   COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS relevance_score,
                   LEFT(p.abstract_text, 200) AS snippet
            FROM Papers p
            LEFT JOIN semantic_search s ON p.paper_id = s.paper_id
            LEFT JOIN keyword_search k ON p.paper_id = k.paper_id
            JOIN Venues v ON p.venue_id = v.venue_id
            WHERE s.paper_id IS NOT NULL OR k.paper_id IS NOT NULL
            ORDER BY relevance_score DESC
            LIMIT %s OFFSET %s
        """, (query_vector, query_vector, query, query, per_page, offset))
        
        results = cur.fetchall()
        total = 200 # Cap pagination for Hybrid search since it blends two top-100 lists
        
    else:
        # --- Standard Keyword Searching for specific sorts (Year/Citations) ---
        order = "p.n_citations DESC" if sort == "citations" else "p.publication_year DESC"

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

        cur.execute("SELECT COUNT(*) AS cnt FROM Papers p WHERE to_tsvector('english', p.title || ' ' || p.abstract_text) @@ plainto_tsquery('english', %s)", (query,))
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

        is_uploader = (paper.get("uploaded_by") == session["user_id"])
        
        if not is_uploader:
            # 2. Time-gating: Only log a view if they haven't viewed it in the last 1 hour
            cur.execute("""
                SELECT 1 FROM Reading_History 
                WHERE user_id = %s AND paper_id = %s 
                AND viewed_at >= NOW() - INTERVAL '1 hour'
            """, (session["user_id"], paper_id))
            
            if not cur.fetchone():
                try:
                    cur.execute("INSERT INTO Reading_History (user_id, paper_id, viewed_at) VALUES (%s, %s, NOW())", 
                                (session["user_id"], paper_id))
                    conn.commit()
                except Exception:
                    conn.rollback()
    cur.execute("SELECT COUNT(*) AS total_views FROM Reading_History WHERE paper_id = %s", (paper_id,))
    total_views = cur.fetchone()["total_views"]

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

    # --- ADVANCED DBMS + ML: Vector Similarity Search ---
    # Find top 5 mathematically similar papers using pgvector's Cosine Distance (<=>)
    # --- ADVANCED DBMS + ML: Vector Similarity Search ---
    cur.execute("""
        WITH SemanticCandidates AS (
            SELECT p.paper_id, p.title, p.publication_year, v.venue_name, p.n_citations,
                   (1 - (pe.embedding <=> target.embedding))::numeric AS similarity_score
            FROM Papers p
            JOIN Paper_Embeddings pe ON p.paper_id = pe.paper_id
            LEFT JOIN Venues v ON p.venue_id = v.venue_id
            CROSS JOIN (SELECT embedding FROM Paper_Embeddings WHERE paper_id = %s) AS target
            WHERE p.paper_id != %s AND target.embedding IS NOT NULL
            ORDER BY pe.embedding <=> target.embedding
            LIMIT 50
        )
        SELECT paper_id, title, publication_year, venue_name, n_citations, similarity_score,
               -- Custom Ranking Algorithm: 70 percent Semantic Match + 30 percent Citation Weight
               ROUND((similarity_score * 0.7) + (LEAST(n_citations, 1000) / 1000.0 * 0.3), 3) AS hybrid_score
        FROM SemanticCandidates
        ORDER BY hybrid_score DESC
        LIMIT 5;
    """, (paper_id, paper_id))
    
    similar_papers = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS total_views FROM Reading_History WHERE paper_id = %s", (paper_id,))
    total_views = cur.fetchone()["total_views"]

    cur.close(); conn.close()

    scholar_url = f"https://scholar.google.com/scholar?q={paper['title'].replace(' ', '+')}"
    arxiv_url   = f"https://arxiv.org/search/?query={paper['title'].replace(' ', '+')}"

    return render_template("paper.html", paper=paper, authors=authors,
                           scholar_url=scholar_url, arxiv_url=arxiv_url, 
                           has_rated=has_rated, is_bookmarked=is_bookmarked,
                           user_note=user_note, citation_tree=citation_tree,
                           similar_papers=similar_papers, total_views=total_views)

@app.route("/for-you")
@login_required
def for_you():
    if session.get("role") == "admin": return redirect(url_for("admin_dashboard"))

    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT query, COUNT(*) AS freq FROM Search_History WHERE user_id = %s GROUP BY query ORDER BY freq DESC, MAX(searched_at) DESC LIMIT 5", (user_id,))
    top_queries = cur.fetchall()

    cur.execute("SELECT DISTINCT paper_id FROM Reading_History WHERE user_id = %s", (user_id,))
    read_ids = {r["paper_id"] for r in cur.fetchall()}

    cur.execute("SELECT paper_id FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmarked_ids = {r["paper_id"] for r in cur.fetchall()}

    already_seen = read_ids | bookmarked_ids
    recommendations = {}
    exclude_list = list(already_seen) if already_seen else [-1]

    # --- NOVEL SQL: Implicit User Affinity (Research DNA) ---
    # Complex UNION and JOIN to find which authors the user is implicitly following based on interactions
    cur.execute("""
        SELECT a.author_id, a.name, COUNT(DISTINCT p.paper_id) as interacted_papers, COALESCE(SUM(p.n_citations), 0) as impact
        FROM Authors a
        JOIN Paper_Authors pa ON a.author_id = pa.author_id
        JOIN Papers p ON pa.paper_id = p.paper_id
        JOIN (
            SELECT paper_id FROM Reading_History WHERE user_id = %s
            UNION
            SELECT paper_id FROM Bookmarks WHERE user_id = %s
        ) user_interactions ON p.paper_id = user_interactions.paper_id
        GROUP BY a.author_id, a.name
        ORDER BY interacted_papers DESC, impact DESC
        LIMIT 5;
    """, (user_id, user_id))
    affinity_authors = cur.fetchall()

    if top_queries:
        # --- ADVANCED ML: Re-Ranked Semantic Recommendations ---
        terms = [row["query"] for row in top_queries[:3]]
        combined_interest = " ".join(terms)
        interest_vector = embedder.encode(combined_interest).tolist()

        # Fetch Top 100 by vector distance, then re-rank with Citation Popularity
        cur.execute("""
            WITH SemanticCandidates AS (
                SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name, 
                       LEFT(p.abstract_text, 180) AS snippet,
                       (1 - (pe.embedding <=> %s::vector))::numeric AS similarity_score
                FROM Papers p 
                JOIN Venues v ON p.venue_id = v.venue_id
                JOIN Paper_Embeddings pe ON p.paper_id = pe.paper_id
                WHERE p.paper_id != ALL(%s)
                ORDER BY pe.embedding <=> %s::vector
                LIMIT 100
            )
            SELECT *,
                   ROUND((similarity_score * 0.7) + (LEAST(n_citations, 1000) / 1000.0 * 0.3), 3) AS hybrid_score
            FROM SemanticCandidates
            ORDER BY hybrid_score DESC
            LIMIT 12
        """, (interest_vector, exclude_list, interest_vector))

        for row in cur.fetchall():
            pid = row["paper_id"]
            recommendations[pid] = dict(row)
            recommendations[pid]["reason"] = "Matches your research interests & popularity"
            recommendations[pid]["reason_type"] = "search"

    # Fill remaining slots with Trending papers if needed
    if len(recommendations) < 12:
        exclude_all = list(already_seen | set(recommendations.keys())) or [-1]
        cur.execute("""
            SELECT p.paper_id, p.title, p.publication_year, p.n_citations, v.venue_name, 
                   LEFT(p.abstract_text, 180) AS snippet, p.n_citations::float AS hybrid_score
            FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
            WHERE p.paper_id != ALL(%s) ORDER BY p.n_citations DESC LIMIT %s
        """, (exclude_all, 12 - len(recommendations)))

        for row in cur.fetchall():
            pid = row["paper_id"]
            r = dict(row)
            r["reason"] = "Trending in the database"
            r["reason_type"] = "popular"
            recommendations[pid] = r

    # Sort final list by the new hybrid score
    sorted_recs = sorted(recommendations.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)[:12]

    cur.execute("SELECT COUNT(DISTINCT paper_id) AS cnt FROM Reading_History WHERE user_id = %s", (user_id,))
    read_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM Bookmarks WHERE user_id = %s", (user_id,))
    bookmark_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM Search_History WHERE user_id = %s", (user_id,))
    search_count = cur.fetchone()["cnt"]

    cur.close(); conn.close()
    
    from flask import make_response
    response = make_response(render_template("foryou.html", recommendations=sorted_recs, 
                           top_queries=top_queries, affinity_authors=affinity_authors,
                           read_count=read_count, bookmark_count=bookmark_count, 
                           search_count=search_count, has_history=bool(read_ids or top_queries)))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response    


# ── AUTH & ADMIN ───────────────────────────────────────────────────────────────

# Make sure BOTH of these are imported at the top of app.py!
from werkzeug.security import generate_password_hash, check_password_hash

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Handle both JSON (from fetch) and Form Data just in case
        data = request.get_json(silent=True) or request.form
        username = data.get("username", "").strip()
        password = data.get("password", "")

        # 1. Check Hardcoded Admin
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["user_id"] = 0  
            session["username"] = "admin"
            session["role"] = "admin"
            return jsonify({"success": True}) 

        # 2. Fetch User from DB (ONLY by username)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        # 3. Securely check the hashed password
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["role"] = user.get("role", "user")
            return jsonify({"success": True})

        # If user doesn't exist OR password doesn't match the hash:
        return jsonify({"success": False, "error": "Invalid credentials"})
        
    return render_template("login.html")

from werkzeug.security import generate_password_hash # Make sure you have this import at the top

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        role = "user"
        
        # safely extract the username
        username = data.get("username", "")
        
        # FIX 1 & 2: Check for spaces and return a consistent JSON response
        if " " in username:
            return jsonify({"success": False, "error": "Username cannot contain spaces. Use underscores or hyphens instead."})
            
        conn = get_db()
        cur = conn.cursor()
        try:
            # FIX 3: Hash the password securely before inserting it into the database
            hashed_pw = generate_password_hash(data.get("password"))
            
            cur.execute("""
                INSERT INTO Users (username, password_hash, gender, age, institute, role, trust_factor)
                VALUES (%s, %s, %s, %s, %s, %s, 5.0)
            """, (username, hashed_pw, data.get("gender"), data.get("age"), data.get("institute"), role))
            
            conn.commit()
            return jsonify({"success": True})
            
        except psycopg2.IntegrityError:
            conn.rollback() # Always rollback on error before returning
            return jsonify({"success": False, "error": "Username already taken"})
        finally:
            cur.close()
            conn.close()
            
    # GET request just serves the page
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# --- ADMIN DASHBOARD & DATA GOVERNANCE ---


@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db()
    cur = conn.cursor()
    
    # 1. Users and Papers for standard moderation
    cur.execute("SELECT user_id, username, role, institute, trust_factor FROM Users ORDER BY user_id DESC")
    users = cur.fetchall()
    
    cur.execute("SELECT paper_id, title, publication_year FROM Papers ORDER BY uploaded_at DESC NULLS LAST LIMIT 100")
    papers = cur.fetchall()

    # 2. READ: Global Business Intelligence Totals
    cur.execute("SELECT COUNT(*) AS cnt FROM Reading_History")
    total_views = cur.fetchone()["cnt"]
    
    cur.execute("SELECT COUNT(*) AS cnt FROM Search_History")
    total_searches = cur.fetchone()["cnt"]

    # 3. READ: Top Searched Trends
    cur.execute("""
        SELECT query, COUNT(*) as search_count
        FROM Search_History
        GROUP BY query
        ORDER BY search_count DESC
        LIMIT 20
    """)
    top_searches = cur.fetchall()

    # 4. READ: Deep Engagement (Papers with the most substantial notes)
    cur.execute("""
        SELECT p.paper_id, p.title, COUNT(pn.user_id) as note_count
        FROM Paper_Notes pn
        JOIN Papers p ON pn.paper_id = p.paper_id
        WHERE LENGTH(pn.note_text) > 15
        GROUP BY p.paper_id, p.title
        ORDER BY note_count DESC
        LIMIT 20
    """)
    top_annotated = cur.fetchall()

    cur.close(); conn.close()
    
    return render_template("admin.html", 
                           users=users, 
                           papers=papers,
                           total_views=total_views,
                           total_searches=total_searches,
                           top_searches=top_searches,
                           top_annotated=top_annotated)

# --- ADMIN HELPER ENDPOINTS ---

@app.route("/admin/delete-user/<int:del_uid>", methods=["DELETE"])
@admin_required
def admin_delete_user(del_uid):
    """DELETE: Completely wipe a user and their uploaded papers."""
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
            cur.execute("DELETE FROM Collection_Papers WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Notes WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Authors WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Citations WHERE citing_paper_id = ANY(%s) OR cited_paper_id = ANY(%s)", (user_papers, user_papers))
            cur.execute("DELETE FROM Papers WHERE paper_id = ANY(%s)", (user_papers,))
        
        cur.execute("DELETE FROM Paper_Ratings WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Bookmarks WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Reading_History WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Search_History WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Paper_Notes WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Collections WHERE user_id = %s", (del_uid,))
        cur.execute("DELETE FROM Users WHERE user_id = %s", (del_uid,))
        
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/admin/delete-paper/<int:pid>", methods=["DELETE"])
@admin_required
def admin_delete_paper(pid):
    """DELETE: Specifically for admins to remove rule-breaking papers."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM Paper_Ratings WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Paper_Embeddings WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Bookmarks WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Reading_History WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Collection_Papers WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Paper_Notes WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Paper_Authors WHERE paper_id = %s", (pid,))
        cur.execute("DELETE FROM Citations WHERE citing_paper_id = %s OR cited_paper_id = %s", (pid, pid))
        cur.execute("DELETE FROM Papers WHERE paper_id = %s", (pid,))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/admin/search-papers")
@admin_required
def admin_search_papers():
    """READ: Async endpoint for the Live Search bar."""
    query = request.args.get("q", "").strip()
    conn = get_db()
    cur = conn.cursor()
    
    if query:
        cur.execute("""
            SELECT paper_id, title, publication_year 
            FROM Papers WHERE title ILIKE %s ORDER BY paper_id DESC LIMIT 50
        """, (f"%{query}%",))
    else:
        cur.execute("SELECT paper_id, title, publication_year FROM Papers ORDER BY uploaded_at DESC NULLS LAST LIMIT 100")
        
    papers = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"papers": papers})
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

    cur.execute("""
        SELECT c.collection_id, c.name as collection_name,
               p.paper_id, p.title, p.publication_year, p.n_citations
        FROM Collections c
        LEFT JOIN Collection_Papers cp ON c.collection_id = cp.collection_id
        LEFT JOIN Papers p ON cp.paper_id = p.paper_id
        WHERE c.user_id = %s
        ORDER BY c.created_at DESC, cp.added_at DESC
    """, (session["user_id"],))

    collections_raw = cur.fetchall()

    my_collections = {}
    for row in collections_raw:
        cid = row["collection_id"]
        if cid not in my_collections:
            my_collections[cid] = {"collection_id": cid, "name": row["collection_name"], "papers": []}
        
        # If there is a paper attached to this row, add it to the list
        if row["paper_id"]: 
            my_collections[cid]["papers"].append(row)

    # Convert the dictionary to a list for Jinja2
    collections_list = list(my_collections.values())

    cur.execute("""
        SELECT pn.paper_id, pn.note_text, pn.last_updated, p.title
        FROM Paper_Notes pn
        JOIN Papers p ON pn.paper_id = p.paper_id
        WHERE pn.user_id = %s
        ORDER BY pn.last_updated DESC
    """, (session["user_id"],))
    my_notes = cur.fetchall()

    

    cur.close(); conn.close()


    return render_template("profile.html", user=user, history=history, 
                           bookmarks=bookmarks, search_history=search_history, 
                           my_uploads=my_uploads, collections=collections_list,
                           my_notes=my_notes)


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

    # 1. Global Views (For the Hero Banner)
    cur.execute("SELECT COUNT(*) FROM Reading_History")
    view_row = cur.fetchone()
    total_views = view_row[0] if isinstance(view_row, tuple) else view_row["count"]

    # 2. READ: Most Popular Venues (With Type)
    cur.execute("""
        SELECT v.venue_name, v.venue_type, COUNT(p.paper_id) as paper_count
        FROM Papers p JOIN Venues v ON p.venue_id = v.venue_id
        GROUP BY v.venue_name, v.venue_type 
        ORDER BY paper_count DESC LIMIT 10
    """)
    top_venues = []
    for r in cur.fetchall():
        name = r[0] if isinstance(r, tuple) else r["venue_name"]
        vtype = r[1] if isinstance(r, tuple) else r["venue_type"]
        count = r[2] if isinstance(r, tuple) else r["paper_count"]
        
        # Format the type nicely
        if vtype in ('C', 'Conference', 'conference'):
            type_label = "Conference"
        elif vtype in ('J', 'Journal', 'journal'):
            type_label = "Journal"
        else:
            type_label = "Unknown Type"
            
        top_venues.append({"name": name, "type": type_label, "count": count})

    # 3. READ: Most Cited Authors
    cur.execute("""
        SELECT a.name, SUM(p.n_citations) as total_cites
        FROM Authors a
        JOIN Paper_Authors pa ON a.author_id = pa.author_id
        JOIN Papers p ON pa.paper_id = p.paper_id
        GROUP BY a.name ORDER BY total_cites DESC NULLS LAST LIMIT 10
    """)
    top_cited_authors = [{"name": r[0] if isinstance(r, tuple) else r["name"], 
                          "count": int(r[1] if isinstance(r, tuple) else r["total_cites"] or 0)} for r in cur.fetchall()]

    # 4. READ: Most Viewed Authors
    cur.execute("""
        SELECT a.name, COUNT(rh.user_id) as view_count
        FROM Authors a
        JOIN Paper_Authors pa ON a.author_id = pa.author_id
        JOIN Reading_History rh ON pa.paper_id = rh.paper_id
        GROUP BY a.name ORDER BY view_count DESC LIMIT 10
    """)
    top_viewed_authors = [{"name": r[0] if isinstance(r, tuple) else r["name"], 
                           "count": int(r[1] if isinstance(r, tuple) else r["view_count"])} for r in cur.fetchall()]

    # 5. READ: Top 10 Most Viewed Papers (Leaderboard)
    cur.execute("""
        SELECT p.paper_id, p.title, p.publication_year, v.venue_name, 
               COUNT(rh.user_id) AS view_count
        FROM Papers p
        LEFT JOIN Venues v ON p.venue_id = v.venue_id
        JOIN Reading_History rh ON p.paper_id = rh.paper_id
        GROUP BY p.paper_id, p.title, p.publication_year, v.venue_name
        ORDER BY view_count DESC LIMIT 10
    """)
    top_viewed_papers = [{"paper_id": r[0] if isinstance(r, tuple) else r["paper_id"], 
                          "title": r[1] if isinstance(r, tuple) else r["title"], 
                          "publication_year": r[2] if isinstance(r, tuple) else r["publication_year"], 
                          "venue_name": r[3] if isinstance(r, tuple) else r["venue_name"], 
                          "view_count": r[4] if isinstance(r, tuple) else r["view_count"]} for r in cur.fetchall()]

    cur.close(); conn.close()

    return render_template("analytics.html",
        total_views=total_views,
        top_venues=top_venues,
        top_cited_authors=top_cited_authors,
        top_viewed_authors=top_viewed_authors,
        top_viewed_papers=top_viewed_papers
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

@app.route("/api/collections", methods=["GET", "POST"])
@login_required
def manage_collections():
    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        # CREATE a new collection
        data = request.get_json()
        name = data.get("name", "").strip()
        if not name: return jsonify({"success": False, "error": "Name required"})
        
        cur.execute("INSERT INTO Collections (user_id, name) VALUES (%s, %s) RETURNING collection_id", (user_id, name))
        new_id = cur.fetchone()["collection_id"]
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"success": True, "collection_id": new_id, "name": name})

    # GET all collections for this user
    cur.execute("SELECT collection_id, name FROM Collections WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    collections = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"collections": collections})

@app.route("/api/collections/<int:collection_id>/toggle", methods=["POST"])
@login_required
def toggle_collection_paper(collection_id):
    data = request.get_json()
    paper_id = data.get("paper_id")
    user_id = session["user_id"]
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Security check: Ensure this user owns the collection
        cur.execute("SELECT 1 FROM Collections WHERE collection_id = %s AND user_id = %s", (collection_id, user_id))
        if not cur.fetchone():
            return jsonify({"success": False, "error": "Unauthorized"}), 403

        # Check if paper is already in collection
        cur.execute("SELECT 1 FROM Collection_Papers WHERE collection_id = %s AND paper_id = %s", (collection_id, paper_id))
        if cur.fetchone():
            # DELETE operation
            cur.execute("DELETE FROM Collection_Papers WHERE collection_id = %s AND paper_id = %s", (collection_id, paper_id))
            action = "removed"
        else:
            # INSERT operation
            cur.execute("INSERT INTO Collection_Papers (collection_id, paper_id) VALUES (%s, %s)", (collection_id, paper_id))
            action = "added"
            
        conn.commit()
        return jsonify({"success": True, "action": action})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/api/collections/<int:collection_id>", methods=["DELETE"])
@login_required
def delete_collection(collection_id):
    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()
    try:
        # ADVANCED DBMS: Deleting the parent (Collections) automatically deletes 
        # the children (Collection_Papers) because of ON DELETE CASCADE.
        cur.execute("""
            DELETE FROM Collections 
            WHERE collection_id = %s AND user_id = %s 
            RETURNING collection_id
        """, (collection_id, user_id))
        
        if cur.fetchone():
            conn.commit()
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Unauthorized or not found"}), 403
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close()
        conn.close()

@app.route("/api/history/reading", methods=["DELETE"])
@login_required
def delete_reading_history():
    data = request.get_json()
    paper_id = data.get("paper_id")
    user_id = session["user_id"]
    
    conn = get_db()
    cur = conn.cursor()
    try:
        if paper_id == "all":
            cur.execute("DELETE FROM Reading_History WHERE user_id = %s", (user_id,))
        else:
            cur.execute("DELETE FROM Reading_History WHERE user_id = %s AND paper_id = %s", (user_id, paper_id))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/api/history/search", methods=["DELETE"])
@login_required
def delete_search_history():
    data = request.get_json()
    query = data.get("query")
    user_id = session["user_id"]
    
    conn = get_db()
    cur = conn.cursor()
    try:
        if query == "all":
            cur.execute("DELETE FROM Search_History WHERE user_id = %s", (user_id,))
        else:
            cur.execute("DELETE FROM Search_History WHERE user_id = %s AND query = %s", (user_id, query))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/api/user/settings", methods=["POST"])
@login_required
def update_user_settings():
    data = request.get_json()
    age = data.get("age")
    gender = data.get("gender")
    institute = data.get("institute", "").strip()
    
    conn = get_db()
    cur = conn.cursor()
    try:
        # ADVANCED DBMS: Standard UPDATE query for user profile data
        cur.execute("""
            UPDATE Users 
            SET age = %s, gender = %s, institute = %s 
            WHERE user_id = %s
        """, (age if age else None, gender, institute, session["user_id"]))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/api/user/delete_account", methods=["DELETE"])
@login_required
def delete_my_account():
    user_id = session["user_id"]
    conn = get_db()
    cur = conn.cursor()
    try:
        # ADVANCED DBMS: The ultimate DELETE operation. 
        # First, find all papers uploaded by the user and delete their dependencies.
        cur.execute("SELECT paper_id FROM Papers WHERE uploaded_by = %s", (user_id,))
        user_papers = [r["paper_id"] for r in cur.fetchall()]
        
        if user_papers:
            cur.execute("DELETE FROM Paper_Ratings WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Embeddings WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Bookmarks WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Reading_History WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Collection_Papers WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Notes WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Paper_Authors WHERE paper_id = ANY(%s)", (user_papers,))
            cur.execute("DELETE FROM Citations WHERE citing_paper_id = ANY(%s) OR cited_paper_id = ANY(%s)", (user_papers, user_papers))
            cur.execute("DELETE FROM Papers WHERE paper_id = ANY(%s)", (user_papers,))
        
        # Now delete the user's personal activity
        cur.execute("DELETE FROM Paper_Ratings WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM Bookmarks WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM Reading_History WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM Search_History WHERE user_id = %s", (user_id,))
        # Collections and Paper_Notes have ON DELETE CASCADE, but explicit deletion is safe
        cur.execute("DELETE FROM Paper_Notes WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM Collections WHERE user_id = %s", (user_id,))
        
        # Finally, delete the user
        cur.execute("DELETE FROM Users WHERE user_id = %s", (user_id,))
        
        conn.commit()
        session.clear() # Log them out
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "error": str(e)})
    finally:
        cur.close(); conn.close()

@app.route("/admin/update-trust/<int:uid>", methods=["POST"])
@admin_required
def admin_update_trust(uid):
    """UPDATE: Allows admin to manually adjust a user's Trust Factor (1-10)."""
    data = request.get_json()
    new_tf = data.get("trust_factor")
    
    if new_tf is None:
        return jsonify({"success": False, "error": "Missing trust factor."})
        
    try:
        # FIX: Parse as float first to handle values like "5.00"
        new_tf = float(new_tf)
        
        if new_tf < 1 or new_tf > 10:
            return jsonify({"success": False, "error": "Trust factor must be between 1 and 10."})
    except ValueError:
        return jsonify({"success": False, "error": "Invalid trust factor value."})

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE Users SET trust_factor = %s WHERE user_id = %s", (new_tf, uid))
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