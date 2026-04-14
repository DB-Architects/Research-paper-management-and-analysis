# Research Portal: Research Paper Management & Analytics System

**Research Portal** is an intelligent, full-stack research paper database system that seamlessly integrates advanced relational database techniques with machine learning analytics. Built for researchers, it allows users to discover relevant literature, explore multi-hop citation networks, organize collections, and receive AI-powered semantic recommendations. 

Built by **The DB Architects**.

---

## 🚀 Key DBMS & ML Features

### Advanced Database Concepts
* **Recursive CTEs (Graph Traversal):** Utilizes `WITH RECURSIVE` queries to dynamically traverse multi-hop citation chains (up to 3 levels deep).
* **Window Functions:** Employs `ROW_NUMBER() OVER (PARTITION BY ...)` to accurately rank and limit citation graphs by depth level.
* **ACID Transactions & Strict Concurrency:** Uses `SERIALIZABLE` isolation levels and explicit `BEGIN/COMMIT` blocks during multi-table paper uploads to prevent dirty or phantom reads.
* **Upserts (`ON CONFLICT DO UPDATE`):** Manages user private annotations efficiently without race conditions.
* **Cascading Deletes (`ON DELETE CASCADE`):** Maintains referential integrity across Many-to-Many relationships (e.g., Folders/Collections and User Accounts).
* **Materialized Views:** Pre-computes expensive aggregations (Top Venues, Top Authors) for instantaneous analytics loading.
* **Indexing Strategies:** Employs B-Tree indexes for relational sorting and GIN indexes for Full-Text Search.

### Machine Learning Integration
* **Vector Database (`pgvector`):** Stores 384-dimensional embeddings for research paper abstracts directly inside PostgreSQL.
* **Hybrid Search (Reciprocal Rank Fusion):** Combines Lexical Keyword Search (`to_tsvector`) with Semantic Vector Search (Cosine Distance) to deliver state-of-the-art query results.
* **Two-Stage Re-Ranking Pipeline:** For personalized recommendations, the database acts as a candidate generator (fetching top semantic matches) and then re-ranks them based on citation popularity.
* **Implicit User Affinity ("Research DNA"):** Uses complex `UNION` and `JOIN` combinations on user history logs to deduce implicit author networks.

---

## 🗄️ Database Schema Details

The database is highly normalized (3NF) to ensure referential integrity and minimize redundancy[cite: 13, 30].

### 1. Academic Entities
* **`Venues`**: Stores `venue_id`, `venue_name`, and `venue_type` ('C' for Conference, 'J' for Journal)[cite: 13].
* **`Papers`**: Core entity storing `paper_id`, `title`, `publication_year`, `abstract_text`, `n_citations`, `venue_id` (FK), and upload metadata.
* **`Authors`**: Stores `author_id` and `name`.
* **`Paper_Authors`**: Junction table (Many-to-Many) linking Papers and Authors.
* **`Citations`**: Directed graph edges storing `citing_paper_id` and `cited_paper_id`.

### 2. Machine Learning Entities
* **`Paper_Embeddings`**: Stores `paper_id` (PK/FK) and `embedding` (`vector(384)`).

### 3. User & Application Entities
* **`Users`**: Stores authentication, demographics, `role` (Admin/User), and dynamic `trust_factor`.
* **`Reading_History` / `Search_History`**: Interaction logs for analytics and ML personalization.
* **`Bookmarks`**: User-saved papers.
* **`Paper_Ratings`**: Implements `CHECK` constraints (1-10) to dynamically adjust uploader Trust Factors.
* **`Paper_Notes`**: Private user annotations (Upsert enabled).
* **`Collections` / `Collection_Papers`**: Custom reading folders with Many-to-Many relationships and cascading deletes.

---

## 💻 Installation & Execution Steps

### Prerequisites
1. **Python 3.9+**
2. **PostgreSQL 16(don't use postgreSQL 18 as pgvector extension not supported)**
3. **pgvector extension** (Must be installed in your PostgreSQL instance) 

### Step 1: Database Setup
1. Log into your PostgreSQL terminal.
2. Create the database and enable the vector extension:
   ```sql
   CREATE DATABASE dblp_project;
   \c dblp_project
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Execute the schema SQL to create all tables, indexes, and materialized views.
4. Import your DBLP/arXiv JSONL data into the relational tables. *(Ensure `user_id = 0` exists for admin uploads).*

### Step 2: Environment Setup
1. Clone the repository and navigate to the project folder.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install flask psycopg2-binary sentence-transformers werkzeug
   ```

### Step 3: Generate ML Vector Embeddings
Before running the web application, you must embed the research abstracts.
1. Open `generate_embeddings.py` and ensure the database credentials match your local setup.
2. Run the script. *(Note: The first run will download the ~90MB `BAAI/bge-small-en-v1.5` model).*
   ```bash
   python generate_embeddings.py
   ```
   *Tip: If using an Apple Silicon Mac, ensure `device='mps'` is set in the model initialization for GPU acceleration.*

### Step 4: Run the Application
1. Open `app.py` and verify your database credentials in the `DB_PARAMS` dictionary.
2. Start the Flask development server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://localhost:5000`.

### Default Admin Credentials
To access the Admin Dashboard for user moderation and venue standardization:
* **Username:** `admin`
* **Password:** `admin123`

---

## 📂 Project Structure

```text
├── app.py                     # Main Flask application and API routes
├── generate_embeddings.py     # ML script for populating the pgvector database
├── static/
│   └── uploads/papers/        # Directory for user-uploaded PDF files
└── templates/
    ├── base.html              # Master layout and global navigation
    ├── index.html             # Homepage and global search
    ├── search.html            # Hybrid Search (Keyword + Semantic) results
    ├── paper.html             # Paper details, notes, citations, and ML recommendations
    ├── profile.html           # User dashboard (History, Folders, Uploads, Settings)
    ├── foryou.html            # Personalized feed based on Implicit Affinity & ML
    ├── analytics.html         # Materialized View charts and database statistics
    ├── admin.html             # Admin dashboard for DML overrides
    └── ...
```
