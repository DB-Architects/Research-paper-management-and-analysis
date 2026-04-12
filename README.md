📚 Nexus: AI-Powered Research Discovery System
Built by: The DB Architects | IIT Kharagpur

Nexus is a full-stack research paper management system that uses Machine Learning to understand the semantic meaning of abstracts. It moves beyond simple keyword matching to provide AI-driven recommendations and local PDF hosting.

🛠️ Phase 1: Database Setup (The "One-Click" Schema)
Open pgAdmin or your terminal, connect to your server, and run this entire block to build the database architecture from scratch:

```sql
-- 1. Create Database
CREATE DATABASE dblp_project;

-- 2. Enable ML Vector Support (Run inside dblp_project)
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Academic Tables
CREATE TABLE Venues (
    venue_id BIGINT PRIMARY KEY,
    venue_name VARCHAR(255),
    venue_type CHAR(1) -- 'C' for Conference, 'J' for Journal
);

CREATE TABLE Papers (
    paper_id BIGINT PRIMARY KEY,
    title TEXT,
    publication_year INT,
    abstract_text TEXT,
    n_citations INT DEFAULT 0,
    venue_id BIGINT REFERENCES Venues(venue_id),
    uploaded_by INT, 
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pdf_path VARCHAR(255)
);

CREATE TABLE Authors (
    author_id BIGINT PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE Paper_Authors (
    paper_id BIGINT REFERENCES Papers(paper_id),
    author_id BIGINT REFERENCES Authors(author_id),
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE Citations (
    citing_paper_id BIGINT,
    cited_paper_id BIGINT,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

-- 4. User & Application Tables
CREATE TABLE Users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255),
    gender VARCHAR(50),
    age INT,
    institute VARCHAR(255)
);

CREATE TABLE Reading_History (
    log_id BIGSERIAL PRIMARY KEY,
    user_id INT REFERENCES Users(user_id),
    paper_id BIGINT,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Bookmarks (
    user_id INT REFERENCES Users(user_id),
    paper_id BIGINT,
    bookmarked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE Search_History (
    user_id INT REFERENCES Users(user_id),
    query TEXT,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Paper_Embeddings (
    paper_id BIGINT PRIMARY KEY REFERENCES Papers(paper_id),
    embedding vector(384) -- Matches all-MiniLM-L6-v2 dimensions
);

```

## ⚙️ Phase 2: Python Environment
Install all necessary libraries with a single command:

```bash
pip install -r requirements.txt


📂 Phase 3: The Data Pipeline
Nexus is powered by the DBLP Citation Network Dataset (v12). Run these scripts in order to populate your database:

python3 filter_data.py: Downloads the 12GB dataset, filters for Computer Science papers (2018 onwards), and cleans abstracts.

python3 populate_db.py: Executes high-performance batch insertion into your local Postgres tables.

python3 generate_embeddings.py: Runs the local ML model to create semantic AI vectors for the first 10,000 papers.

🚀 Phase 4: Run the Application
Start the Flask server:

python3 app.py


Open http://127.0.0.1:5000 in your browser to start exploring.

🌟 Key Features for Users:
Account System: Sign up to track your search history and reading logs.

Semantic Search: Find papers based on the actual meaning of your query, not just exact words.

AI Feed: Use the "For You" page to get personalized recommendations based on your reading history.

Paper Management: Upload your own research papers (with PDF support), edit details, or perform bulk deletions.

Analytics: View global publication trends and author impact via the interactive dashboard.
