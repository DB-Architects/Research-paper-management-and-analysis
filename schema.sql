-- 1. Create Database
DROP DATABASE dblp_project;
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
    institute VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user',         -- Admin/User Role Access
    trust_factor NUMERIC(4,2) DEFAULT 5.0    -- User Reputation System
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

-- Rating System Table
CREATE TABLE Paper_Ratings (
    user_id INT REFERENCES Users(user_id),
    paper_id BIGINT REFERENCES Papers(paper_id),
    rating INT CHECK (rating >= 1 AND rating <= 10),
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE Paper_Embeddings (
    paper_id BIGINT PRIMARY KEY REFERENCES Papers(paper_id),
    embedding vector(384) -- Matches all-MiniLM-L6-v2 dimensions
);

CREATE TABLE Paper_Notes (
    user_id INT REFERENCES Users(user_id) ON DELETE CASCADE,
    paper_id BIGINT REFERENCES Papers(paper_id) ON DELETE CASCADE,
    note_text TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE Collections (
    collection_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES Users(user_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Collection_Papers (
    collection_id INT REFERENCES Collections(collection_id) ON DELETE CASCADE,
    paper_id BIGINT REFERENCES Papers(paper_id) ON DELETE CASCADE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (collection_id, paper_id)
);

-- Speed up sorting by citations (used on home page and search)
CREATE INDEX idx_papers_citations ON Papers(n_citations DESC);

-- Speed up filtering and sorting by year
CREATE INDEX idx_papers_year ON Papers(publication_year DESC);

-- Speed up JOINs between Papers and Venues
CREATE INDEX idx_papers_venue_id ON Papers(venue_id);

-- Speed up JOINs for the Authors relationship
CREATE INDEX idx_pa_paper_id ON Paper_Authors(paper_id);
CREATE INDEX idx_pa_author_id ON Paper_Authors(author_id);

-- Create a GIN index for lightning-fast keyword search on title and abstract
CREATE INDEX idx_papers_fts ON Papers USING GIN (to_tsvector('english', title || ' ' || abstract_text));

-- Materialized View for Top Venues
CREATE MATERIALIZED VIEW mv_top_venues AS
SELECT v.venue_name, COUNT(p.paper_id) as total_papers, 
       COALESCE(AVG(p.n_citations), 0) as avg_citations, 
       COALESCE(SUM(p.n_citations), 0) as total_citations
FROM Venues v
JOIN Papers p ON v.venue_id = p.venue_id
GROUP BY v.venue_id, v.venue_name
ORDER BY avg_citations DESC
LIMIT 10;

-- Materialized View for Top Authors
CREATE MATERIALIZED VIEW mv_top_authors AS
SELECT a.author_id, a.name, COUNT(pa.paper_id) as total_papers, 
       COALESCE(SUM(p.n_citations), 0) as total_citations
FROM Authors a
JOIN Paper_Authors pa ON a.author_id = pa.author_id
JOIN Papers p ON pa.paper_id = p.paper_id
GROUP BY a.author_id, a.name
ORDER BY total_citations DESC
LIMIT 10;