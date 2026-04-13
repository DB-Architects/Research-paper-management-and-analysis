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