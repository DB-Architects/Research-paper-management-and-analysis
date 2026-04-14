import psycopg2
from psycopg2.extras import execute_values
import json

# --- POINT THIS TO SCRIPT's OUTPUT FILE ---
FILTERED_DATA_FILE = "dblp_filtered.jsonl"

DB_PARAMS = {
    "dbname": "dblp_project",
    "user": "siddharthkonnur",
    "password": "", # Change ts gng🥀
    "host": "localhost",
    "port": "5432"
}

BATCH_SIZE = 5000

def insert_batches(conn, venues_batch, authors_batch, papers_batch, paper_authors_batch, citations_batch):
    with conn.cursor() as cur:
        if venues_batch:
            execute_values(cur, """
                INSERT INTO Venues (venue_id, venue_name, venue_type)
                VALUES %s ON CONFLICT (venue_id) DO NOTHING
            """, venues_batch)
        if authors_batch:
            execute_values(cur, """
                INSERT INTO Authors (author_id, name)
                VALUES %s ON CONFLICT (author_id) DO NOTHING
            """, authors_batch)
        if papers_batch:
            execute_values(cur, """
                INSERT INTO Papers (paper_id, title, publication_year, abstract_text, n_citations, venue_id)
                VALUES %s ON CONFLICT (paper_id) DO NOTHING
            """, papers_batch)
        if paper_authors_batch:
            execute_values(cur, """
                INSERT INTO Paper_Authors (paper_id, author_id)
                VALUES %s ON CONFLICT (paper_id, author_id) DO NOTHING
            """, paper_authors_batch)
        if citations_batch:
            execute_values(cur, """
                INSERT INTO Citations (citing_paper_id, cited_paper_id)
                VALUES %s ON CONFLICT (citing_paper_id, cited_paper_id) DO NOTHING
            """, citations_batch)
    conn.commit()

def populate_database():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_PARAMS)
    
    seen_authors = set()
    venues_map = {} 
    next_venue_id = 1
    
    venues_batch, authors_batch, papers_batch = [], [], []
    paper_authors_batch, citations_batch = [], []
    
    count = 0
    print(f"Reading from {FILTERED_DATA_FILE}...")
    
    # Read the JSONL file line by line
    with open(FILTERED_DATA_FILE, 'r') as f:
        for line in f:
            paper = json.loads(line)
            
            p_id = int(paper['id'])
            
            # 1. Venues
            venue_name = paper.get('venue', 'Unknown')[:255]
            if venue_name not in venues_map:
                venues_map[venue_name] = next_venue_id
                v_type = 'J' if 'journal' in venue_name.lower() else 'C'
                venues_batch.append((next_venue_id, venue_name, v_type))
                next_venue_id += 1
            v_id = venues_map[venue_name]

            # 2. Papers
            papers_batch.append((
                p_id, 
                paper.get('title', ''), 
                paper.get('year'), 
                paper.get('abstract', ''), 
                paper.get('n_citation', 0), 
                v_id
            ))
            
            # 3. Authors & Paper_Authors
            for author in paper.get('authors', []):
                try:
                    a_id = int(author['id'])
                    if a_id not in seen_authors:
                        seen_authors.add(a_id)
                        authors_batch.append((a_id, author.get('name', '')[:255]))
                    paper_authors_batch.append((p_id, a_id))
                except (ValueError, KeyError):
                    continue
            
            # 4. Citations
            for ref_id in paper.get('references', []):
                try:
                    citations_batch.append((p_id, int(ref_id)))
                except ValueError:
                    continue
            
            count += 1
            
            # Execute batch
            if len(papers_batch) >= BATCH_SIZE:
                insert_batches(conn, venues_batch, authors_batch, papers_batch, paper_authors_batch, citations_batch)
                print(f"Inserted {count} filtered papers...")
                venues_batch.clear(); authors_batch.clear(); papers_batch.clear()
                paper_authors_batch.clear(); citations_batch.clear()
                
        # Insert any leftovers
        if papers_batch:
            insert_batches(conn, venues_batch, authors_batch, papers_batch, paper_authors_batch, citations_batch)
            print(f"Final insert complete. Total processed: {count}")

    conn.close()

if __name__ == "__main__":
    populate_database() 