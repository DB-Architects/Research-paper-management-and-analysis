import ijson
import json
import time
import kagglehub
import os
import glob

# --- CONFIGURATION & FILE LOCATING ---
print("Locating dataset via kagglehub...")
path = kagglehub.dataset_download("mathurinache/citation-network-dataset")
json_files = glob.glob(os.path.join(path, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON file found in the Kaggle download folder.")

INPUT_FILE = json_files[0] # Automatically uses the hidden Kaggle path!
OUTPUT_FILE = "dblp_filtered.jsonl" 
TARGET_YEAR = 2018
TARGET_FOS = {
    "Computer science", "Database", "Machine learning", 
    "Artificial intelligence", "Data mining", "Information retrieval"
}

def reconstruct_abstract(indexed_abstract):
    """Reconstructs the original abstract text from the DBLP inverted index."""
    if not indexed_abstract or 'IndexLength' not in indexed_abstract or 'InvertedIndex' not in indexed_abstract:
        return None
        
    length = indexed_abstract.get('IndexLength', 0)
    if length == 0:
        return None
        
    words = [""] * length
    for word, positions in indexed_abstract.get('InvertedIndex', {}).items():
        for pos in positions:
            if pos < length:
                words[pos] = word
                
    return " ".join(words).strip()

def is_valid_paper(obj):
    """Determines if a paper meets the strict criteria for the project."""
    if obj.get("year", 0) < TARGET_YEAR:
        return False
        
    if not obj.get("references") or not obj.get("authors"):
        return False
        
    if not obj.get("indexed_abstract"):
        return False
        
    fos_list = obj.get("fos", [])
    has_relevant_fos = False
    for fos in fos_list:
        if isinstance(fos, dict) and fos.get("name") in TARGET_FOS:
            has_relevant_fos = True
            break
            
    if not has_relevant_fos:
        return False
        
    return True

def process_dataset():
    print(f"Starting to process {INPUT_FILE}...")
    start_time = time.time()
    processed_count = 0
    saved_count = 0

    with open(INPUT_FILE, "r",encoding ='utf-8') as infile, open(OUTPUT_FILE, "w") as outfile:
        objects = ijson.items(infile, "item")
        
        for obj in objects:
            processed_count += 1
            
            if is_valid_paper(obj):
                abstract_text = reconstruct_abstract(obj.get("indexed_abstract"))
                
                if abstract_text:
                    cleaned_paper = {
                        "id": obj.get("id"),
                        "title": obj.get("title"),
                        "year": obj.get("year"),
                        "authors": [{"id": a.get("id"), "name": a.get("name")} for a in obj.get("authors", [])],
                        "venue": obj.get("venue", {}).get("raw", "Unknown"),
                        "venue_type": obj.get("venue", {}).get("type", "Unknown"),
                        "n_citation": obj.get("n_citation", 0),
                        "references": obj.get("references", []),
                        "abstract": abstract_text
                    }
                    
                    outfile.write(json.dumps(cleaned_paper) + "\n")
                    saved_count += 1
            
            if processed_count % 100000 == 0:
                print(f"Processed: {processed_count:,} | Saved: {saved_count:,} | Time: {round(time.time() - start_time, 2)}s")

    print("\n--- DONE ---")
    print(f"Total Processed: {processed_count:,}")
    print(f"Total Saved: {saved_count:,}")
    print(f"Filtered dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()