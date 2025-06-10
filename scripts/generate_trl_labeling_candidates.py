# scripts/generate_trl_labeling_candidates.py
import pandas as pd
import logging

# --- Configuration ---
INPUT_DATA_PATH = "data/processed/full_cleaned_data.csv"
OUTPUT_CANDIDATE_PATH = "data/labelled/trl_labeling_candidates.csv"
NUM_CANDIDATES = 600 # Total number of random documents to select for labeling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Loading data from %s", INPUT_DATA_PATH)
    try:
        df = pd.read_csv(INPUT_DATA_PATH)
    except FileNotFoundError:
        logging.error(f"FATAL: Input data file not found at {INPUT_DATA_PATH}. Please ensure this file exists.")
        return

    # --- FIX for the KeyError ---
    # The previous crash was due to a missing 'doc_id' column.
    # We will now check for it and create it if it's missing.
    if 'doc_id' not in df.columns:
        logging.warning("Warning: 'doc_id' column not found in the input CSV.")
        logging.warning("Creating a temporary 'doc_id' from the DataFrame index for traceability.")
        df['doc_id'] = [f"temp_id_{i}" for i in df.index]

    # --- FIX for the slow, ineffective weak labeling ---
    # The old method was not working. A simple random sample is faster
    # and still achieves the primary goal: getting a pool of documents to label.
    logging.info(f"Selecting {NUM_CANDIDATES} random candidates for the labeling pool...")
    
    # Ensure we don't try to sample more rows than exist
    num_to_sample = min(len(df), NUM_CANDIDATES)
    candidate_df = df.sample(n=num_to_sample, random_state=42)

    # Prepare the final CSV for labeling tools
    labeling_pool = candidate_df[['doc_id', 'title', 'abstract']].copy()
    labeling_pool['text_to_label'] = labeling_pool['title'].fillna('') + ". " + labeling_pool['abstract'].fillna('')
    
    logging.info(f"Generated {len(labeling_pool)} candidates for manual labeling.")
    
    # We only need the text and an ID, so we simplify the output file.
    labeling_pool[['doc_id', 'text_to_label']].to_csv(OUTPUT_CANDIDATE_PATH, index=False)
    logging.info(f"Candidates ready for labeling at: {OUTPUT_CANDIDATE_PATH}")
    logging.info("Script finished successfully.")

if __name__ == "__main__":
    main()