import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
INPUT_DATA_PATH = "data/processed/cleaned_graphene_data.csv"
OUTPUT_CANDIDATE_PATH = "data/labelled/trl_labeling_candidates.csv"
NUM_CANDIDATES = 600

def main():
    logging.info("Loading data from %s", INPUT_DATA_PATH)
    
    try:
        df = pd.read_csv(INPUT_DATA_PATH)
    except FileNotFoundError:
        logging.error(f"FATAL: Input data file not found at {INPUT_DATA_PATH}. Please ensure this file exists.")
        return
    
    logging.info(f"Original dataset: {len(df)} records")
    
    # --- CRITICAL FILTERING STEP ---
    # 1. Exclude patents by their source name.
    # 2. Ensure the abstract column is not empty/null and meaningful.
    df_filtered = df[
        (~df['source'].str.contains('patent', case=False, na=False)) & 
        (df['abstract'].notna()) &
        (df['abstract'].str.strip().str.len() > 20)  # Ensures abstract is meaningful
    ].copy()
    
    logging.info(f"After filtering: {len(df_filtered)} non-patent documents with abstracts")
    logging.info(f"Filtered out: {len(df) - len(df_filtered)} patents/short abstracts")
    
    # Check if we have enough data
    if len(df_filtered) < NUM_CANDIDATES:
        logging.warning(f"Only {len(df_filtered)} records available, sampling all of them")
        num_to_sample = len(df_filtered)
    else:
        num_to_sample = NUM_CANDIDATES
    
    # Sample from the clean, high-quality pool
    candidate_df = df_filtered.sample(n=num_to_sample, random_state=42)
    
    # Create doc_id if missing
    if 'doc_id' not in candidate_df.columns:
        logging.warning("Creating temporary 'doc_id' from DataFrame index")
        candidate_df['doc_id'] = [f"temp_id_{i}" for i in candidate_df.index]
    
    # Prepare the final dataset for labeling
    labeling_pool = candidate_df[['doc_id', 'title', 'abstract']].copy()
    labeling_pool['text_to_label'] = labeling_pool['title'].fillna('') + ". " + labeling_pool['abstract'].fillna('')
    
    # Save only what we need for labeling
    final_output = labeling_pool[['doc_id', 'text_to_label']].copy()
    final_output.to_csv(OUTPUT_CANDIDATE_PATH, index=False)
    
    logging.info(f"Generated {len(final_output)} high-quality candidates for manual labeling")
    logging.info(f"Candidates saved to: {OUTPUT_CANDIDATE_PATH}")
    logging.info("Script finished successfully.")

if __name__ == "__main__":
    main()