# scripts/label_all_papers_with_gpt.py
import os
import pandas as pd
import openai
from openai import OpenAI
import json
import time
from tqdm import tqdm
import logging
from pathlib import Path

# --- Configuration (Using proven settings) ---
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-2025-04-14"  # The model that worked well for you

# File paths
INPUT_DATA_PATH = "data/processed/cleaned_graphene_data.csv"
EXISTING_LABELS_FILE = "data/labelled/prelabeled_for_studio.jsonl"  # Your 600 existing labels
FINAL_OUTPUT_FILE = "data/labelled/all_papers_gpt_labels.jsonl"     # Final combined output

# --- The Proven System Prompt ---
SYSTEM_PROMPT = """
You are an expert in materials science and technology commercialization. Your task is to classify scientific abstracts into one of three Technology Readiness Level (TRL) buckets. You must respond with only one of the following three labels: 'Early Stage (TRL 1-3)', 'Mid Stage (TRL 4-6)', or 'Late Stage (TRL 7-9)'. Do not provide any explanation or other text.
Here are the definitions for each bucket:
- 'Early Stage (TRL 1-3)': Basic research and feasibility. Focus on theoretical studies, simulations, fundamental properties, initial synthesis, or proof-of-concept. The work is foundational.
- 'Mid Stage (TRL 4-6)': Lab-scale technology development. Focus on creating a lab prototype, device fabrication, performance testing, and validation in a controlled or simulated lab environment. The work is about building and testing one.
- 'Late Stage (TRL 7-9)': Demonstration and commercialization. Focus on prototypes in a real-world operational environment, pilot-scale production, manufacturing scale-up, or patents describing a specific invention for market. The work is about real-world application and manufacturing.
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_trl_label(client, abstract_text: str) -> str:
    """Sends a request to the OpenAI API and returns the predicted label."""
    # This is the exact function that worked well for you
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": abstract_text}
            ],
            temperature=0.0,
            max_tokens=20, # Increased slightly just in case
        )
        label = response.choices[0].message.content.strip()
        valid_labels = ['Early Stage (TRL 1-3)', 'Mid Stage (TRL 4-6)', 'Late Stage (TRL 7-9)']
        if label in valid_labels:
            return label
        else:
            logging.warning(f"LLM returned an invalid label: '{label}'. Skipping.")
            return None
    except openai.APIError as e:
        logging.error(f"OpenAI API returned an API Error: {e}. Skipping.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}. Skipping.")
        return None

def create_full_candidate_list():
    """Creates the full list of papers using the proven filtering logic."""
    logging.info("Loading data from %s", INPUT_DATA_PATH)
    
    try:
        df = pd.read_csv(INPUT_DATA_PATH)
    except FileNotFoundError:
        logging.error(f"FATAL: Input data file not found at {INPUT_DATA_PATH}. Please ensure this file exists.")
        return None

    logging.info(f"Original dataset: {len(df)} records")
    
    # --- PROVEN FILTERING LOGIC (exactly as it worked) ---
    # 1. Exclude patents by their source name.
    # 2. Ensure the abstract column is not empty/null and meaningful.
    df_filtered = df[
        (~df['source'].str.contains('patent', case=False, na=False)) & 
        (df['abstract'].notna()) &
        (df['abstract'].str.strip().str.len() > 20)  # Ensures abstract is meaningful
    ].copy()
    
    logging.info(f"After filtering: {len(df_filtered)} non-patent documents with abstracts")
    logging.info(f"Filtered out: {len(df) - len(df_filtered)} patents/short abstracts")
    
    # Create doc_id if missing (same logic that worked)
    if 'doc_id' not in df_filtered.columns:
        logging.warning("Creating temporary 'doc_id' from DataFrame index")
        df_filtered['doc_id'] = [f"temp_id_{i}" for i in df_filtered.index]
    
    # Prepare the text_to_label column (same logic that worked)
    df_filtered['text_to_label'] = df_filtered['title'].fillna('') + ". " + df_filtered['abstract'].fillna('')
    
    return df_filtered[['doc_id', 'text_to_label']].copy()

def main():
    if not API_KEY:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=API_KEY)
    
    # Step 1: Create the full candidate list using proven logic
    logging.info("Creating full candidate list using proven filtering logic...")
    df_all_candidates = create_full_candidate_list()
    if df_all_candidates is None:
        return
    
    logging.info(f"Total papers to potentially process: {len(df_all_candidates):,}")
    
    # Step 2: Load existing labels to avoid duplicates (using proven resume logic)
    processed_ids = set()
    
    # Load from existing labels file (your 600 samples)
    existing_path = Path(EXISTING_LABELS_FILE)
    if existing_path.exists():
        logging.info(f"Loading existing labels from {EXISTING_LABELS_FILE}")
        with open(existing_path, 'r') as f:
            for line in f:
                try:
                    task = json.loads(line)
                    processed_ids.add(task['data']['doc_id'])
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in existing file: {line}")
        logging.info(f"Found {len(processed_ids)} existing labels from previous run")
    
    # Load from final output file (for resume capability)
    output_path = Path(FINAL_OUTPUT_FILE)
    if output_path.exists():
        logging.info(f"Resuming from existing output file: {FINAL_OUTPUT_FILE}")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    task = json.loads(line)
                    processed_ids.add(task['data']['doc_id'])
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in output file: {line}")
        logging.info(f"Total processed documents found: {len(processed_ids)}")
    
    # Step 3: Copy existing labels to final output file if starting fresh
    if not output_path.exists() and existing_path.exists():
        logging.info("Copying existing labels to final output file...")
        with open(output_path, 'w') as out_f:
            with open(existing_path, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
        logging.info("Existing labels copied successfully")
    
    # Step 4: Filter out already processed papers
    df_remaining = df_all_candidates[~df_all_candidates['doc_id'].isin(processed_ids)].copy()
    logging.info(f"Papers already processed: {len(processed_ids):,}")
    logging.info(f"Remaining papers to process: {len(df_remaining):,}")
    
    if len(df_remaining) == 0:
        logging.info("All papers have already been processed!")
        logging.info(f"Final results are in: {FINAL_OUTPUT_FILE}")
        return
    
    # Calculate cost estimate based on your empirical data
    cost_per_sample = 0.44 / 600  # Based on your actual costs
    estimated_cost = len(df_remaining) * cost_per_sample
    logging.info(f"Estimated cost for remaining {len(df_remaining):,} papers: ${estimated_cost:.2f}")
    
    # Step 5: Process remaining papers (using proven approach)
    successful_labels = 0
    with open(output_path, 'a') as f:  # Append mode for resume capability
        for index, row in tqdm(df_remaining.iterrows(), total=len(df_remaining), desc="Pre-labeling with LLM"):
            doc_id = row['doc_id']
            text = row['text_to_label']
            
            predicted_label = get_trl_label(client, text)
            
            if predicted_label:
                task = {
                    "data": {"text_to_label": text, "doc_id": doc_id},
                    "predictions": [{
                        "model_version": MODEL_NAME,
                        "result": [{"from_name": "choice", "to_name": "text", "type": "choices", "value": {"choices": [predicted_label]}}]
                    }]
                }
                # Write this single result as a new line in the file
                f.write(json.dumps(task) + '\n')
                successful_labels += 1
                
                # Log progress periodically
                if successful_labels % 100 == 0:
                    logging.info(f"Successfully processed {successful_labels} new papers...")
            
            time.sleep(1)  # Proven rate limiting that worked well
    
    # Final summary
    total_processed = len(processed_ids) + successful_labels
    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE!")
    logging.info(f"Previously processed papers: {len(processed_ids):,}")
    logging.info(f"Newly processed papers: {successful_labels:,}")
    logging.info(f"Total papers with TRL labels: {total_processed:,}")
    logging.info(f"Final results saved to: {FINAL_OUTPUT_FILE}")
    logging.info("Script finished successfully.")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()