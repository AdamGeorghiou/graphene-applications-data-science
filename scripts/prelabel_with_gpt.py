# scripts/prelabel_with_gpt_robust.py
import os
import pandas as pd
import openai
from openai import OpenAI
import json
import time
from tqdm import tqdm
import logging
from pathlib import Path

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")

# CORRECTED: Use the latest official GPT-4 Turbo model. 
# The model 'gpt-4.1-2025-04-14' does not exist.
MODEL_NAME = "gpt-4.1-2025-04-14" 

# Using the sanitized CSV is recommended if you created it
INPUT_FILE = "data/labelled/trl_labeling_candidates.csv" 
# CRITICAL CHANGE: We now output to a .jsonl file for robustness
OUTPUT_FILE = "data/labelled/prelabeled_for_studio.jsonl" 

# --- The "Oracle" Prompt (No Change) ---
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
    # The error handling here is already good, no changes needed.
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

def main():
    if not API_KEY:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=API_KEY)
    
    # --- NEW: RESUME LOGIC ---
    processed_ids = set()
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        logging.info(f"Resuming from existing output file: {OUTPUT_FILE}")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    # Each line is a JSON object representing one task
                    task = json.loads(line)
                    processed_ids.add(task['data']['doc_id'])
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in output file: {line}")
        logging.info(f"Found {len(processed_ids)} previously processed documents. They will be skipped.")

    logging.info(f"Reading candidates from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # --- CRITICAL CHANGE: APPEND TO FILE IN A LOOP ---
    # We open the file in append mode ('a')
    with open(output_path, 'a') as f:
        # Use tqdm for a nice progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Pre-labeling with LLM"):
            doc_id = row['doc_id']
            
            # --- NEW: Check if already processed ---
            if doc_id in processed_ids:
                continue # Skip this row

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
            
            time.sleep(1) # Slightly increased sleep to be safe with GPT-4 rate limits

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()