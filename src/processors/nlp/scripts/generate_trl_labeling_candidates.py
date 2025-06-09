# scripts/generate_trl_labeling_candidates.py
import pandas as pd
from src.processors.nlp.document_processor import GrapheneDocumentProcessor
from tqdm import tqdm
import logging
import torch

# --- Configuration ---
# Use your main cleaned CSV file as the source
INPUT_DATA_PATH = "data/processed/cleaned_graphene_data.csv"
OUTPUT_CANDIDATE_PATH = "data/labelled/trl_labeling_candidates.csv"
SAMPLES_PER_CATEGORY = 200 # Aim for ~600 total candidates to label
INITIAL_SAMPLE_SIZE = 5000 # Process this many random docs to find candidates
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_trl_to_category(trl_val: int) -> int:
    """Maps a 1-9 TRL to a 0-2 category index."""
    if 1 <= trl_val <= 3: return 0  # Early
    if 4 <= trl_val <= 6: return 1  # Mid
    if 7 <= trl_val <= 9: return 2  # Late
    return -1 # Invalid or unclassified

def main():
    logging.info("Loading data from %s", INPUT_DATA_PATH)
    df = pd.read_csv(INPUT_DATA_PATH)

    # Use a manageable slice for this process to speed it up
    df_sample = df.sample(n=min(len(df), INITIAL_SAMPLE_SIZE), random_state=42).copy()
    
    logging.info("Initializing GrapheneDocumentProcessor to generate weak TRL labels...")
    # Note: We are using the OLD TRLAssessor here to generate weak labels.
    # This is fine, as it's just for creating a balanced candidate pool.
    processor = GrapheneDocumentProcessor(use_gpu=(DEVICE != "cpu"), device=DEVICE)
    
    texts = (df_sample['title'].fillna('') + ". " + df_sample['abstract'].fillna('')).tolist()
    sources = df_sample['source'].tolist()
    
    logging.info(f"Processing {len(texts)} documents to get weak TRL labels...")
    results = processor.trl_assessor.process_batch(texts, batch_source_types=sources)
    
    # In your old TRL assessor, the result is in 'trl_level'
    df_sample['weak_trl'] = [res.get('trl_level', -1) for res in results]
    df_sample['weak_trl_category'] = df_sample['weak_trl'].apply(map_trl_to_category)

    logging.info("\nWeak label distribution found:")
    print(df_sample['weak_trl_category'].value_counts().sort_index())

    logging.info(f"\nSampling up to {SAMPLES_PER_CATEGORY} candidates per category...")
    
    # Use stratified sampling to get a balanced set for labeling
    candidate_df = df_sample.groupby('weak_trl_category').apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_CATEGORY), random_state=42)
    ).reset_index(drop=True)
    
    # Prepare the final CSV for labeling tools
    # The 'doc_id' is crucial if you need to merge back later.
    labeling_pool = candidate_df[['doc_id', 'title', 'abstract']].copy()
    labeling_pool['text_to_label'] = labeling_pool['title'].fillna('') + ". " + labeling_pool['abstract'].fillna('')
    
    logging.info(f"\nGenerated {len(labeling_pool)} candidates for manual labeling.")
    labeling_pool[['doc_id', 'text_to_label']].to_csv(OUTPUT_CANDIDATE_PATH, index=False)
    logging.info(f"Candidates ready for labeling at: {OUTPUT_CANDIDATE_PATH}")

if __name__ == "__main__":
    import torch
    main()