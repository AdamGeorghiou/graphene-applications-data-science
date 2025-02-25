import os
import pandas as pd
import json

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nlp_results_path = os.path.join(project_root, 'data', 'nlp_results', 'nlp_results.json')
topics_path = os.path.join(project_root, 'data', 'nlp_results', 'documents_with_topics.csv')
relationships_path = os.path.join(project_root, 'data', 'nlp_results', 'relationship_extraction.json')
output_path = os.path.join(project_root, 'data', 'nlp_results', 'unified_nlp_results.csv')

# 1. Load the NLP results (from your basic NLP processor)
with open(nlp_results_path, 'r') as f:
    nlp_results = json.load(f)
nlp_df = pd.DataFrame(nlp_results)
print(f"Loaded NLP results for {len(nlp_df)} documents.")

# 2. Load the topic assignments (from advanced NLP - BERTopic)
topics_df = pd.read_csv(topics_path)
print(f"Loaded topics for {len(topics_df)} documents.")

# 3. Merge the two dataframes on the common 'id' field.
#    (Assuming both nlp_results and topics_df share the same document IDs.)
merged_df = pd.merge(nlp_df, topics_df[['id', 'topic']], on='id', how='left')
print("Merged basic NLP results with topic assignments.")

# 4. Load the relationship extraction results (JSON file).
with open(relationships_path, 'r') as f:
    rel_results = json.load(f)
# Create a mapping from document index to the extracted relationships.
rel_mapping = {item['document_index']: item['relationships'] for item in rel_results}
print("Loaded relationship extraction results.")

# 5. Add a new column 'relationships' to the merged dataframe.
# Here we assume the row order in merged_df corresponds to the document index used during relationship extraction.
relationships = []
for idx, _ in merged_df.iterrows():
    # If relationships exist for this document index, assign them; else, assign an empty list.
    relationships.append(rel_mapping.get(idx, []))
merged_df['relationships'] = relationships

# 6. Save the unified NLP results to a CSV file.
merged_df.to_csv(output_path, index=False)
print("Unified NLP results saved to:", output_path)
