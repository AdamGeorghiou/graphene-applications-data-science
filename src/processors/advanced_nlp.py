import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm  # progress bar
import json

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_path = os.path.join(project_root, 'data', 'processed', 'cleaned_graphene_data.csv')
results_dir = os.path.join(project_root, 'data', 'nlp_results')
os.makedirs(results_dir, exist_ok=True)

# Load cleaned data
df = pd.read_csv(data_path)
# Combine title and abstract for topic modeling
documents = (df['title'].fillna('') + " " + df['abstract'].fillna('')).tolist()

# --- Topic Modeling with BERTopic ---
# Initialize a Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create and fit the BERTopic model
print("Fitting BERTopic model. This may take a while...")
topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
topics, probabilities = topic_model.fit_transform(documents)

# Retrieve and display topic information
topic_info = topic_model.get_topic_info()
print("Topic Information:")
print(topic_info)

# Save the topic model for later use
topic_model.save(os.path.join(results_dir, "bertopic_model"))

# Save topics for each document
df['topic'] = topics
df.to_csv(os.path.join(results_dir, 'documents_with_topics.csv'), index=False)

# --- Relationship Extraction using SpaCy ---
# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_relationships(text: str):
    """
    A simple relationship extractor that identifies subject-verb-object (SVO)
    triples in sentences that mention 'graphene'.
    """
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        if "graphene" in sent.text.lower():
            for token in sent:
                if token.dep_ == "ROOT":
                    subjects = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                    objects = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj")]
                    if subjects and objects:
                        relationships.append({
                            "sentence": sent.text,
                            "subject": subjects,
                            "verb": token.text,
                            "object": objects
                        })
    return relationships

# Apply relationship extraction to a sample of documents (adjust the range as needed)
relationship_results = []
print("Extracting relationships from documents...")
for idx, doc_text in enumerate(tqdm(documents[:50], desc="Processing documents")):
    rels = extract_relationships(doc_text)
    if rels:
        relationship_results.append({
            "document_index": idx,
            "relationships": rels
        })

# Save the relationship extraction results to a JSON file
with open(os.path.join(results_dir, 'relationship_extraction.json'), 'w') as f:
    json.dump(relationship_results, f, indent=2)

print("Advanced NLP analysis complete. Check the results in the 'data/nlp_results' directory.")
