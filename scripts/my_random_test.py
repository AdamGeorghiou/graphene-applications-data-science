import pandas as pd
df = pd.read_csv("data/processed/cleaned_graphene_data.csv")

# Quick check: papers with abstracts (excluding patents)
papers_with_abstracts = df[
    (~df['source'].str.contains('patent', case=False, na=False)) & 
    (df['abstract'].notna()) &
    (df['abstract'].str.strip().str.len() > 20)
]

print(f"Papers with proper abstracts: {len(papers_with_abstracts):,}")
print(f"Estimated API cost: ${len(papers_with_abstracts) * 0.88 / 600:.2f}")