import gzip, json, argparse, itertools, pandas as pd
from collections import Counter
from pathlib import Path # <--- ADD THIS IMPORT

def load_jsonl(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            # Add a try-except block here for robustness if some lines might not be valid JSON
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSONDecodeError: {e} - Line: {line.strip()[:100]}...") # Log problematic line (partially)
                continue


def main(inp, out_dir): # Renamed 'out' to 'out_dir' for clarity
    docs = list(load_jsonl(inp))
    if not docs:
        print(f"No documents loaded from {inp}. Exiting aggregation.")
        return

    print(f"Loaded {len(docs)} documents for aggregation.")

    # Ensure output directory exists
    output_path_obj = Path(out_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # --- Yearly trends ---
    # Ensure year is treated as an integer if possible, handle None or non-numeric
    years = []
    for d in docs:
        year_val = d.get("year")
        if isinstance(year_val, (int, float)) and not pd.isna(year_val):
            years.append(int(year_val))
        elif isinstance(year_val, str) and year_val.isdigit():
             try:
                 years.append(int(year_val))
             except ValueError:
                 pass # ignore if string year is not a valid int
    
    if years:
        year_cnt = Counter(years)
        pd.Series(year_cnt, name="docs").sort_index().to_csv(output_path_obj / "yearly_trends.csv")
        print(f"Saved yearly_trends.csv to {out_dir}")
    else:
        print("No valid year data found to create yearly_trends.csv.")


    # --- Materials count ---
    # Assuming 'materials' in each doc is a list of dicts like [{'text': 'graphene', ...}, ...]
    # We need to extract the 'text' field from each material dict
    all_materials_texts = []
    for d in docs:
        materials_list = d.get("entities", {}).get("materials", []) # Get from 'entities' key
        if isinstance(materials_list, list):
            for mat_dict in materials_list:
                if isinstance(mat_dict, dict) and "text" in mat_dict:
                    all_materials_texts.append(mat_dict["text"].lower().strip()) # Normalize

    if all_materials_texts:
        mats_cnt = Counter(all_materials_texts)
        pd.Series(mats_cnt, name="count").sort_values(ascending=False).head(100).to_csv(output_path_obj / "materials_count.csv") # Top 100
        print(f"Saved materials_count.csv to {out_dir}")
    else:
        print("No material data found to create materials_count.csv.")


    # --- Fabrication methods count ---
    # Assuming 'fabrication_methods' in each doc is a list of dicts like [{'text': 'CVD', ...}, ...]
    all_fab_method_texts = []
    for d in docs:
        fab_methods_list = d.get("entities", {}).get("fabrication_methods", []) # Get from 'entities' key
        if isinstance(fab_methods_list, list):
            for fab_dict in fab_methods_list:
                if isinstance(fab_dict, dict) and "text" in fab_dict:
                    all_fab_method_texts.append(fab_dict["text"].lower().strip()) # Normalize

    if all_fab_method_texts:
        fabs_cnt = Counter(all_fab_method_texts)
        pd.Series(fabs_cnt, name="count").sort_values(ascending=False).head(50).to_csv(output_path_obj / "fabrication_methods.csv") # Top 50
        print(f"Saved fabrication_methods.csv to {out_dir}")
    else:
        print("No fabrication method data found to create fabrication_methods.csv.")

    print("Aggregation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate NLP results from JSONL to CSVs.")
    parser.add_argument("--input", required=True, help="Path to the input docs_full.jsonl.gz file.")
    parser.add_argument("--outdir", required=True, help="Directory to save the aggregated CSV files.")
    args = parser.parse_args()

    # Path creation moved to main for clarity, already done above
    # Path(args.outdir).mkdir(parents=True, exist_ok=True)
    main(args.input, args.outdir)