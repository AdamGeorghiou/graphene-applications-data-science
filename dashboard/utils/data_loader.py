import os
import json
import pandas as pd
import numpy as np

# Updated path calculation
def get_project_root():
    """Get the project root directory based on the current file location"""
    # Current file is in dashboard/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
    dashboard_dir = os.path.dirname(current_dir)              # dashboard directory
    project_root = os.path.dirname(dashboard_dir)             # project root
    return project_root

# Use the function to define paths
PROJECT_ROOT = get_project_root()
NLP_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "nlp_results")

def load_summary():
    """Load the NLP analysis summary"""
    summary_path = os.path.join(NLP_RESULTS_DIR, "enhanced_nlp_summary.json")
    print(f"Loading summary from: {summary_path}")
    print(f"File exists: {os.path.exists(summary_path)}")
    with open(summary_path, 'r') as f:
        return json.load(f)

def load_full_results():
    """Load the full NLP results dataset"""
    results_path = os.path.join(NLP_RESULTS_DIR, "nlp_results.json")
    print(f"Loading results from: {results_path}")
    print(f"File exists: {os.path.exists(results_path)}")
    with open(results_path, 'r') as f:
        data = json.load(f)
        return data

def get_application_data(results):
    """Extract application data from results for visualization"""
    app_data = []
    for doc in results:
        source = doc.get('source', 'Unknown')
        year = None
        if 'publication_info' in doc and 'year' in doc['publication_info']:
            year_val = doc['publication_info']['year']
            if isinstance(year_val, (int, float)) and not pd.isna(year_val):
                year = int(year_val)
            elif isinstance(year_val, str) and year_val.isdigit():
                year = int(year_val)
        
        trl = doc.get('trl_assessment', {}).get('trl_score', 0)
        
        for app in doc.get('applications', []):
            app_name = app.get('application', 'Unknown')
            category = app.get('category', 'Unknown')
            
            app_data.append({
                'id': doc.get('id', ''),
                'source': source,
                'year': year,
                'application': app_name,
                'category': category,
                'trl': trl
            })
    
    return pd.DataFrame(app_data)

def get_fabrication_data(results):
    """Extract fabrication method data from results"""
    fab_data = []
    for doc in results:
        source = doc.get('source', 'Unknown')
        year = None
        if 'publication_info' in doc and 'year' in doc['publication_info']:
            year_val = doc['publication_info']['year']
            if isinstance(year_val, (int, float)) and not pd.isna(year_val):
                year = int(year_val)
            elif isinstance(year_val, str) and year_val.isdigit():
                year = int(year_val)
        
        for method in doc.get('fabrication_methods', []):
            method_name = method.get('method', 'Unknown')
            
            fab_data.append({
                'id': doc.get('id', ''),
                'source': source,
                'year': year,
                'method': method_name
            })
    
    return pd.DataFrame(fab_data)

def filter_dataframe(df, sources=None, year_range=None):
    """Filter a dataframe based on sources and year range"""
    filtered_df = df.copy()
    
    if sources and len(sources) > 0:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    if year_range and len(year_range) == 2:
        min_year, max_year = year_range
        filtered_df = filtered_df[
            (filtered_df['year'] >= min_year) & 
            (filtered_df['year'] <= max_year)
        ]
    
    return filtered_df