import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

def setup_logging():
    """Set up logging configuration"""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('patents_preprocessor')
    logger.setLevel(logging.INFO)
    
    # Check if handler already exists
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, 'patents_preprocessing.log'))
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        
    logger.propagate = False
    return logger

def preprocess_google_patents():
    """Preprocess Google Patents CSV files to a simplified format"""
    logger = setup_logging()
    logger.info("Starting Google Patents preprocessing with simplified output")
    
    patents_dir = os.path.join(project_root, 'data', 'raw', 'patents')
    processed_dir = os.path.join(patents_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Looking for CSV files in: {patents_dir}")
    
    # List all files in the directory
    all_files = os.listdir(patents_dir)
    
    # Filter for CSV files
    csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('google_patents_processed')]
    print(f"Found {len(csv_files)} CSV files to process")
    
    total_patents = 0
    merged_patents = []
    
    for file in csv_files:
        file_path = os.path.join(patents_dir, file)
        print(f"Processing file: {file}")
        
        try:
            # Skip the first row which contains the search URL
            df = pd.read_csv(file_path, skiprows=1)
            
            if df.empty:
                print(f"File is empty: {file}")
                continue
                
            print(f"Found {len(df)} patents in file")
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Get title 
                    title = ""
                    if 'title' in df.columns and pd.notna(row['title']):
                        title = str(row['title'])
                    else:
                        continue  # Skip patents without a title
                    
                    # Get publication date (or most recent date available)
                    date = ""
                    if 'publication date' in df.columns and pd.notna(row['publication date']):
                        date = str(row['publication date'])
                    elif 'grant date' in df.columns and pd.notna(row['grant date']):
                        date = str(row['grant date'])
                    elif 'filing/creation date' in df.columns and pd.notna(row['filing/creation date']):
                        date = str(row['filing/creation date'])
                    elif 'priority date' in df.columns and pd.notna(row['priority date']):
                        date = str(row['priority date'])
                    
                    # Get authors
                    authors = ""
                    if 'inventor/author' in df.columns and pd.notna(row['inventor/author']):
                        authors = str(row['inventor/author'])
                    
                    # Simplified patent entry with just the essentials
                    patent_data = {
                        'title': title,
                        'date': date,
                        'authors': authors,
                        'source': 'Google Patents'
                    }
                    
                    # Only add if we have at least a title
                    if patent_data['title']:
                        merged_patents.append(patent_data)
                        total_patents += 1
                    
                except Exception as e:
                    print(f"Error processing patent row: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    # Save the merged and processed patents
    if merged_patents:
        output_df = pd.DataFrame(merged_patents)
        output_path = os.path.join(processed_dir, 'google_patents_simplified.csv')
        output_df.to_csv(output_path, index=False)
        print(f"Saved {total_patents} processed patents to {output_path}")
        
        # Also save as a text file format for easier reading
        text_path = os.path.join(processed_dir, 'google_patents_simplified.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for patent in merged_patents:
                f.write(f"Title: {patent['title']}\n")
                f.write(f"Date: {patent['date']}\n")
                f.write(f"Authors: {patent['authors']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Also saved as text file to {text_path}")
    else:
        print("No patents were processed.")

if __name__ == "__main__":
    preprocess_google_patents()