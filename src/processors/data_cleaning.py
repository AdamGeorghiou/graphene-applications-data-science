import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from typing import Dict, List, Any, Optional


# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

class GrapheneDataCleaner:
    def __init__(self):
        self.setup_logging()
        self.raw_data_dir = os.path.join(project_root, 'data', 'raw')
        self.processed_data_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
    def setup_logging(self) -> None:
        """Set up logging configuration"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('data_cleaner')
        self.logger.setLevel(logging.INFO)
        
        # Check if handler already exists
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, 'data_cleaning.log'))
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            
        self.logger.propagate = False
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from all CSV files in the raw data directory recursively."""
        data_frames = {}
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.endswith('.csv') and not file.endswith('_summary.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        # Derive a 'source' from the immediate parent folder name
                        source = os.path.basename(root)
                        if source in data_frames:
                            data_frames[source] = pd.concat([data_frames[source], df], ignore_index=True)
                        else:
                            data_frames[source] = df
                        print(f"Loaded {file} from {root} with {len(df)} records")
                        self.logger.info(f"Loaded {file} from {root} with {len(df)} records")
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
                        self.logger.error(f"Error loading {file}: {str(e)}")
        return data_frames

    def clean_text(self, text: str) -> str:
        """Clean and standardize text fields"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^\w\s.,;?!-]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def standardize_dates(self, date_val: Any) -> Optional[str]:
        """Extract only the year from date values.
        Converts the input to a string if needed."""
        if pd.isna(date_val):
            return None
        try:
            # Convert the date value to a string
            date_str = str(date_val)
            # Extract a 4-digit year from the string
            match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if match:
                return match.group(0)  # Return only the year
        except Exception as e:
            self.logger.warning(f"Could not parse date: {date_val}")
        return None  
    
    def standardize_graphene_terms(self, text: str) -> str:
        """Standardize graphene-specific terminology"""
        if pd.isna(text) or text == "":
            return ""
        # Mapping of variant terms to standardized forms
        term_map = {
            r'\bgraphene oxide\b': 'graphene oxide',
            r'\bGO\b': 'graphene oxide',
            r'\breduced graphene oxide\b': 'reduced graphene oxide',
            r'\brGO\b': 'reduced graphene oxide',
            r'\bgraphene\b': 'graphene'
            # Extend this mapping as needed
        }
        for pattern, standard in term_map.items():
            text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        return text
    
    def normalize_authors(self, authors: str) -> str:
        """Normalize author names for consistency"""
        if pd.isna(authors):
            return ""
        # Clean the text and convert to lower case
        authors = self.clean_text(authors).lower()
        # Optionally, split, trim, and sort to ensure consistent ordering
        authors_list = [a.strip() for a in authors.split(',') if a.strip()]
        authors_list = sorted(authors_list)
        return ', '.join(authors_list)
    
    def normalize_affiliations(self, affiliations: str) -> str:
        """Normalize affiliations for consistency"""
        if pd.isna(affiliations):
            return ""
        affiliations = self.clean_text(affiliations).lower()
        return affiliations
    
    def clean_dataframe(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Clean and standardize a single dataframe"""
        print(f"Cleaning {source} data")
        self.logger.info(f"Cleaning {source} data")
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = [col.lower().strip() for col in df_clean.columns]
        
        # Clean and standardize text fields (e.g., title, abstract)
        text_columns = ['title', 'abstract']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
                df_clean[col] = df_clean[col].apply(self.standardize_graphene_terms)
        
        # Normalize authors and affiliations if they exist
        if 'authors' in df_clean.columns:
            df_clean['authors'] = df_clean['authors'].apply(self.normalize_authors)
        if 'affiliations' in df_clean.columns:
            df_clean['affiliations'] = df_clean['affiliations'].apply(self.normalize_affiliations)
        
        
        
        # Standardize dates to extract only the year
        date_columns = ['published_date', 'collection_date', 'date']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.standardize_dates)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')  # Convert to integer year

        # If the cleaned dataframe has a 'date' column but no 'published_date', rename it.
        if 'date' in df_clean.columns and 'published_date' not in df_clean.columns:
            df_clean = df_clean.rename(columns={'date': 'published_date'})
        
        # Add source column if missing
        if 'source' not in df_clean.columns:
            df_clean['source'] = source
        
        # Remove duplicates within each source
        df_clean = df_clean.drop_duplicates(subset=['title'])
        # Remove rows with empty titles
        df_clean = df_clean.dropna(subset=['title'])
        
        return df_clean

    
    def merge_and_structure_data(self, data_frames: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge and structure all cleaned dataframes with cross-source deduplication"""
        cleaned_dfs = []
        for source, df in data_frames.items():
            cleaned_df = self.clean_dataframe(df, source)
            cleaned_dfs.append(cleaned_df)
        
        if not cleaned_dfs:
            print("No data frames to merge")
            self.logger.error("No data frames to merge")
            return None
        
        # Merge all cleaned dataframes
        merged_df = pd.concat(cleaned_dfs, ignore_index=True)
        # Cross-source deduplication based on title
        merged_df = merged_df.drop_duplicates(subset=['title'])
        
        # Create unique identifiers for each record
        merged_df['id'] = merged_df.apply(self.create_identifier, axis=1)
        print(f"Merged data shape: {merged_df.shape}")
        self.logger.info(f"Merged data shape: {merged_df.shape}")
        
        return merged_df
    
    def create_identifier(self, row: pd.Series) -> str:
        """Create a unique identifier for each row, handling missing dates"""
        try:
            source = str(row['source']).lower()
            # Handle date part
            date_str = 'unknown'
            if pd.notna(row['published_date']):
                date = pd.to_datetime(row['published_date'])
                if pd.notna(date):
                    date_str = date.strftime('%Y%m')
            # Create a hash of the title
            title_hash = str(abs(hash(str(row['title']))))[:8]
            return f"{source}_{date_str}_{title_hash}"
        except Exception as e:
            self.logger.warning(f"Error creating identifier for row: {str(e)}")
            return f"unknown_{abs(hash(str(row['title'])))}"
    
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics of the cleaned data"""
        summary = {
            'total_records': len(df),
            'records_by_source': df['source'].value_counts().to_dict(),
            'date_range': {
                'start': pd.to_datetime(df['published_date']).min() if not df['published_date'].isna().all() else 'No valid dates',
                'end': pd.to_datetime(df['published_date']).max() if not df['published_date'].isna().all() else 'No valid dates'
            },
            'missing_data': df.isnull().sum().to_dict(),
            'unique_titles': df['title'].nunique(),
            'records_without_dates': df['published_date'].isna().sum(),
            'records_without_abstracts': df['abstract'].isna().sum() if 'abstract' in df.columns else 'N/A'
        }
        return summary
    
    def process_all_data(self) -> tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Main function to process all data"""
        print("Loading data files...")
        data_frames = self.load_data()
        if not data_frames:
            print("No data files found")
            self.logger.error("No data files found")
            return None, None

        print("Merging and cleaning data...")
        cleaned_data = self.merge_and_structure_data(data_frames)
        if cleaned_data is None:
            return None, None

        # If 'published_date' is missing but 'date' exists, rename it
        if 'published_date' not in cleaned_data.columns and 'date' in cleaned_data.columns:
            cleaned_data = cleaned_data.rename(columns={'date': 'published_date'})

        # Now subset to only the necessary columns
        columns_to_keep = ['title', 'abstract', 'published_date', 'source', 'authors']
        cleaned_data = cleaned_data[[col for col in columns_to_keep if col in cleaned_data.columns]]

        print("Generating summary...")
        summary = self.generate_summary(cleaned_data)
        
        # Save cleaned data and summary
        output_path = os.path.join(self.processed_data_dir, 'cleaned_graphene_data.csv')
        cleaned_data.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        
        summary_path = os.path.join(self.processed_data_dir, 'data_summary.txt')
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}:\n{value}\n\n")
        print(f"Saved summary to {summary_path}")
        
        return cleaned_data, summary


def main():
    cleaner = GrapheneDataCleaner()
    print("Starting data cleaning process...")
    cleaned_data, summary = cleaner.process_all_data()
    
    if cleaned_data is not None and summary is not None:
        print("\nData Cleaning Summary:")
        print(f"Total records processed: {summary['total_records']}")
        print("\nRecords by source:")
        for source, count in summary['records_by_source'].items():
            print(f"  {source}: {count}")
        print(f"\nRecords without dates: {summary['records_without_dates']}")
        print(f"Records without abstracts: {summary['records_without_abstracts']}")
        print("\nMissing data summary:")
        for field, count in summary['missing_data'].items():
            if count > 0:
                print(f"  {field}: {count} missing values")

if __name__ == "__main__":
    main()
