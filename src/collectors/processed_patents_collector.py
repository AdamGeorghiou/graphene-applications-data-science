from src.collectors.base_collector import BaseCollector
import os
import pandas as pd
import glob

class ProcessedPatentsCollector(BaseCollector):
    """Collector that reads from pre-processed Google Patents CSV files"""
    
    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        patents_dir = os.path.join(project_root, "data", "raw", "patents")
        super().__init__(source_name="patents", data_dir=patents_dir)
        
        # Set the path to the processed patents file
        self.processed_file = os.path.join(patents_dir, "processed", "google_patents_simplified.csv")
        print(f"Patents file path: {self.processed_file}")
        print(f"Patents file exists: {os.path.exists(self.processed_file)}")
    
    def search_items(self, num_results=150, **kwargs):
        """Read data from processed patents CSV file"""
        self.logger.info(f"Reading processed patents data from {self.processed_file}")
        
        if not os.path.exists(self.processed_file):
            self.logger.error(f"Processed patents file not found: {self.processed_file}")
            return []
        
        try:
            # Read the processed patents file
            df = pd.read_csv(self.processed_file)
            self.logger.info(f"Read {len(df)} patents from processed file")
            
            # Limit to requested number
            if len(df) > num_results:
                df = df.head(num_results)
                self.logger.info(f"Limited to {num_results} patents")
            
            # Convert to list of dictionaries and explicitly store in self.data
            self.data = []
            for _, row in df.iterrows():
                patent_data = {
                    'title': row['title'],
                    'date': row['date'],
                    'authors': row['authors'],
                    'source': 'patents'  # Ensure consistent source naming
                }
                self.data.append(patent_data)
                
            self.logger.info(f"Processed {len(self.data)} patent records")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error reading processed patents file: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_data(self, filename=None):
        """Override to properly format data for saving"""
        if not self.data:
            self.logger.warning("No data to save")
            return None
            
        # Convert data list to DataFrame
        df = pd.DataFrame(self.data)
        
        # Save to file if filename provided
        if filename:
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(df)} records to {filepath}")
        
        return df