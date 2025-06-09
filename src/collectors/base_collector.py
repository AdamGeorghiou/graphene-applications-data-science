from abc import ABC, abstractmethod
import pandas as pd
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
import json

class BaseCollector(ABC):
    def __init__(self, source_name: str, data_dir: str):
        self.source_name = source_name
        self.data_dir = data_dir
        self.data = []
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.join(self.data_dir, '../logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(f'{self.source_name}_collector')
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(log_dir, f'{self.source_name}_collection.log'))
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.propagate = False
    
    @abstractmethod
    def search_items(self, num_results: int) -> None:
        """Search and collect items from the source"""
        pass
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate collected item data"""
        required_fields = ['title', 'abstract', 'published_date', 'collection_date', 'source']
        for field in required_fields:
            if field not in item or not item[field]:
                self.logger.warning(f"Item validation failed: missing required field '{field}'")
                return False

        # Log validation success
        return True
    
    def save_data(self, filename: str) -> pd.DataFrame:
        """Save collected data to CSV with validation"""
        if not self.data:
            self.logger.warning("No data to save!")
            return None
        
        self.logger.info(f"Attempting to save {len(self.data)} items to CSV")

        try:
            # Filter out invalid items
            valid_data = [item for item in self.data if self.validate_item(item)]
            self.logger.info(f"Valid items: {len(valid_data)}/{len(self.data)}")
            
            df = pd.DataFrame(valid_data)
            # After filtering out invalid items
            invalid_items = [item for item in self.data if not self.validate_item(item)]
            if invalid_items:
                self.logger.warning(f"Filtered out {len(invalid_items)} invalid items")
                for item in invalid_items:
                    self.logger.warning(f"Invalid item title: {item.get('title', 'NO TITLE')[:100]}")
            # Save full dataset
            output_path = os.path.join(self.data_dir, filename)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(df)} items to {output_path}")
            
            # Save metadata about the collection
            metadata = {
                'source': self.source_name,
                'collection_date': datetime.now().isoformat(),
                'total_items': len(df),
                'fields_collected': list(df.columns),
                'date_range': [
                    df['published_date'].min(),
                    df['published_date'].max()
                ] if 'published_date' in df.columns else None
            }
            
            metadata_path = os.path.join(self.data_dir, f'{self.source_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return None
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze collected data"""
        if df is None or df.empty:
            return {'status': 'No data to analyze'}
            
        try:
            analysis = {
                'total_items': len(df),
                'unique_titles': df['title'].nunique(),
                'date_range': [
                    df['published_date'].min(),
                    df['published_date'].max()
                ] if 'published_date' in df.columns else None,
                'fields_present': list(df.columns),
                'missing_data': df.isnull().sum().to_dict()
            }
            
            self.logger.info(f"Analysis complete: {json.dumps(analysis, indent=2)}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return {'error': str(e)}