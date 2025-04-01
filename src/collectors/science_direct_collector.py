# science_direct_collector.py

import os
import sys
import requests
import pandas as pd
from datetime import datetime
import time
import logging
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.collectors.base_collector import BaseCollector

class ScienceDirectCollector(BaseCollector):
    """Collector for ScienceDirect (Elsevier) academic articles"""
    
    def __init__(self):
        # Initialize with source name and data directory
        data_dir = os.path.join(project_root, "data", "raw", "sciencedirect")
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize base collector
        super().__init__(source_name="sciencedirect", data_dir=data_dir)
        
        # ScienceDirect API key from environment
        self.api_key = os.getenv("SCIENCEDIRECT_API_KEY")
        if not self.api_key:
            self.logger.warning("ScienceDirect API key not found in environment")
        
        self.base_url = "https://api.elsevier.com/content/search/sciencedirect"
        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }
    
    def search_items(self, query="graphene applications", num_results=100):
        """Search ScienceDirect for articles"""
        if not self.api_key:
            self.logger.error("Cannot search without API key")
            return None
        
        self.logger.info(f"Searching ScienceDirect for: {query} (max: {num_results})")
        
        # Reset data store
        self.data = []
        
        # Process in batches (ScienceDirect has a limit per request)
        batch_size = 25
        for offset in range(0, num_results, batch_size):
            params = {
                "query": query,
                "count": min(batch_size, num_results - offset),
                "start": offset,
                "date": "2010-present",
                "field": "all"
            }
            
            try:
                self.logger.info(f"Fetching batch {offset//batch_size + 1} (start: {offset})")
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("search-results", {}).get("entry", [])
                    
                    for item in results:
                        try:
                            # Extract article details
                            article = {
                                "id": item.get("dc:identifier", ""),
                                "title": item.get("dc:title", ""),
                                "abstract": item.get("dc:description", ""),
                                "authors": item.get("authors", ""),
                                "published_date": item.get("prism:coverDate", ""),
                                "doi": item.get("prism:doi", ""),
                                "url": item.get("prism:url", ""),
                                "venue": item.get("prism:publicationName", ""),
                                "collection_date": datetime.now().isoformat(),
                                "source": "ScienceDirect"
                            }
                            
                            self.data.append(article)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing article: {str(e)}")
                    
                    # Respect API rate limits
                    time.sleep(1)
                    
                elif response.status_code == 401:
                    self.logger.error("API key unauthorized")
                    break
                else:
                    self.logger.error(f"API error: {response.status_code}")
                    self.logger.error(response.text)
                    break
            
            except Exception as e:
                self.logger.error(f"Error in search: {str(e)}")
        
        self.logger.info(f"Total items collected: {len(self.data)}")
        return len(self.data)
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the collected data"""
        if df is None or df.empty:
            return {"error": "No data to analyze"}
        
        analysis = {
            "total_items": len(df),
            "date_range": [df["published_date"].min(), df["published_date"].max()]
            if "published_date" in df.columns else None
        }
        
        # Journal statistics
        if "venue" in df.columns:
            top_venues = df["venue"].value_counts().head(10).to_dict()
            analysis["top_venues"] = top_venues
        
        return analysis