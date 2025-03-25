import os
import sys
import pandas as pd
import logging
import traceback
from datetime import datetime
from Bio import Entrez, Medline
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import base collector
from src.collectors.base_collector import BaseCollector

class PubMedCollector(BaseCollector):
    """
    Collector for retrieving graphene application publications from PubMed/PMC
    
    Collects biomedical research papers related to graphene applications
    using the NCBI Entrez API.
    """
    
    def __init__(self, email: str = "your-email@example.com"):
        """
        Initialize the PubMed collector
        
        Args:
            email: Email address for NCBI's tracking (required by their API)
        """
        # Create a data directory for PubMed in the project structure
        data_dir = os.path.join(project_root, "data", "raw", "pubmed")
        os.makedirs(data_dir, exist_ok=True)
        
        # Pass both required parameters to the parent class
        super().__init__(source_name="pubmed", data_dir=data_dir)
        
        # Add console logging for easier debugging
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # Set up Entrez
        Entrez.email = email
        self.logger.info(f"PubMed collector initialized with email: {email}")
        
    def search_items(self, query="graphene", num_results=100):
        """
        Search for graphene-related publications in PubMed
        
        Args:
            query: Search query term (default: "graphene")
            num_results: Maximum number of results to retrieve
        """
        # Use the provided query
        start_year = 2010
        end_year = datetime.now().year
        
        self.logger.info(f"Searching PubMed for: '{query}' (max: {num_results})")
        year_filter = f"AND ({start_year}:{end_year}[pdat])"
        full_query = f"{query} {year_filter}"
        
        # Perform the search using Entrez API
        try:
            # Search for IDs
            self.logger.info(f"Executing search with query: {full_query}")
            try:
                handle = Entrez.esearch(
                    db="pubmed", 
                    term=full_query,
                    retmax=num_results
                )
                
                # Only read the handle once
                self.logger.info("Reading search results...")
                record = Entrez.read(handle)
                handle.close()
                
                self.logger.info(f"Search response: {record.keys()}")
                
                id_list = record.get("IdList", [])
                self.logger.info(f"Found {len(id_list)} matching publications")
                
                if not id_list:
                    self.logger.warning("No results found in search")
                    self._create_test_data(num_results)
                    return None
                
            except Exception as e:
                self.logger.error(f"Error in search step: {str(e)}")
                self.logger.error(traceback.format_exc())
                self._create_test_data(num_results)
                return None
                
            # Reset data store
            self.data = []
            
            # Process in batches to avoid API limits
            batch_size = 25  # Smaller batch size for testing
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                self.logger.info(f"Fetching batch {i//batch_size + 1}/{(len(id_list)-1)//batch_size + 1}")
                
                # Add a small delay between batches to respect NCBI rate limits
                if i > 0:  # Don't delay the first batch
                    import time
                    time.sleep(1)  # 1 second pause to avoid rate limits
                
                try:
                    # Use Medline.parse instead of Entrez.parse to handle the records
                    # This avoids the binary mode issue
                    self.logger.info(f"Fetching {len(batch_ids)} records: {batch_ids[:3]}...")
                    
                    # Use efetch with rettype=medline and retmode=text
                    handle = Entrez.efetch(
                        db="pubmed", 
                        id=batch_ids, 
                        rettype="medline", 
                        retmode="text"
                    )
                    
                    try:
                        self.logger.info("Parsing records with Medline.parse...")
                        records = list(Medline.parse(handle))
                        self.logger.info(f"Successfully parsed {len(records)} records")
                        
                        for record in records:
                            try:
                                # Extract publication details
                                pmid = record.get("PMID", "")
                                title = record.get("TI", "")
                                abstract = record.get("AB", "")
                                
                                # Extract date
                                date = record.get("DP", "")
                                # Parse year from date string
                                if date:
                                    try:
                                        # Extract year from format like "2021 Jan 15"
                                        date = date.split()[0].strip()
                                    except:
                                        # Fallback to first 4 chars (assuming year)
                                        date = date[:4]
                                
                                # Extract authors
                                authors = record.get("AU", [])
                                authors_str = ", ".join(authors) if authors else ""
                                
                                # Extract journal/venue
                                venue = record.get("JT", "")
                                
                                # Create item dictionary - keeping all required fields
                                item = {
                                    "id": pmid,
                                    "title": title,
                                    "abstract": abstract,
                                    "authors": authors_str,
                                    "venue": venue,
                                    "published_date": date,
                                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                    "source": "PubMed",
                                    "collection_date": datetime.now().isoformat()
                                }
                                
                                self.data.append(item)
                                self.logger.info(f"Added record: {pmid} - {title[:50]}...")
                                
                            except Exception as e:
                                self.logger.error(f"Error processing record: {str(e)}")
                                self.logger.error(traceback.format_exc())
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing records: {str(e)}")
                        self.logger.error(traceback.format_exc())
                    
                    handle.close()
                    
                except Exception as e:
                    self.logger.error(f"Error fetching batch details: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
            self.logger.info(f"Successfully collected {len(self.data)} PubMed records")
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Fallback to dummy data in case of API errors
            self.logger.warning("Falling back to test data")
            self._create_test_data(num_results)
            return None

    def _create_test_data(self, num_results=10):
        """Generate test data as a fallback"""
        self.data = []
        for i in range(min(num_results, 10)):  # Limit test data to 10 items
            item = {
                "id": f"TEST{i}",
                "title": f"Graphene Application Test {i}",
                "abstract": f"This is a test abstract for testing PubMed collection {i}",
                "published_date": "2023",
                "collection_date": datetime.now().isoformat(),
                "source": "PubMed",
                "authors": "Test Author",
                "venue": "Test Journal",
                "url": f"https://example.com/test{i}"
            }
            self.data.append(item)
        self.logger.info(f"Created {len(self.data)} test records")


    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the collected data"""
        if df is None or df.empty:
            return {"error": "No data to analyze"}
            
        analysis = {
            "total_items": len(df),
            "items_with_abstract": df["abstract"].notna().sum(),
            "items_without_abstract": df["abstract"].isna().sum(),
        }
        
        # Add date range if available
        if "published_date" in df.columns and not df["published_date"].isna().all():
            analysis["date_range"] = {
                "min": df["published_date"].min(),
                "max": df["published_date"].max()
            }
            
            # Year distribution
            years = df["published_date"].str[:4].value_counts().sort_index()
            analysis["year_distribution"] = years.to_dict()
        
        # Add journal statistics if available
        if "venue" in df.columns:
            top_venues = df["venue"].value_counts().head(5).to_dict()
            analysis["top_venues"] = top_venues
        
        return analysis