import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.collectors.base_collector import BaseCollector
import requests
import pandas as pd
from datetime import datetime
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ScopusCollector(BaseCollector):
    def __init__(self):
        # Initialize base collector
        data_dir = os.path.join(project_root, 'data/raw/scopus')
        super().__init__(source_name="scopus", data_dir=data_dir)
        
        # Scopus specific initialization
        self.api_key = os.getenv('SCOPUS_API_KEY')
        self.base_url = "https://api.elsevier.com/content/search/scopus"
        self.headers = {
            'X-ELS-APIKey': self.api_key,
            'Accept': 'application/json'
        }
        
    def search_items(self, num_results=50):
        """Search Scopus for graphene-related papers"""
        if not self.api_key:
            self.logger.error("Scopus API key not found. Please set SCOPUS_API_KEY environment variable.")
            return
            
        print(f"Searching Scopus for: graphene applications")
        print(f"Using API key: {self.api_key[:5]}...")
        
        try:
            batch_size = 25  # Scopus limit per request
            for start in range(0, num_results, batch_size):
                params = {
                    'query': 'TITLE-ABS-KEY(graphene applications)',
                    'count': min(batch_size, num_results - start),
                    'start': start,
                    'sort': '-coverDate',
                    'view': 'COMPLETE'
                }
                
                print(f"\nFetching batch {start//batch_size + 1} (records {start+1} to {start+batch_size})...")
                
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    entries = data.get('search-results', {}).get('entry', [])
                    total_results = int(data.get('search-results', {}).get('opensearch:totalResults', 0))
                    
                    print(f"Found {len(entries)} papers in batch (Total available: {total_results})")
                    
                    for entry in entries:
                        try:
                            # Extract authors
                            authors = entry.get('author', [])
                            author_names = [f"{author.get('given-name', '')} {author.get('surname', '')}" 
                                          for author in authors]
                            
                            paper_data = {
                                'title': entry.get('dc:title', ''),
                                'authors': '; '.join(author_names),
                                'abstract': entry.get('dc:description', ''),
                                'published_date': entry.get('prism:coverDate', ''),
                                'doi': entry.get('prism:doi', ''),
                                'journal': entry.get('prism:publicationName', ''),
                                'volume': entry.get('prism:volume', ''),
                                'issue': entry.get('prism:issueIdentifier', ''),
                                'pages': entry.get('prism:pageRange', ''),
                                'citations': int(entry.get('citedby-count', 0)),
                                'keywords': entry.get('authkeywords', ''),
                                'url': entry.get('prism:url', ''),
                                'collection_date': datetime.now().isoformat(),
                                'source': 'Scopus'
                            }
                            
                            self.data.append(paper_data)
                            print(f"Collected: {paper_data['title'][:100]}...")
                            
                        except Exception as e:
                            print(f"Error processing paper: {str(e)}")
                            self.logger.error(f"Error processing paper: {str(e)}")
                            continue
                            
                elif response.status_code == 401:
                    error_msg = "Authentication Error (401). Please verify your API key."
                    print(error_msg)
                    self.logger.error(error_msg)
                    print(f"Response: {response.text}")
                    break
                else:
                    error_msg = f"Error: API returned status code {response.status_code}"
                    print(error_msg)
                    self.logger.error(error_msg)
                    print(f"Response: {response.text}")
                    break
                    
                time.sleep(0.2)  # Rate limiting
                
        except Exception as e:
            error_msg = f"Error in search: {str(e)}"
            print(error_msg)
            self.logger.error(error_msg)
            
    def analyze_data(self, df):
        """Analyze the collected data"""
        if df is None or df.empty:
            return
            
        print("\nCollection Summary:")
        print(f"Total papers collected: {len(df)}")
        
        # Date range
        df['published_date'] = pd.to_datetime(df['published_date'])
        print(f"Date range: {df['published_date'].min().date()} to {df['published_date'].max().date()}")
        
        # Citation statistics
        try:
            df['citations'] = pd.to_numeric(df['citations'], errors='coerce')
            df['citations'] = df['citations'].fillna(0).astype(int)
            
            total_citations = df['citations'].sum()
            avg_citations = df['citations'].mean()
            
            print(f"\nCitation Statistics:")
            print(f"Total citations: {total_citations:,}")
            print(f"Average citations per paper: {avg_citations:.2f}")
            
            if len(df) > 0:
                most_cited_idx = df['citations'].idxmax()
                most_cited_title = df.loc[most_cited_idx, 'title']
                most_cited_count = df.loc[most_cited_idx, 'citations']
                print(f"Most cited paper: {most_cited_title} ({most_cited_count:,} citations)")
        except Exception as e:
            print(f"Error processing citation statistics: {str(e)}")
        
        # Journal statistics
        print(f"\nTop Journals:")
        print(df['journal'].value_counts().head())
        
        # Author statistics
        all_authors = [author.strip() for authors in df['authors'].str.split(';') 
                    for author in authors if author.strip()]
        top_authors = pd.Series(all_authors).value_counts().head()
        print(f"\nTop Authors:")
        print(top_authors)

def main():
    collector = ScopusCollector()
    print("Starting Scopus data collection...")
    collector.search_items(num_results=50)
    df = collector.save_data('scopus_papers.csv')
    if df is not None:
        collector.analyze_data(df)

if __name__ == "__main__":
    main()