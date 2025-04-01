import os
import sys
import requests
import pandas as pd
from datetime import datetime
import time
import logging
import json
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.collectors.base_collector import BaseCollector
from src.config.settings import SEMANTIC_SCHOLAR_API_KEY, API_RETRY_ATTEMPTS, API_RETRY_DELAY

# Load environment variables
load_dotenv()

class SemanticScholarCollector(BaseCollector):
    def __init__(self):
        # Initialize base collector
        data_dir = os.path.join(project_root, 'data/raw/semantic_scholar')
        os.makedirs(data_dir, exist_ok=True)
        super().__init__(source_name="semantic_scholar", data_dir=data_dir)
        
        # Semantic Scholar specific initialization
        self.api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {
            'x-api-key': self.api_key
        }
        
    def search_items(self, num_results=50, query="graphene applications", start_year=2015):
        """Search Semantic Scholar for graphene-related papers"""
        if not self.api_key:
            self.logger.error("Semantic Scholar API key not found. Please set SEMANTIC_SCHOLAR_API_KEY environment variable.")
            return
        
        self.logger.info(f"Collecting from Semantic Scholar with max_results={num_results}")    
        print(f"Searching Semantic Scholar for: {query}")
        print(f"Using API key: {self.api_key[:5]}..." if self.api_key else "No API key provided")
        
        try:
            search_url = f"{self.base_url}/paper/search"
            
            # Parameters for pagination
            batch_size = 100  # Semantic Scholar allows up to 100 per request
            offset = 0
            papers_collected = 0
            
            while papers_collected < num_results:
                batch_size = min(batch_size, num_results - papers_collected)
                params = {
                    'query': query,
                    'limit': batch_size,
                    'offset': offset,
                    'fields': 'paperId,title,abstract,year,authors,venue,publicationDate,externalIds,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf,citationCount'
                }
                
                if start_year:
                    params['year'] = f"{start_year}-"
                
                print(f"\nFetching batch {(offset // batch_size) + 1} (papers {offset+1} to {offset+batch_size})...")
                
                # Attempt API call with retries
                for attempt in range(API_RETRY_ATTEMPTS):
                    try:
                        response = requests.get(
                            search_url,
                            headers=self.headers,
                            params=params
                        )
                        response.raise_for_status()
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt < API_RETRY_ATTEMPTS - 1:
                            print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {API_RETRY_DELAY} seconds...")
                            time.sleep(API_RETRY_DELAY)
                        else:
                            print(f"All attempts failed: {str(e)}")
                            self.logger.error(f"API request failed after {API_RETRY_ATTEMPTS} attempts: {str(e)}")
                            return
                
                # Process response
                data = response.json()
                # Log total results if available
                if 'total' in data:
                    self.logger.info(f"API reports {data['total']} total results available")

                if 'data' not in data or not data['data']:
                    print("No more papers found")
                    break
                
                papers = data['data']
                print(f"Found {len(papers)} papers in this batch")
                self.logger.info(f"API returned {len(papers)} papers in batch (offset={offset}, total collected so far: {papers_collected})")
                # Process each paper
                for paper in papers:
                    try:
                        # Extract authors
                        authors = []
                        if 'authors' in paper and paper['authors']:
                            authors = [author.get('name', '') for author in paper['authors'] if 'name' in author]
                        
                        # Extract DOI
                        doi = None
                        if 'externalIds' in paper and paper['externalIds'] and 'DOI' in paper['externalIds']:
                            doi = paper['externalIds']['DOI']
                        
                        # Extract fields of study / keywords
                        keywords = []
                        if 'fieldsOfStudy' in paper and paper['fieldsOfStudy']:
                            keywords = paper['fieldsOfStudy']
                        elif 's2FieldsOfStudy' in paper and paper['s2FieldsOfStudy']:
                            keywords = [field.get('category', '') for field in paper['s2FieldsOfStudy'] if 'category' in field]
                        
                        # Format date
                        pub_date = paper.get('publicationDate', '')
                        if not pub_date and 'year' in paper:
                            pub_date = str(paper['year'])
                        
                        # Prepare paper data
                        paper_data = {
                            'title': paper.get('title', ''),
                            'authors': '; '.join(authors),
                            'abstract': paper.get('abstract', ''),
                            'published_date': pub_date,
                            'doi': doi,
                            'journal': paper.get('venue', ''),
                            'keywords': '; '.join(keywords) if keywords else '',
                            'url': f"https://www.semanticscholar.org/paper/{paper['paperId']}" if 'paperId' in paper else '',
                            'citations': paper.get('citationCount', 0),
                            'collection_date': datetime.now().isoformat(),
                            'source': 'Semantic Scholar',
                            'paper_id': paper.get('paperId', '')
                        }
                        
                        self.data.append(paper_data)
                        print(f"Collected: {paper_data['title'][:100]}...")
                        
                    except Exception as e:
                        print(f"Error processing paper: {str(e)}")
                        self.logger.error(f"Error processing paper: {str(e)}")
                        continue
                
                # Update counters for next batch
                papers_collected += len(papers)
                offset += len(papers)
                
                self.logger.info(f"Checking if we should continue: papers in batch={len(papers)}, batch_size={batch_size}, total collected={papers_collected}, target={num_results}")
                # Check if we should continue
                if len(papers) == 0 or papers_collected >= num_results:
                    break
                self.logger.info(f"Collection complete. Total papers collected: {len(self.data)} of {num_results} requested")
                # Rate limiting
                time.sleep(1)
                
            print(f"\nCollection complete. Total papers collected: {len(self.data)}")
            self.logger.info(f"Collection complete. Total papers collected: {len(self.data)}")
                
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
        if 'published_date' in df.columns and not df['published_date'].isna().all():
            try:
                df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
                valid_dates = df['published_date'].dropna()
                if not valid_dates.empty:
                    print(f"Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
            except Exception as e:
                print(f"Error processing dates: {str(e)}")
        
        # Citation statistics
        if 'citations' in df.columns:
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
        if 'journal' in df.columns:
            print(f"\nTop Journals:")
            print(df['journal'].value_counts().head())
        
        # Keyword statistics
        if 'keywords' in df.columns:
            try:
                all_keywords = []
                for kw_string in df['keywords'].dropna():
                    keywords = [k.strip() for k in kw_string.split(';') if k.strip()]
                    all_keywords.extend(keywords)
                
                keyword_counts = pd.Series(all_keywords).value_counts()
                print(f"\nTop Keywords/Fields of Study:")
                print(keyword_counts.head(10))
            except Exception as e:
                print(f"Error processing keywords: {str(e)}")

def main():
    collector = SemanticScholarCollector()
    print("Starting Semantic Scholar data collection...")
    collector.search_items(num_results=50)
    df = collector.save_data('semantic_scholar_papers.csv')
    if df is not None:
        collector.analyze_data(df)

if __name__ == "__main__":
    main()