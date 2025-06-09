from src.collectors.base_collector import BaseCollector
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from datetime import datetime
import os

class ArxivCollector(BaseCollector):
    def __init__(self):
        # Initialize with source name and data directory
        super().__init__(
            source_name="arxiv",
            data_dir="/Users/adamgeorghiou/Desktop/GIM/Project/data/raw/arxiv"
        )
        self.base_url = "http://export.arxiv.org/api/query"
        
    def search_items(self, num_results=100):
        """
        Implementation of the abstract method from BaseCollector
        """
        self.logger.info(f"Starting search for {num_results} papers")
        query = urllib.parse.quote("graphene applications")
        print(f"Searching arXiv for: graphene applications")
        
        try:
            query_url = f"{self.base_url}?search_query=all:{query}&start=0&max_results={num_results}"
            
            print("Fetching data from arXiv...")
            with urllib.request.urlopen(query_url) as response:
                response_data = response.read()
            
            root = ET.fromstring(response_data)
            namespace = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespace)
            print(f"Found {len(entries)} papers")
            self.logger.info(f"Found {len(entries)} papers")
            
            for entry in entries:
                try:
                    title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
                    abstract = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
                    published = entry.find('atom:published', namespace).text
                    
                    authors = [author.find('atom:name', namespace).text 
                             for author in entry.findall('atom:author', namespace)]
                    
                    categories = [cat.get('term') 
                                for cat in entry.findall('atom:category', namespace)]
                    
                    paper_data = {
                        'title': title,
                        'authors': ', '.join(authors),
                        'abstract': abstract,
                        'published_date': published,
                        'url': entry.find('atom:id', namespace).text,
                        'categories': ', '.join(categories),
                        'collection_date': datetime.now().isoformat(),
                        'source': 'arXiv'
                    }
                    
                    # Use parent class's validate_item method
                    if self.validate_item(paper_data):
                        self.data.append(paper_data)
                        self.logger.info(f"Collected paper: {title[:100]}")
                        print(f"Successfully collected paper: {title[:100]}...")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing paper: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            
def main():
    collector = ArxivCollector()
    collector.search_items(num_results=50)
    df = collector.save_data('arxiv_papers.csv')
    if df is not None:
        analysis = collector.analyze_data(df)
        print("Data collection summary:", analysis)

if __name__ == "__main__":
    main()
