"""
Unified Collection Runner for Graphene Applications Project

This script provides a single interface for collecting data from all sources,
with optimized parameters for each source and advanced configuration options.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional
import concurrent.futures
from dotenv import load_dotenv


# Add the project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Now import the collectors
from src.collectors.arxiv_collector import ArxivCollector
from src.collectors.scopus_collector import ScopusCollector
from src.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.collectors.processed_patents_collector import ProcessedPatentsCollector  # Add this line
from src.collectors.pubmed_collector import PubMedCollector

# Conditionally import IEEE collector if module exists
try:
    from src.collectors.ieee_collector import IEEECollector # type: ignore
    IEEE_AVAILABLE = True
except ImportError:
    IEEE_AVAILABLE = False

# Source-specific optimized parameters
SOURCE_PARAMS = {
    "arxiv": {
        "default_query": "graphene AND (application OR applications OR use OR uses OR device OR devices)",
        "categories": ["cond-mat.mtrl-sci", "cond-mat.mes-hall", "physics.app-ph"],
        "sort_by": "submittedDate",
        "sort_order": "descending",
        "max_results": 200
    },
    "scopus": {
        "default_query": "TITLE-ABS-KEY(graphene AND (application OR applications OR use OR uses OR device OR technology))",
        "view": "COMPLETE",
        "sort": "-coverDate",
        "max_results": 200
    },
    "patents": {  # Add this if not already there
        "default_query": "graphene applications",
        "max_results": 150
    },
    "semantic_scholar": {
        "default_query": "graphene applications",
        "fields": "paperId,title,abstract,year,authors,venue,publicationDate,externalIds,fieldsOfStudy,s2FieldsOfStudy,citationCount,openAccessPdf",
        "max_results": 200,
        "start_year": 2010
    },
    "ieee": {
        "default_query": "graphene AND (application OR applications OR uses OR device OR technology OR implementation)",
        "content_type": "Conferences,Journals,Early Access,Standards",
        "max_results": 200,
        "sort_order": "desc",
        "sort_field": "publication_date"
    },  
    "pubmed": {
        "default_query": "graphene AND (application OR applications OR device OR technology OR biomedical OR biosensor)",
        "max_results": 100
    }
}

class UnifiedCollectionRunner:
    """
    Unified runner for all data collection sources
    
    Provides a centralized interface for collecting data from multiple sources
    with optimized parameters, parallel execution, and comprehensive logging.
    """
    
    def __init__(self, output_dir: Optional[str] = None, parallel: bool = True):
        """Initialize the unified collection runner"""
        self.setup_logging()
        
        # Set up data directories
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(project_root, "data", "raw", f"collection_{timestamp}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Configure parallel execution
        self.parallel = parallel
        self.logger.info(f"Parallel execution: {'enabled' if parallel else 'disabled'}")
        
        # Initialize collectors
        self.collectors = self._initialize_collectors()
        
        # Collection results
        self.results = {}
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'unified_collection_{timestamp}.log')
        
        # Configure logger
        self.logger = logging.getLogger('unified_collector')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
    def _initialize_collectors(self) -> Dict[str, Any]:
        """Initialize all available collectors"""
        collectors = {}
        
        # Initialize each collector
        try:
            collectors["arxiv"] = ArxivCollector()
            self.logger.info("ArXiv collector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing ArXiv collector: {e}")
        
        try:
            collectors["scopus"] = ScopusCollector()
            self.logger.info("Scopus collector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Scopus collector: {e}")
        
        # Add this block to initialize the ProcessedPatentsCollector
        try:
            collectors["patents"] = ProcessedPatentsCollector()
            self.logger.info("Patents collector initialized (using processed data)")
        except Exception as e:
            self.logger.error(f"Error initializing Patents collector: {e}")
    

        try:
            collectors["semantic_scholar"] = SemanticScholarCollector()
            self.logger.info("Semantic Scholar collector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Semantic Scholar collector: {e}")
        
        # Initialize IEEE collector if available
        if IEEE_AVAILABLE:
            try:
                collectors["ieee"] = IEEECollector()
                self.logger.info("IEEE collector initialized")
            except Exception as e:
                self.logger.error(f"Error initializing IEEE collector: {e}")
                
        self.logger.info(f"Initialized {len(collectors)} collectors: {', '.join(collectors.keys())}")

        try:
            # Get email from environment variable or use a default
            email = os.getenv("ENTREZ_EMAIL", "your-email@example.com")
            collectors["pubmed"] = PubMedCollector(email=email)
            self.logger.info(f"PubMed collector initialized with email: {email}")
        except Exception as e:
            self.logger.error(f"Error initializing PubMed collector: {e}")

        return collectors
    
    def run_collection(self, sources: List[str] = None, custom_params: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Run data collection from specified sources
        
        Args:
            sources: List of sources to collect from (default: all available)
            custom_params: Dict of custom parameters for each source
            
        Returns:
            Dict containing results for each source
        """
        # Determine which sources to use
        if not sources:
            sources = list(self.collectors.keys())
        else:
            # Filter to only available collectors
            sources = [s for s in sources if s in self.collectors]
            
        if not sources:
            self.logger.error("No valid sources specified for collection")
            return {}
            
        self.logger.info(f"Running collection for sources: {', '.join(sources)}")
        
        # Merge default and custom parameters
        params = {}
        for source in sources:
            # Start with default parameters
            source_params = SOURCE_PARAMS.get(source, {}).copy()
            
            # Override with custom parameters if provided
            if custom_params and source in custom_params:
                source_params.update(custom_params[source])
                
            params[source] = source_params
            
        # Run collection
        if self.parallel and len(sources) > 1:
            self.results = self._run_parallel_collection(sources, params)
        else:
            self.results = self._run_sequential_collection(sources, params)
            
        # Save collection report
        self._save_collection_report()
        
        return self.results
    
    def _run_sequential_collection(self, sources: List[str], params: Dict[str, Dict]) -> Dict[str, Any]:
        """Run collection sequentially for each source"""
        results = {}
        
        for source in sources:
            self.logger.info(f"Collecting from {source}")
            print(f"\n{'='*60}\nCollecting from {source.upper()}\n{'='*60}")
            
            try:
                # Get collector and parameters
                collector = self.collectors[source]
                source_params = params[source]
                
                # Extract parameters
                query = source_params.get('default_query', "graphene applications")
                max_results = source_params.get('max_results', 100)
                
                # Additional parameters vary by source
                additional_params = {k: v for k, v in source_params.items() 
                                    if k not in ['default_query', 'max_results']}
                
                # Run collection with optimized parameters
                start_time = time.time()
                
                if source == "arxiv":
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "scopus":
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "patents":  # Add this block
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "semantic_scholar":
                    collector.search_items(
                        query=query,
                        num_results=max_results,
                        start_year=additional_params.get('start_year', 2010)
                    )
                elif source == "ieee" and IEEE_AVAILABLE:
                    collector.search_items(
                        query=query,
                        num_results=max_results,
                        content_type=additional_params.get('content_type', None),
                        start_year=additional_params.get('start_year', 2010),
                        end_year=additional_params.get('end_year', datetime.now().year)
                    )
                elif source == "pubmed":
                    collector.search_items(
                        query=query,  # Pass the query parameter
                        num_results=max_results
                    )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Save data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source}_{timestamp}.csv"
                output_path = os.path.join(self.output_dir, filename)
                
                # Save to source-specific directory and unified directory
                df = collector.save_data(filename)
                
                if df is not None and not df.empty:
                    # Also save to unified output directory
                    df.to_csv(output_path, index=False)
                    
                    # Record results
                    results[source] = {
                        'source': source,
                        'query': query,
                        'count': len(df),
                        'file': output_path,
                        'time': elapsed_time,
                        'timestamp': timestamp
                    }
                    
                    self.logger.info(f"Collected {len(df)} items from {source} in {elapsed_time:.2f} seconds")
                    print(f"Collected {len(df)} items from {source} in {elapsed_time:.2f} seconds")
                    print(f"Data saved to: {output_path}")
                    
                    # Run data analysis
                    collector.analyze_data(df)
                else:
                    self.logger.warning(f"No data collected from {source}")
                    results[source] = {
                        'source': source,
                        'query': query,
                        'count': 0,
                        'file': None,
                        'time': elapsed_time,
                        'timestamp': timestamp,
                        'error': 'No data collected'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {source}: {str(e)}", exc_info=True)
                results[source] = {
                    'source': source,
                    'count': 0,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
        return results
    
    def _run_parallel_collection(self, sources: List[str], params: Dict[str, Dict]) -> Dict[str, Any]:
        """Run collection in parallel for all sources"""
        results = {}
        
        # Define worker function
        def collect_from_source(source):
            self.logger.info(f"Starting parallel collection from {source}")
            
            try:
                # Get collector and parameters
                collector = self.collectors[source]
                source_params = params[source]
                
                # Extract parameters
                query = source_params.get('default_query', "graphene applications")
                max_results = source_params.get('max_results', 100)
                
                # Additional parameters vary by source
                additional_params = {k: v for k, v in source_params.items() 
                                    if k not in ['default_query', 'max_results']}
                
                # Run collection with optimized parameters
                start_time = time.time()
                
                if source == "arxiv":
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "scopus":
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "patents":  # Add this block
                    collector.search_items(
                    num_results=max_results
                    )
                elif source == "semantic_scholar":
                    collector.search_items(
                        query=query,
                        num_results=max_results,
                        start_year=additional_params.get('start_year', 2010)
                    )
                elif source == "ieee" and IEEE_AVAILABLE:
                    collector.search_items(
                        query=query,
                        num_results=max_results,
                        content_type=additional_params.get('content_type', None),
                        start_year=additional_params.get('start_year', 2010),
                        end_year=additional_params.get('end_year', datetime.now().year)
                    )
                elif source == "pubmed":
                    collector.search_items(
                        query=query,  # Pass the query parameter
                        num_results=max_results
                    )

                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Save data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source}_{timestamp}.csv"
                output_path = os.path.join(self.output_dir, filename)
                
                # Save to source-specific directory and unified directory
                df = collector.save_data(filename)
                
                if df is not None and not df.empty:
                    # Also save to unified output directory
                    df.to_csv(output_path, index=False)
                    
                    # Record results
                    result = {
                        'source': source,
                        'query': query,
                        'count': len(df),
                        'file': output_path,
                        'time': elapsed_time,
                        'timestamp': timestamp
                    }
                    
                    self.logger.info(f"Collected {len(df)} items from {source} in {elapsed_time:.2f} seconds")
                    
                    # Run data analysis
                    collector.analyze_data(df)
                else:
                    self.logger.warning(f"No data collected from {source}")
                    result = {
                        'source': source,
                        'query': query,
                        'count': 0,
                        'file': None,
                        'time': elapsed_time,
                        'timestamp': timestamp,
                        'error': 'No data collected'
                    }
                    
                return result
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {source}: {str(e)}", exc_info=True)
                return {
                    'source': source,
                    'count': 0,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
        
        # Run collection in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sources), 4)) as executor:
            # Submit tasks
            future_to_source = {executor.submit(collect_from_source, source): source 
                            for source in sources}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    results[source] = result
                    print(f"\nCompleted collection from {source.upper()}")
                    print(f"Collected {result.get('count', 0)} items")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                except Exception as e:
                    self.logger.error(f"Exception in parallel collection from {source}: {e}")
                    results[source] = {
                        'source': source,
                        'count': 0,
                        'error': str(e),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
        return results
    
    def _save_collection_report(self):
        """Save a report of the collection results"""
        if not self.results:
            self.logger.warning("No results to save in collection report")
            return
            
        # Create report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(self.results),
            'total_items': sum(r.get('count', 0) for r in self.results.values()),
            'sources': self.results,
            'output_directory': self.output_dir
        }
        
        # Save as JSON
        report_path = os.path.join(self.output_dir, 'collection_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Collection report saved to {report_path}")
        print(f"\nCollection report saved to {report_path}")
        
        # Print summary
        print("\nCollection Summary:")
        print(f"Total sources: {report['total_sources']}")
        print(f"Total items collected: {report['total_items']}")
        print("\nResults by source:")
        
        for source, result in self.results.items():
            status = "✓" if result.get('count', 0) > 0 else "✗"
            count = result.get('count', 0)
            error = f" - Error: {result.get('error')}" if 'error' in result else ""
            print(f"  {status} {source}: {count} items{error}")
    
    def merge_collections(self, output_file: str = None) -> Optional[pd.DataFrame]:
        """
        Merge all collected data into a single DataFrame
        
        Args:
            output_file: Path to save the merged data (default: merged_{timestamp}.csv in output_dir)
            
        Returns:
            DataFrame containing all merged data, or None if no data available
        """
        if not self.results:
            self.logger.warning("No results available to merge")
            return None
            
        # Collect all dataframes
        dfs = []
        
        for source, result in self.results.items():
            file_path = result.get('file')
            if file_path and os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    # Add source column if not present
                    if 'source' not in df.columns:
                        df['source'] = source
                    dfs.append(df)
                    self.logger.info(f"Added {len(df)} rows from {source}")
                except Exception as e:
                    self.logger.error(f"Error reading data from {source}: {e}")
        
        if not dfs:
            self.logger.warning("No data available to merge")
            return None
            
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Merged {len(merged_df)} total rows from {len(dfs)} sources")
        
        # Save merged data
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"merged_{timestamp}.csv")
            
        merged_df.to_csv(output_file, index=False)
        self.logger.info(f"Merged data saved to {output_file}")
        print(f"Merged data saved to {output_file}")
        
        return merged_df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified data collection for Graphene Applications project')
    
    parser.add_argument('--sources', nargs='+', default=None,
                        help='Sources to collect from (default: all available)')
    
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Run collection in parallel (default: True)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output (default: data/raw/collection_{timestamp})')
    
    parser.add_argument('--merge', action='store_true', default=True,
                        help='Merge all collected data (default: True)')
    
    parser.add_argument('--arxiv-max', type=int, default=None,
                        help='Maximum results from ArXiv (default: use SOURCE_PARAMS)')
    
    parser.add_argument('--scopus-max', type=int, default=None,
                        help='Maximum results from Scopus (default: use SOURCE_PARAMS)')
    
    parser.add_argument('--patents-max', type=int, default=None,
                    help='Maximum results from Patents (default: use SOURCE_PARAMS)')
    
    
    parser.add_argument('--semantic-max', type=int, default=None,
                        help='Maximum results from Semantic Scholar (default: use SOURCE_PARAMS)')
    
    parser.add_argument('--ieee-max', type=int, default=None,
                        help='Maximum results from IEEE (default: use SOURCE_PARAMS)')
    
    return parser.parse_args()

def main():
    """Main function to run unified collection"""
    # Parse arguments
    args = parse_args()
    
    # Initialize runner
    runner = UnifiedCollectionRunner(output_dir=args.output_dir, parallel=args.parallel)
    
    # Prepare custom parameters based on command line arguments
    custom_params = {}
    
    if args.arxiv_max:
        custom_params['arxiv'] = {'max_results': args.arxiv_max}
        
    if args.scopus_max:
        custom_params['scopus'] = {'max_results': args.scopus_max}
        
    if args.patents_max:
        custom_params['patents'] = {'max_results': args.patents_max}

    if args.semantic_max:
        custom_params['semantic_scholar'] = {'max_results': args.semantic_max}
        
    if args.ieee_max and IEEE_AVAILABLE:
        custom_params['ieee'] = {'max_results': args.ieee_max}
    
    # Run collection
    results = runner.run_collection(sources=args.sources, custom_params=custom_params)
    
    # Merge collections if requested
    if args.merge:
        merged_df = runner.merge_collections()
        if merged_df is not None:
            print(f"\nSuccessfully merged {len(merged_df)} total items")

if __name__ == "__main__":
    main()