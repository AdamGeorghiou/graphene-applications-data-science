"""
Enhanced Batch Collection Module for Graphene Applications Project

This module provides advanced batch collection capabilities with:
- Parallel processing
- Retry mechanisms
- Progress tracking
- Incremental collection
- Detailed logging and reporting
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.collectors.arxiv_collector import ArxivCollector
from src.collectors.scopus_collector import ScopusCollector
from src.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.collectors.processed_patents_collector import ProcessedPatentsCollector
from src.collectors.pubmed_collector import PubMedCollector
from src.collectors.science_direct_collector import ScienceDirectCollector

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Constants
DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEFAULT_PARALLEL_WORKERS = 4

class BatchCollectionManager:
    """
    Advanced batch collection manager with parallel processing, 
    retry mechanisms, and detailed logging.
    """
    
    def __init__(self, 
                output_dir: Optional[str] = None,
                parallel: bool = True,
                max_workers: int = DEFAULT_PARALLEL_WORKERS,
                incremental: bool = True,
                log_level: int = logging.INFO):
        """
        Initialize the batch collection manager
        
        Args:
            output_dir: Directory to save output files
            parallel: Whether to run collections in parallel
            max_workers: Maximum number of parallel workers
            incremental: Whether to perform incremental collection
            log_level: Logging level
        """
        self.setup_logging(log_level)
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(project_root, "data", "raw", f"batch_{timestamp}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Configuration
        self.parallel = parallel
        self.max_workers = max_workers
        self.incremental = incremental
        
        # Collection status tracking
        self.results = {}
        self.collection_stats = {
            'start_time': None,
            'end_time': None,
            'total_items': 0,
            'sources': {},
            'errors': [],
            'status': 'initialized'
        }
        
        # Initialize collectors
        self.collectors = self._initialize_collectors()
        
        # Load previous collection data if incremental
        self.previous_data = {}
        if incremental:
            self._load_previous_data()
    
    def setup_logging(self, log_level: int):
        """Set up logging configuration"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'batch_collection_{timestamp}.log')
        
        # Configure logger
        self.logger = logging.getLogger('batch_collection')
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
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
        
        # Initialize standard collectors
        collector_classes = {
            "arxiv": ArxivCollector,
            "scopus": ScopusCollector,
            "patents": ProcessedPatentsCollector,
            "semantic_scholar": SemanticScholarCollector,
            "pubmed": PubMedCollector,
            "sciencedirect": ScienceDirectCollector
        }
        
        
        
        # Initialize each collector
        for name, cls in collector_classes.items():
            try:
                if name == "pubmed":
                    # Use your actual email or set the ENTREZ_EMAIL environment variable
                    email = os.getenv("ENTREZ_EMAIL")
                    if not email:
                        email = "aaageorghiou@gmail.com"  # Replace with your actual email
                        os.environ["ENTREZ_EMAIL"] = email
                        self.logger.warning(f"ENTREZ_EMAIL environment variable not set, using default: {email}")
                    collectors[name] = cls(email=email)
                    self.logger.info(f"PubMed collector initialized with email: {email}")
                else:
                    collectors[name] = cls()
                self.logger.info(f"{name.capitalize()} collector initialized")
            except Exception as e:
                self.logger.error(f"Error initializing {name} collector: {str(e)}")
        
        self.logger.info(f"Initialized {len(collectors)} collectors: {', '.join(collectors.keys())}")
        return collectors
    
    def _load_previous_data(self):
        """Load previous collection data for incremental collection"""
        self.logger.info("Loading previous collection data for incremental collection")
        
        # Create data directories if they don't exist
        processed_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Try to load the latest merged data
        try:
            merged_files = [f for f in os.listdir(processed_dir) if f.startswith("merged_") and f.endswith(".csv")]
            if merged_files:
                # Get the latest file
                latest_file = sorted(merged_files)[-1]
                file_path = os.path.join(processed_dir, latest_file)
                
                # Load the data
                df = pd.read_csv(file_path)
                self.logger.info(f"Loaded previous data from {file_path}: {len(df)} records")
                
                # Group by source
                grouped = df.groupby('source')
                
                for source, group_df in grouped:
                    if source.lower() in self.collectors:
                        self.previous_data[source.lower()] = group_df
                        unique_ids = set(group_df['id'].unique()) if 'id' in group_df.columns else set()
                        self.logger.info(f"Loaded {len(group_df)} previous records for {source} with {len(unique_ids)} unique IDs")
        
        except Exception as e:
            self.logger.warning(f"Error loading previous data: {str(e)}")
            self.logger.warning("Will perform full collection instead of incremental")
    
    def run_batch_collection(self, 
                           sources: Optional[List[str]] = None, 
                           batch_params: Optional[Dict[str, Dict]] = None,
                           query: str = "graphene applications") -> Dict[str, Any]:
        """
        Run batch collection from specified sources
        
        Args:
            sources: List of sources to collect from (default: all available)
            batch_params: Dict of source-specific parameters
            query: Base query to use across all sources
            
        Returns:
            Dict containing collection results
        """
        # Start timing
        self.collection_stats['start_time'] = datetime.now().isoformat()
        self.collection_stats['status'] = 'running'
        
        # Determine sources to use
        if sources is None:
            sources = list(self.collectors.keys())
        else:
            sources = [s.lower() for s in sources if s.lower() in self.collectors]
        
        if not sources:
            self.logger.error("No valid sources specified for collection")
            self.collection_stats['status'] = 'failed'
            self.collection_stats['end_time'] = datetime.now().isoformat()
            return {}
        
        self.logger.info(f"Running batch collection for sources: {', '.join(sources)}")
        self.collection_stats['sources'] = {source: {'status': 'pending'} for source in sources}
        
        # Prepare batch parameters
        if batch_params is None:
            batch_params = {}
        
        # Default parameters per source
        default_params = {
            "arxiv": {
                "query": query,
                "batch_size": 100,
                "max_results": 500,
                "categories": ["cond-mat.mtrl-sci", "physics.app-ph"]
            },
            "scopus": {
                "query": query,
                "batch_size": 50,
                "max_results": 500
            },
            "patents": {
                "query": query,
                "batch_size": 50,
                "max_results": 500
            },
            "semantic_scholar": {
                "query": query,
                "batch_size": 100,
                "max_results": 3500,
                "start_year": 2010
            },
            "pubmed": {
                "query": query,
                "batch_size": 50,
                "max_results": 500
            },
            "sciencedirect": {
                "query": query,
                "batch_size": 50,
                "max_results": 500
            }
        }
        
        # Merge default and provided parameters
        params = {}
        for source in sources:
            params[source] = default_params.get(source, {}).copy()
            if source in batch_params:
                params[source].update(batch_params[source])
        
        # Run collection
        if self.parallel and len(sources) > 1:
            results = self._run_parallel_batch(sources, params)
        else:
            results = self._run_sequential_batch(sources, params)
        
        # Finalize collection
        self.results = results
        self.collection_stats['end_time'] = datetime.now().isoformat()
        self.collection_stats['status'] = 'completed'
        self.collection_stats['total_items'] = sum(r.get('count', 0) for r in results.values())
        
        # Save collection report
        self._save_collection_report()
        
        return results
    
    def _run_sequential_batch(self, sources: List[str], params: Dict[str, Dict]) -> Dict[str, Any]:
        """Run collection sequentially for each source"""
        results = {}
        
        for source in sources:
            self.logger.info(f"Collecting from {source}")
            print(f"\n{'='*60}\nCollecting from {source.upper()}\n{'='*60}")
            
            self.collection_stats['sources'][source]['status'] = 'running'
            self.collection_stats['sources'][source]['start_time'] = datetime.now().isoformat()
            
            try:
                result = self._collect_from_source_with_retry(source, params[source])
                results[source] = result
                self.collection_stats['sources'][source].update(result)
                self.collection_stats['sources'][source]['status'] = 'completed'
            except Exception as e:
                self.logger.error(f"Error collecting from {source}: {str(e)}")
                results[source] = {
                    'source': source,
                    'count': 0,
                    'error': str(e),
                    'status': 'failed'
                }
                self.collection_stats['sources'][source]['status'] = 'failed'
                self.collection_stats['sources'][source]['error'] = str(e)
                self.collection_stats['errors'].append({
                    'source': source,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            self.collection_stats['sources'][source]['end_time'] = datetime.now().isoformat()
        
        return results
    
    def _run_parallel_batch(self, sources: List[str], params: Dict[str, Dict]) -> Dict[str, Any]:
        """Run collection in parallel for all sources"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sources), self.max_workers)) as executor:
            # Submit tasks
            future_to_source = {}
            for source in sources:
                self.collection_stats['sources'][source]['status'] = 'running'
                self.collection_stats['sources'][source]['start_time'] = datetime.now().isoformat()
                
                future = executor.submit(self._collect_from_source_with_retry, source, params[source])
                future_to_source[future] = source
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_source), 
                            total=len(future_to_source),
                            desc="Processing sources"):
                source = future_to_source[future]
                self.collection_stats['sources'][source]['end_time'] = datetime.now().isoformat()
                
                try:
                    result = future.result()
                    results[source] = result
                    self.collection_stats['sources'][source].update(result)
                    self.collection_stats['sources'][source]['status'] = 'completed'
                    
                    print(f"\nCompleted collection from {source.upper()}")
                    print(f"Collected {result.get('count', 0)} items")
                except Exception as e:
                    self.logger.error(f"Exception in parallel collection from {source}: {e}")
                    results[source] = {
                        'source': source,
                        'count': 0,
                        'error': str(e),
                        'status': 'failed'
                    }
                    self.collection_stats['sources'][source]['status'] = 'failed'
                    self.collection_stats['sources'][source]['error'] = str(e)
                    self.collection_stats['errors'].append({
                        'source': source,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        return results
    
    def _collect_from_source_with_retry(self, source: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect data from a source with retry mechanism
        
        Args:
            source: Source name
            params: Collection parameters
            
        Returns:
            Collection result information
        """
        collector = self.collectors[source]
        retries = 0
        query = params.get('query', 'graphene applications')
        max_results = params.get('max_results', 100)
        batch_size = params.get('batch_size', 50)
        self.logger.info(f"Parameters for {source}: query='{query}', max_results={max_results}, batch_size={batch_size}")
        while retries <= MAX_RETRIES:
            try:
                start_time = time.time()
                
                # Handle source-specific collection logic
                # Replace with:
                if source == "arxiv":
                    # Check your actual ArxivCollector implementation to see what parameters it accepts
                    collector.search_items(
                        num_results=max_results
                        # Remove the categories parameter or adapt to match your implementation
                    )
                elif source == "scopus":
                    collector.search_items(
                        num_results=max_results
                    )
                elif source == "patents":
                    collector.search_items(
                        num_results=max_results
                    )
                elif source == "semantic_scholar":
                    collector.search_items(
                        query=query,
                        num_results=max_results,
                        start_year=params.get('start_year', 2010)
                    )
                
                elif source == "pubmed":
                    collector.search_items(
                        query=query,
                        num_results=max_results
                    )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Save data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source}_{timestamp}.csv"
                output_path = os.path.join(self.output_dir, filename)
                self.logger.info(f"Will save {source} data to: {output_path}")
                # Apply incremental filtering if enabled
                if self.incremental and source in self.previous_data:
                    self._apply_incremental_filtering(collector, source)
                
                # Save collected data
                df = collector.save_data(filename)
                
                if df is not None and not df.empty:
                    # Also save to output directory
                    df.to_csv(output_path, index=False)
                    
                    # Record results
                    result = {
                        'source': source,
                        'query': query,
                        'count': len(df),
                        'file': output_path,
                        'elapsed_time': elapsed_time,
                        'timestamp': timestamp,
                        'status': 'completed'
                    }
                    
                    self.logger.info(f"Collected {len(df)} items from {source} in {elapsed_time:.2f} seconds")
                    
                    # Run data analysis
                    analysis = collector.analyze_data(df)
                    if analysis:
                        result['analysis'] = analysis
                    
                    return result
                else:
                    self.logger.warning(f"No data collected from {source}")
                    return {
                        'source': source,
                        'query': query,
                        'count': 0,
                        'elapsed_time': elapsed_time,
                        'timestamp': timestamp,
                        'error': 'No data collected',
                        'status': 'completed'
                    }
            
            except Exception as e:
                retries += 1
                if retries <= MAX_RETRIES:
                    retry_wait = RETRY_DELAY * retries
                    self.logger.warning(f"Retry {retries}/{MAX_RETRIES} for {source} after error: {str(e)}")
                    self.logger.warning(f"Waiting {retry_wait} seconds before retrying...")
                    time.sleep(retry_wait)
                else:
                    self.logger.error(f"Failed to collect from {source} after {MAX_RETRIES} retries")
                    raise
        
        # This should not be reached due to the exception in the last retry
        self.logger.info(f"Collecting from {source} with max_results={max_results}")
        return {'source': source, 'status': 'failed', 'error': 'Unknown error'}
    
    def _apply_incremental_filtering(self, collector, source):
        """
        Filter out previously collected items for incremental collection
        
        Args:
            collector: Source collector
            source: Source name
        """
        if not hasattr(collector, 'data') or not collector.data:
            return
        
        try:
            self.logger.info(f"Applying incremental filtering to {source}, starting with {len(collector.data)} items")

            
            prev_df = self.previous_data.get(source)
            if prev_df is None or prev_df.empty:
                return
            
            # Extract unique identifiers from previous data
            if 'id' in prev_df.columns:
                prev_ids = set(prev_df['id'].unique())
            elif 'doi' in prev_df.columns:
                prev_ids = set(prev_df['doi'].unique())
            else:
                # Try to use a combination of title and published_date as identifier
                prev_ids = set(prev_df.apply(
                    lambda x: f"{x.get('title', '')}-{x.get('published_date', '')}", 
                    axis=1
                ).unique())
            
            # Filter out previously collected items
            new_data = []
            for item in collector.data:
                item_id = item.get('id') or item.get('doi')
                if not item_id:
                    # Use title-date combination
                    item_id = f"{item.get('title', '')}-{item.get('published_date', '')}"
                
                if item_id not in prev_ids:
                    new_data.append(item)
            
            # Update collector data with filtered items
            filtered_count = len(collector.data) - len(new_data)
            collector.data = new_data
            
            if filtered_count > 0:
                self.logger.info(f"Incremental collection: filtered out {filtered_count} previously collected items from {source}")
            self.logger.info(f"After incremental filtering: {len(collector.data)} items remain for {source}")

        
        except Exception as e:
            self.logger.error(f"Error applying incremental filtering for {source}: {str(e)}")
            # Continue with all data if filtering fails
    
    def _save_collection_report(self):
        """Save a report of the collection results"""
        if not self.collection_stats:
            self.logger.warning("No collection stats to save in report")
            return
        
        # Create report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(self.results),
            'total_items': self.collection_stats['total_items'],
            'sources': self.collection_stats['sources'],
            'status': self.collection_stats['status'],
            'start_time': self.collection_stats['start_time'],
            'end_time': self.collection_stats['end_time'],
            'errors': self.collection_stats['errors'],
            'output_directory': self.output_dir,
            'incremental': self.incremental
        }
        
        # Save as JSON
        report_path = os.path.join(self.output_dir, 'batch_collection_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Collection report saved to {report_path}")
        print(f"\nCollection report saved to {report_path}")
        
        # Print summary
        print("\nCollection Summary:")
        print(f"Total sources: {report['total_sources']}")
        print(f"Total items collected: {report['total_items']}")
        print("\nResults by source:")
        
        for source, result in self.collection_stats['sources'].items():
            status_icon = "✓" if result.get('status') == 'completed' else "✗"
            count = result.get('count', 0)
            error_msg = f" - Error: {result.get('error')}" if 'error' in result else ""
            print(f"  {status_icon} {source}: {count} items{error_msg}")
    
    def merge_collections(self, output_file: str = None) -> Optional[pd.DataFrame]:
        """
        Merge all collected data into a single DataFrame
        
        Args:
            output_file: Path to save the merged data
            
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
            processed_dir = os.path.join(project_root, "data", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, f"merged_{timestamp}.csv")
        
        merged_df.to_csv(output_file, index=False)
        self.logger.info(f"Merged data saved to {output_file}")
        print(f"Merged data saved to {output_file}")
        
        return merged_df


def run_batch_collection(sources=None, 
                    query="graphene applications", 
                    max_results=100,
                    output_dir=None,
                    parallel=True,
                    incremental=True):
    """
    Run batch collection with simplified parameters
    
    Args:
        sources: List of sources to collect from (default: all available)
        query: Query string to use for searches
        max_results: Maximum results per source
        output_dir: Output directory
        parallel: Whether to run in parallel
        incremental: Whether to use incremental collection
    
    Returns:
        Merged DataFrame of all collected data
    """
    # Initialize batch collection manager
    manager = BatchCollectionManager(
        output_dir=output_dir,
        parallel=parallel,
        incremental=incremental
    )
    
    # Set up batch parameters
    batch_params = {}
    for source in manager.collectors.keys():
        batch_params[source] = {
            'query': query,
            'max_results': max_results
        }
    
    # Run collection
    manager.run_batch_collection(sources=sources, batch_params=batch_params)
    
    # Merge and return the data
    return manager.merge_collections()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch data collection for Graphene Applications project")
    parser.add_argument("--sources", nargs="+", help="Sources to collect from (default: all)")
    parser.add_argument("--query", type=str, default="graphene applications", help="Search query")
    parser.add_argument("--max-results", type=int, default=200, help="Maximum results per source")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel collection")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental collection")
    
    args = parser.parse_args()
    
    merged_df = run_batch_collection(
        sources=args.sources,
        query=args.query,
        max_results=args.max_results,
        output_dir=args.output_dir,
        parallel=not args.no_parallel,
        incremental=not args.no_incremental
    )
    
    if merged_df is not None:
        print(f"\nSuccessfully collected and merged {len(merged_df)} items")
        print(f"Sources represented: {merged_df['source'].value_counts().to_dict()}")