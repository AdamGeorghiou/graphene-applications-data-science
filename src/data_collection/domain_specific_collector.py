#!/usr/bin/env python3
"""
Domain-Specific Collection Runner for Graphene Applications Project

This script collects graphene application data for specific domains
such as energy, electronics, materials, biomedical, environmental, and chemical.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import configuration
from src.config.collection_config import (
    SOURCE_PARAMS, 
    DOMAIN_KEYWORDS,
    EXPANDED_QUERIES,
    generate_expanded_query
)

# Import unified collector
from src.data_collection.unified_collector import UnifiedCollectionRunner

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Domain-specific data collection for Graphene Applications project')
    
    parser.add_argument('--domains', nargs='+', default=['all'],
                        choices=['all', 'energy', 'electronics', 'materials', 'biomedical', 'environmental', 'chemical'],
                        help='Domains to collect data for (default: all)')
    
    parser.add_argument('--sources', nargs='+', default=None,
                        help='Sources to collect from (default: all available)')
    
    parser.add_argument('--max-results', type=int, default=100,
                        help='Maximum results per domain per source (default: 100)')
    
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Run collection in parallel (default: True)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output (default: data/raw/domain_collection_{timestamp})')
    
    return parser.parse_args()

def run_domain_collections(domains: List[str], sources: Optional[List[str]] = None, 
                          max_results: int = 100, parallel: bool = True, 
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run data collection for specific domains
    
    Args:
        domains: List of domains to collect data for
        sources: List of sources to collect from (default: all available)
        max_results: Maximum results per domain per source
        parallel: Whether to run collection in parallel
        output_dir: Directory to save output
        
    Returns:
        Dictionary of results for each domain
    """
    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "data", "raw", f"domain_collection_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Expand 'all' domains to all available domains
    if 'all' in domains:
        domains = list(DOMAIN_KEYWORDS.keys())
    
    print(f"Running collection for domains: {', '.join(domains)}")
    
    # Initialize results dictionary
    results = {}
    
    # Run collection for each domain
    for domain in domains:
        print(f"\n{'='*80}")
        print(f"COLLECTING DATA FOR DOMAIN: {domain.upper()}")
        print(f"{'='*80}\n")
        
        # Create domain-specific output directory
        domain_output_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # Generate domain-specific queries
        custom_params = {}
        
        for source in SOURCE_PARAMS.keys():
            # Start with default parameters
            source_params = SOURCE_PARAMS[source].copy()
            
            # Update query with domain-specific query
            base_query = source_params.get('default_query', "graphene")
            domain_query = generate_expanded_query(base_query, domain)
            
            # Format query according to source requirements
            if source == "scopus":
                # Scopus uses specific query syntax
                domain_terms = " OR ".join([f'"{term}"' for term in DOMAIN_KEYWORDS[domain]])
                domain_query = f"TITLE-ABS-KEY(graphene AND ({domain_terms}))"
            
            # Update parameters
            source_params['default_query'] = domain_query
            source_params['max_results'] = max_results
            
            custom_params[source] = source_params
        
        # Initialize domain-specific collector
        collector = UnifiedCollectionRunner(output_dir=domain_output_dir, parallel=parallel)
        
        # Run collection
        domain_results = collector.run_collection(sources=sources, custom_params=custom_params)
        
        # Merge domain-specific collections
        merged_file = os.path.join(domain_output_dir, f"{domain}_merged.csv")
        merged_df = collector.merge_collections(output_file=merged_file)
        
        if merged_df is not None:
            print(f"\nSuccessfully merged {len(merged_df)} items for domain: {domain}")
            print(f"Merged data saved to: {merged_file}")
            
            domain_results['merged'] = {
                'file': merged_file,
                'count': len(merged_df)
            }
        
        # Add to overall results
        results[domain] = domain_results
    
    # Save overall results
    results_file = os.path.join(output_dir, "domain_collection_results.json")
    
    # Calculate summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'domains': domains,
        'total_domains': len(domains),
        'total_items': sum(
            sum(source.get('count', 0) for source in domain_results.values() if source != 'merged')
            for domain_results in results.values()
        ),
        'items_by_domain': {
            domain: sum(source.get('count', 0) for source in domain_results.values() if source != 'merged')
            for domain, domain_results in results.items()
        },
        'output_directory': output_dir
    }
    
    # Add summary to results
    results['summary'] = summary
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDomain collection results saved to: {results_file}")
    
    # Print summary
    print("\nDomain Collection Summary:")
    print(f"Total domains: {summary['total_domains']}")
    print(f"Total items collected: {summary['total_items']}")
    print("\nItems by domain:")
    
    for domain, count in summary['items_by_domain'].items():
        print(f"  {domain}: {count} items")
    
    return results

def main():
    """Main function to run domain-specific collection"""
    # Parse arguments
    args = parse_args()
    
    # Run domain collections
    results = run_domain_collections(
        domains=args.domains,
        sources=args.sources,
        max_results=args.max_results,
        parallel=args.parallel,
        output_dir=args.output_dir
    )
    
    print("\nDomain-specific collection complete!")

if __name__ == "__main__":
    main()