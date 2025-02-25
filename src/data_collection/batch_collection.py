#!/usr/bin/env python3
"""
Batch Collection Script for Graphene Applications Project

This script provides a simplified interface for running collections
at different scales (small, medium, large) for testing and production.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Collection size presets
COLLECTION_SIZES = {
    'small': {
        'arxiv': 50,
        'scopus': 50,
        'patents': 50,
        'semantic_scholar': 50,
        'ieee': 50
    },
    'medium': {
        'arxiv': 200,
        'scopus': 200,
        'patents': 150,
        'semantic_scholar': 200,
        'ieee': 200
    },
    'large': {
        'arxiv': 500,
        'scopus': 500,
        'patents': 300,
        'semantic_scholar': 500,
        'ieee': 500
    }
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch data collection for Graphene Applications project')
    
    parser.add_argument('--size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Collection size preset (default: medium)')
    
    parser.add_argument('--sources', nargs='+', default=None,
                        help='Sources to collect from (default: all available)')
    
    parser.add_argument('--mode', type=str, default='general',
                        choices=['general', 'domain', 'balanced'],
                        help='Collection mode (default: general)')
    
    parser.add_argument('--domains', nargs='+', default=['all'],
                        choices=['all', 'energy', 'electronics', 'materials', 'biomedical', 'environmental', 'chemical'],
                        help='Domains to collect in domain mode (default: all)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output (default: auto-generated)')
    
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel collection')
    
    return parser.parse_args()

def run_general_collection(size, sources=None, parallel=True, output_dir=None):
    """Run general collection with specified size preset"""
    print(f"\n{'='*80}")
    print(f"RUNNING GENERAL COLLECTION - SIZE: {size.upper()}")
    print(f"{'='*80}\n")
    
    # Create command
    cmd = [sys.executable, os.path.join(project_root, "src", "data_collection", "unified_collector.py")]
    
    # Add sources if specified
    if sources:
        cmd.extend(["--sources"] + sources)
    
    # Add parallel flag if enabled
    if parallel:
        cmd.append("--parallel")
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    # Add size-specific parameters
    size_params = COLLECTION_SIZES[size]
    param_name_map = {
        'arxiv': 'arxiv',
        'scopus': 'scopus', 
        'patents': 'patents',
        'semantic_scholar': 'semantic',
        'ieee': 'ieee'
    }
    for source, count in size_params.items():
        param_name = f"--{param_name_map.get(source, source)}-max"
        cmd.extend([param_name, str(count)])
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_domain_collection(size, domains, sources=None, parallel=True, output_dir=None):
    """Run domain-specific collection with specified size preset"""
    print(f"\n{'='*80}")
    print(f"RUNNING DOMAIN-SPECIFIC COLLECTION - SIZE: {size.upper()}")
    print(f"{'='*80}\n")
    
    # Determine max results per domain based on size
    # For domain collections, we adjust the count to be per domain
    size_multiplier = {'small': 0.3, 'medium': 1.0, 'large': 2.5}
    max_results = int(100 * size_multiplier[size])
    
    # Create command
    cmd = [sys.executable, os.path.join(project_root, "src", "data_collection", "domain_specific_collector.py")]
    
    # Add domains
    cmd.extend(["--domains"] + domains)
    
    # Add sources if specified
    if sources:
        cmd.extend(["--sources"] + sources)
    
    # Add parallel flag if enabled
    if parallel:
        cmd.append("--parallel")
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    # Add max results
    cmd.extend(["--max-results", str(max_results)])
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_balanced_collection(size, sources=None, parallel=True, output_dir=None):
    """
    Run balanced collection that includes both general and domain-specific collections
    
    This is a hybrid approach that first collects general data and then
    supplements it with domain-specific collections.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING BALANCED COLLECTION - SIZE: {size.upper()}")
    print(f"{'='*80}\n")
    
    # Create timestamp for consistent output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "raw", f"balanced_collection_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    general_dir = os.path.join(output_dir, "general")
    domain_dir = os.path.join(output_dir, "domains")
    
    os.makedirs(general_dir, exist_ok=True)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Adjust size for balanced approach
    general_size = {'small': 'small', 'medium': 'small', 'large': 'medium'}[size]
    domain_size = {'small': 'small', 'medium': 'small', 'large': 'small'}[size]
    
    # Run general collection
    run_general_collection(general_size, sources, parallel, general_dir)
    
    # Run domain-specific collection
    run_domain_collection(domain_size, ['energy', 'electronics', 'materials'], sources, parallel, domain_dir)
    
    print(f"\nBalanced collection complete!")
    print(f"Output directory: {output_dir}")

def main():
    """Main function to run batch collection"""
    # Parse arguments
    args = parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            project_root, "data", "raw", 
            f"{args.mode}_collection_{args.size}_{timestamp}"
        )
    
    # Run appropriate collection mode
    if args.mode == 'general':
        run_general_collection(
            args.size, 
            args.sources, 
            not args.no_parallel, 
            args.output_dir
        )
    elif args.mode == 'domain':
        run_domain_collection(
            args.size, 
            args.domains, 
            args.sources, 
            not args.no_parallel, 
            args.output_dir
        )
    elif args.mode == 'balanced':
        run_balanced_collection(
            args.size, 
            args.sources, 
            not args.no_parallel, 
            args.output_dir
        )
    
    print("\nBatch collection complete!")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()