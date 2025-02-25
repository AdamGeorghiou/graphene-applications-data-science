#!/usr/bin/env python3
"""
Simple collection script that works with your existing collectors
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

# Import the collectors with their actual class names
from src.collectors.arxiv_collector import GrapheneResearchCollector
from src.collectors.scopus_collector import ScopusCollector
from src.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.collectors.patent_collector import GooglePatentCollector

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"data/raw/collection_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

try:
    # ArXiv collection
    print("\n" + "="*80)
    print("COLLECTING FROM ARXIV")
    print("="*80)
    arxiv = GrapheneResearchCollector()
    arxiv.search_items(num_results=200)
    arxiv.save_data(f"arxiv_{timestamp}.csv")
    
    # Scopus collection
    print("\n" + "="*80)
    print("COLLECTING FROM SCOPUS")
    print("="*80)
    scopus = ScopusCollector()
    scopus.search_items(num_results=200)
    scopus.save_data(f"scopus_{timestamp}.csv")
    
    # Semantic Scholar collection
    print("\n" + "="*80)
    print("COLLECTING FROM SEMANTIC SCHOLAR")
    print("="*80)
    semantic = SemanticScholarCollector()
    semantic.search_items(num_results=200)
    semantic.save_data(f"semantic_{timestamp}.csv")
    
    # Patents collection
    print("\n" + "="*80)
    print("COLLECTING FROM PATENTS")
    print("="*80)
    patents = GooglePatentCollector()
    patents.search_items(num_results=150)
    patents.save_data(f"patents_{timestamp}.csv")
    
except Exception as e:
    print(f"Error during collection: {e}")
    import traceback
    traceback.print_exc()

print("\nCollection complete!")