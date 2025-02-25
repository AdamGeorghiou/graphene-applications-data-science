"""
Collection Configuration for Graphene Applications Project

This file contains optimized parameters for each data source and
configuration options for the collection process.
"""

import os
from typing import Dict, Any

# Source-specific optimized parameters
SOURCE_PARAMS = {
    "arxiv": {
        "default_query": "graphene AND (application OR applications OR use OR uses OR device OR devices OR technology)",
        "categories": ["cond-mat.mtrl-sci", "cond-mat.mes-hall", "physics.app-ph"],
        "sort_by": "submittedDate",
        "sort_order": "descending",
        "max_results": 200,
        "filters": {
            "year_min": 2010,
        }
    },
    "scopus": {
        "default_query": "TITLE-ABS-KEY(graphene AND (application OR applications OR use OR uses OR device OR technology))",
        "view": "COMPLETE",
        "sort": "-coverDate",
        "max_results": 200,
        "filters": {
            "subject_areas": ["MATE", "PHYS", "CHEM", "ENGI"],  # Materials, Physics, Chemistry, Engineering
            "document_types": ["ar", "cp", "re"],  # Articles, Conference Papers, Reviews
            "languages": ["English"],
            "access_types": ["All"]
        }
    },
    "patents": {
        "default_query": "graphene applications",
        "country_codes": ["US", "EP", "WO", "CN", "JP", "KR"],
        "max_results": 150,
        "sort": "recent",
        "filters": {
            "filing_date_range": ["2010-01-01", None],  # None means today
            "patent_type": ["utility", "design"],
            "status": ["ACTIVE", "EXPIRED"]
        }
    },
    "semantic_scholar": {
        "default_query": "graphene applications",
        "fields": "paperId,title,abstract,year,authors,venue,publicationDate,externalIds,fieldsOfStudy,s2FieldsOfStudy,citationCount,openAccessPdf,tldr",
        "max_results": 200,
        "start_year": 2010,
        "filters": {
            "publication_types": ["JournalArticle", "Conference"],
            "open_access_only": False,
            "venues": [],  # Leave empty for all venues
            "min_citation_count": 0
        }
    },
    "ieee": {
        "default_query": "graphene AND (application OR applications OR uses OR device OR technology OR implementation)",
        "content_type": "Conferences,Journals,Early Access,Standards",
        "max_results": 200,
        "sort_order": "desc",
        "sort_field": "publication_date",
        "filters": {
            "start_year": 2010,
            "end_year": 2024,
            "publisher": "",  # Empty for all publishers
            "content_type": ["Journals", "Conferences", "Early Access", "Standards"],
            "open_access": False  # Set to True for open access only
        }
    }
}

# Domain-specific keyword configurations for better results
DOMAIN_KEYWORDS = {
    "energy": [
        "battery", "supercapacitor", "energy storage", "solar cell", 
        "photovoltaic", "fuel cell", "energy harvesting"
    ],
    "electronics": [
        "transistor", "sensor", "electrode", "conductor", "semiconductor", 
        "circuit", "display", "touchscreen", "flexible electronics"
    ],
    "materials": [
        "composite", "film", "coating", "membrane", "filter",
        "reinforcement", "additive", "filler", "barrier"
    ],
    "biomedical": [
        "drug delivery", "biosensor", "tissue engineering", "biomedical",
        "antibacterial", "biocompatible", "implant", "therapeutic"
    ],
    "environmental": [
        "water treatment", "gas separation", "water purification",
        "environmental remediation", "pollution control", "desalination"
    ],
    "chemical": [
        "catalyst", "electrocatalyst", "photocatalyst", "oxidation",
        "reduction", "chemical conversion", "synthesis"
    ]
}

# Generate expanded queries for better coverage
def generate_expanded_query(base_query: str, domain: str = None) -> str:
    """
    Generate an expanded query with domain-specific keywords
    
    Args:
        base_query: Base query string
        domain: Specific domain to focus on (optional)
        
    Returns:
        Expanded query string
    """
    if domain and domain in DOMAIN_KEYWORDS:
        # Create domain-specific query
        domain_terms = " OR ".join([f'"{term}"' for term in DOMAIN_KEYWORDS[domain]])
        return f"({base_query}) AND ({domain_terms})"
    
    return base_query

# Collection process configuration
COLLECTION_CONFIG = {
    "parallel_execution": True,
    "max_workers": 4,
    "rate_limiting": {
        "arxiv": 3.0,  # seconds between requests
        "scopus": 1.0,
        "patents": 2.0,
        "semantic_scholar": 1.0,
        "ieee": 1.0
    },
    "retry_config": {
        "max_retries": 3,
        "retry_delay": 5.0,  # seconds between retries
        "backoff_factor": 2.0  # exponential backoff factor
    },
    "timeout": {
        "connect": 10.0,  # connection timeout in seconds
        "read": 30.0      # read timeout in seconds
    },
    "merge_output": True,  # automatically merge all collections
    "save_metadata": True  # save collection metadata
}

# Domain-specific expanded queries
EXPANDED_QUERIES = {
    "energy_applications": generate_expanded_query("graphene", "energy"),
    "electronic_applications": generate_expanded_query("graphene", "electronics"),
    "materials_applications": generate_expanded_query("graphene", "materials"),
    "biomedical_applications": generate_expanded_query("graphene", "biomedical"),
    "environmental_applications": generate_expanded_query("graphene", "environmental"),
    "chemical_applications": generate_expanded_query("graphene", "chemical"),
    "all_applications": "graphene AND (application OR applications OR use OR uses OR device OR technology OR implementation)"
}

# File paths configuration
def get_output_paths(base_dir: str = None) -> Dict[str, str]:
    """
    Get standardized output file paths
    
    Args:
        base_dir: Base directory for outputs (optional)
        
    Returns:
        Dictionary of output paths
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    return {
        "raw_dir": os.path.join(base_dir, "raw"),
        "processed_dir": os.path.join(base_dir, "processed"),
        "merged_file": os.path.join(base_dir, "raw", "merged", "merged_collection.csv"),
        "metadata_file": os.path.join(base_dir, "raw", "collection_metadata.json"),
        "log_dir": os.path.join(os.path.dirname(base_dir), "logs")
    }