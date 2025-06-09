import os

# Base project directory
PROJECT_DIR = '/Users/adamgeorghiou/Desktop/GIM/Project'

# Data directories
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Source-specific directories
ARXIV_DIR = os.path.join(RAW_DATA_DIR, 'arxiv')
SCOPUS_DIR = os.path.join(RAW_DATA_DIR, 'scopus')
PATENTS_DIR = os.path.join(RAW_DATA_DIR, 'patents')
IEEE_DIR = os.path.join(RAW_DATA_DIR, 'ieee')
SEMANTIC_SCHOLAR_DIR = os.path.join(RAW_DATA_DIR, 'semantic_scholar')

# Logs directory
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
                ARXIV_DIR, SCOPUS_DIR, PATENTS_DIR, IEEE_DIR, SEMANTIC_SCHOLAR_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Settings
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')
IEEE_API_KEY = os.getenv('IEEE_API_KEY')
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')

# Data collection settings
DEFAULT_BATCH_SIZE = 50
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 1  # seconds

# Validation settings
REQUIRED_FIELDS = ['title', 'abstract', 'published_date', 'collection_date', 'source']

# Graphene application keywords for NLP processing
GRAPHENE_APPLICATION_KEYWORDS = [
    # Energy applications
    'battery', 'supercapacitor', 'solar cell', 'energy storage',
    'fuel cell', 'photovoltaic', 'energy harvesting',
    
    # Electronic applications
    'transistor', 'semiconductor', 'conductor', 'electronic', 
    'circuit', 'sensor', 'biosensor', 'display', 'electrode',
    'touchscreen', 'flexible electronics',
    
    # Material applications
    'composite', 'coating', 'membrane', 'filter', 'barrier',
    'reinforcement', 'additive', 'nanomaterial',
    
    # Chemical applications
    'catalyst', 'photocatalyst', 'electrocatalyst', 'oxidation',
    'reduction', 'chemical conversion',
    
    # Biological applications
    'drug delivery', 'tissue engineering', 'biomedical',
    'biosensing', 'cellular', 'antibacterial',
    
    # Environmental applications
    'water treatment', 'gas separation', 'pollution control',
    'environmental remediation', 'water purification'
]