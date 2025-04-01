# src/processors/nlp_pipeline.py
import os
import sys
import pandas as pd
import logging
import torch
from typing import Dict, Any

# Add absolute paths to make imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
nlp_dir = os.path.join(current_dir, "nlp")

sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
sys.path.insert(0, nlp_dir)

# Try direct import
try:
    from nlp.document_processor import GrapheneDocumentProcessor
except ImportError:
    # Fallback to absolute import
    try:
        from src.processors.nlp.document_processor import GrapheneDocumentProcessor
    except ImportError:
        # Last resort - load the file directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "document_processor", 
            os.path.join(nlp_dir, "document_processor.py")
        )
        document_processor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(document_processor)
        GrapheneDocumentProcessor = document_processor.GrapheneDocumentProcessor

def main():
    """
    Main entry point for the NLP pipeline
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(project_root, 'logs', 'nlp_pipeline.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set up paths
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    output_dir = os.path.join(project_root, 'data', 'nlp_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    use_gpu = device != "cpu"
    logger.info(f"Selected device: {device}")
    logger.info(f"GPU available: {use_gpu}")
    
    # Initialize the document processor
    try:
        processor = GrapheneDocumentProcessor(use_gpu=use_gpu, output_dir=output_dir)
        processor.device = device  # Explicit override if needed (optional)
        logger.info("Document processor initialized successfully")
        
        # Load input data
        input_path = os.path.join(processed_data_dir, 'test_sample.csv')
        if not os.path.exists(input_path):
            logger.error(f"Input data not found at {input_path}")
            return
        
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} documents")
        
        # Process the data
        logger.info("Starting document processing")
        results, summary = processor.process_collection(df)
        
        # Log completion
        if results and summary:
            logger.info(f"Processing complete. Processed {len(results)} documents")
            logger.info(f"Results and summary saved to {output_dir}")
        else:
            logger.error("Processing failed or no results returned")
    
    except Exception as e:
        logger.error(f"Error in NLP pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()