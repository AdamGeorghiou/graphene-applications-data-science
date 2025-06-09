import os
import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel

class BaseNLPProcessor:
    """Base class for all NLP processors with common functionality"""
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        self.model_name = model_name

        # ✅ Improved device selection logic
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.logger = self._setup_logging()
        self.logger.info(f"Using device: {self.device}")

        # ✅ Model & tokenizer setup (unchanged)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for this processor"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def get_embeddings(self, text: str, max_length: int = 512):
        """Get embeddings for text using the initialized model"""
        if not text or len(text.strip()) == 0:
            return None
            
        # ✅ Tokenize and move each tensor to device safely
        inputs = self.tokenizer(
            text,
            return_tensors="pt", 
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # key change here

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]
    
    def process(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process()")