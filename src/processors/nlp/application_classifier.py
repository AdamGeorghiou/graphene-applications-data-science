from typing import Dict, List, Any
import torch
from transformers import pipeline
import logging
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .base_processor import BaseNLPProcessor

class ApplicationClassifier(BaseNLPProcessor):
    """
    Specialized processor for classifying graphene applications using zero-shot or finetuned models
    Now with GPU-accelerated batch processing support
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None, taxonomy_path=None):
        super().__init__(model_name=model_name, device=device)
        self.logger = self._setup_logging()
        
        self.use_zero_shot = False
        self.classifier = None

        try:
            if self.device == "mps":
                # Load model and tokenizer manually for MPS
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to("mps")
                self.use_zero_shot = True
            else:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                self.use_zero_shot = True

        except Exception as e:
            self.logger.warning(f"Failed to initialize zero-shot classifier: {str(e)}")
            self.logger.warning("Falling back to keyword-based classification")
        
        # Primary application categories for first-level classification
        self.primary_categories = [
            "Energy Storage & Generation",
            "Electronics & Sensors",
            "Structural Materials & Composites",
            "Biomedical & Healthcare",
            "Environmental & Water Treatment",
            "Catalysis & Chemical Processing"
        ]
        
        # Domain-specific subcategories
        self.application_subcategories = {
            "Energy Storage & Generation": [
                "Batteries", "Supercapacitors", "Fuel Cells", 
                "Solar Cells", "Photovoltaics", "Energy Harvesting"
            ],
            "Electronics & Sensors": [
                "Transistors", "Circuits", "Displays", "Touchscreens",
                "Flexible Electronics", "Conductors", "Semiconductors",
                "Chemical Sensors", "Biosensors", "Gas Sensors", "Pressure Sensors"
            ],
            "Structural Materials & Composites": [
                "Polymer Composites", "Metal Composites", "Ceramic Composites",
                "Coatings", "Membranes", "Barriers", "Reinforcements",
                "Additives", "Nanomaterials"
            ],
            "Biomedical & Healthcare": [
                "Drug Delivery", "Tissue Engineering", "Biosensing",
                "Antibacterial", "Cellular Interfaces", "Imaging",
                "Theranostics", "Neural Interfaces"
            ],
            "Environmental & Water Treatment": [
                "Water Purification", "Water Treatment", "Gas Separation",
                "Pollution Control", "Environmental Remediation",
                "CO2 Capture", "Air Filtration"
            ],
            "Catalysis & Chemical Processing": [
                "Photocatalysts", "Electrocatalysts", "Oxidation Catalysts",
                "Reduction Catalysts", "Chemical Conversion"
            ]
        }
        
        # Load application taxonomy with hierarchical categories
        self.taxonomy = self._load_taxonomy(taxonomy_path)
    
    def _setup_logging(self):
        """Set up logging for the application classifier"""
        logger = logging.getLogger("ApplicationClassifier")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _keyword_based_classification(self, text, categories, threshold=0.3):
        """Fallback method using simple keyword matching when transformer models fail"""
        text = text.lower()
        results = []
        
        for category in categories:
            category_lower = category.lower()
            words = re.findall(r'\b\w+\b', category_lower)
            
            # Count occurrences of each word
            matches = sum(1 for word in words if re.search(r'\b' + re.escape(word) + r'\b', text))
            
            # Calculate a simple confidence score based on word matches
            if matches > 0:
                # More matches and shorter category names get higher confidence
                confidence = min(0.9, matches * 0.3 / len(words))
                if confidence >= threshold:
                    results.append({"category": category, "confidence": float(confidence)})
        
        return results
    
    def _keyword_based_classification_batch(self, texts, categories, threshold=0.3):
        """Batch version of keyword matching for multiple texts"""
        batch_results = []
        
        for text in texts:
            batch_results.append(self._keyword_based_classification(text, categories, threshold))
            
        return batch_results
    
    def _load_taxonomy(self, taxonomy_path):
        """Load application taxonomy from JSON file if provided"""
        if taxonomy_path and os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r') as f:
                return json.load(f)
        
        # Default to built-in taxonomy
        return {
            "categories": self.primary_categories,
            "subcategories": self.application_subcategories
        }
    
    def classify_primary_category(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Classify the document into primary application categories"""
        # Use batch processing with a single item for consistency
        return self.classify_primary_category_batch([text], threshold)[0]
    
    def classify_primary_category_batch(self, texts: List[str], threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
        """
        Batch classify multiple documents into primary application categories
        
        Args:
            texts: List of text strings to classify
            threshold: Confidence threshold for category inclusion
            
        Returns:
            List of lists of category dictionaries
        """
        if not self.use_zero_shot:
            return self._keyword_based_classification_batch(texts, self.primary_categories, threshold)
            
        try:
            # Determine batch size based on input length
            # This helps prevent OOM errors for very large inputs
            avg_length = sum(len(text.split()) for text in texts) / len(texts)
            batch_size = min(32, max(1, int(10000 / max(1, avg_length))))
            
            # Process in smaller sub-batches if needed
            results = []
            for i in range(0, len(texts), batch_size):
                sub_batch = texts[i:i + batch_size]
                
                # Use the HF pipeline in batch mode
                sub_results = self.classifier(
                    sub_batch, 
                    self.primary_categories, 
                    multi_label=True,
                    batch_size=batch_size
                )
                
                # Handle single item vs batch results
                if not isinstance(sub_results, list):
                    sub_results = [sub_results]
                
                # Filter categories with scores above threshold
                filtered_sub_results = []
                for item in sub_results:
                    categories = []
                    for label, score in zip(item['labels'], item['scores']):
                        if score >= threshold:
                            categories.append({"category": label, "confidence": float(score)})
                    filtered_sub_results.append(categories)
                
                results.extend(filtered_sub_results)
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Zero-shot batch classification failed: {str(e)}")
            return self._keyword_based_classification_batch(texts, self.primary_categories, threshold)
    
    def classify_subcategories(self, text: str, primary_categories: List[str], 
                              threshold: float = 0.3) -> Dict[str, List[Dict[str, Any]]]:
        """Classify the document into subcategories for each primary category"""
        # Use batch processing with a single item for consistency
        return self.classify_subcategories_batch([text], [primary_categories], threshold)[0]
    
    def classify_subcategories_batch(self, texts: List[str], batch_primary_categories: List[List[str]], 
                                   threshold: float = 0.3) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Batch classify multiple documents into subcategories for each primary category
        
        Args:
            texts: List of text strings to classify
            batch_primary_categories: List of lists of primary categories for each text
            threshold: Confidence threshold for subcategory inclusion
            
        Returns:
            List of dictionaries mapping primary categories to subcategories
        """
        batch_results = []
        
        # Group by primary category to minimize pipeline calls
        for text_idx, (text, primary_categories) in enumerate(zip(texts, batch_primary_categories)):
            subcategory_results = {}
            
            for category in primary_categories:
                subcategories = self.application_subcategories.get(category, [])
                if not subcategories:
                    continue
                
                if self.use_zero_shot:
                    try:
                        # Classify into subcategories using transformer
                        results = self.classifier(text, subcategories, multi_label=True)
                        
                        # Filter subcategories with scores above threshold
                        filtered_results = []
                        for label, score in zip(results['labels'], results['scores']):
                            if score >= threshold:
                                filtered_results.append({"subcategory": label, "confidence": float(score)})
                        
                        if filtered_results:
                            subcategory_results[category] = filtered_results
                    except Exception:
                        # Fallback to keyword matching
                        fallback_results = self._keyword_based_classification(text, subcategories, threshold)
                        if fallback_results:
                            subcategory_results[category] = [
                                {"subcategory": item["category"], "confidence": item["confidence"]} 
                                for item in fallback_results
                            ]
                else:
                    # Use keyword matching directly
                    fallback_results = self._keyword_based_classification(text, subcategories, threshold)
                    if fallback_results:
                        subcategory_results[category] = [
                            {"subcategory": item["category"], "confidence": item["confidence"]} 
                            for item in fallback_results
                        ]
            
            batch_results.append(subcategory_results)
        
        return batch_results
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process document to classify applications at multiple levels"""
        # Use batch processing with a single item for consistency
        return self.process_batch([text])[0]
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts to classify applications at multiple levels
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of application classification dictionaries, one per input text
        """
        self.logger.info(f"Processing batch of {len(texts)} texts for application classification")
        
        # Skip empty batch
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        batch_results = [{} for _ in range(len(texts))]
        
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                # Add empty results for empty texts
                batch_results[i] = {"categories": [], "subcategories": {}}
        
        # Skip further processing if no valid texts
        if not valid_texts:
            return batch_results
        
        try:
            # First level: classify primary categories for all texts at once
            self.logger.info("Classifying primary categories in batch mode")
            primary_results_batch = self.classify_primary_category_batch(valid_texts)
            
            # Extract primary categories for subcategory classification
            primary_categories_batch = []
            for primary_results in primary_results_batch:
                categories = [item["category"] for item in primary_results]
                primary_categories_batch.append(categories)
            
            # Second level: classify subcategories for each document
            self.logger.info("Classifying subcategories in batch mode")
            subcategory_results_batch = self.classify_subcategories_batch(
                valid_texts, primary_categories_batch
            )
            
            # Combine results and place in correct positions
            for batch_idx, result_idx in enumerate(valid_indices):
                batch_results[result_idx] = {
                    "categories": primary_results_batch[batch_idx],
                    "subcategories": subcategory_results_batch[batch_idx]
                }
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch application classification: {str(e)}")
            
            # Fallback to individual processing
            for i, text_idx in enumerate(valid_indices):
                try:
                    # First level: classify primary categories
                    primary_results = self._keyword_based_classification(valid_texts[i], self.primary_categories)
                    primary_categories = [item["category"] for item in primary_results]
                    
                    # Second level: classify subcategories for each detected primary category
                    subcategory_results = {}
                    for category in primary_categories:
                        subcategories = self.application_subcategories.get(category, [])
                        if subcategories:
                            fallback_results = self._keyword_based_classification(valid_texts[i], subcategories)
                            if fallback_results:
                                subcategory_results[category] = [
                                    {"subcategory": item["category"], "confidence": item["confidence"]} 
                                    for item in fallback_results
                                ]
                    
                    batch_results[text_idx] = {
                        "categories": primary_results,
                        "subcategories": subcategory_results,
                        "error": str(e),
                        "fallback": True
                    }
                except Exception as inner_e:
                    batch_results[text_idx] = {
                        "categories": [],
                        "subcategories": {},
                        "error": f"{str(e)} -> {str(inner_e)}",
                        "fallback": True
                    }
            
            return batch_results