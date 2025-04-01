# src/processors/nlp/application_classifier.py
from typing import Dict, List, Any
import torch
from transformers import pipeline
import logging
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_processor import BaseNLPProcessor

class ApplicationClassifier(BaseNLPProcessor):
    """
    Specialized processor for classifying graphene applications using zero-shot or finetuned models
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None, taxonomy_path=None):
        super().__init__(model_name=model_name, device=device)

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
        if not self.use_zero_shot:
            return self._keyword_based_classification(text, self.primary_categories, threshold)
            
        try:
            results = self.classifier(text, self.primary_categories, multi_label=True)
            
            # Filter categories with scores above threshold
            categories = []
            for label, score in zip(results['labels'], results['scores']):
                if score >= threshold:
                    categories.append({"category": label, "confidence": float(score)})
            
            return categories
        except Exception as e:
            self.logger.warning(f"Zero-shot classification failed: {str(e)}")
            return self._keyword_based_classification(text, self.primary_categories, threshold)
    
    def classify_subcategories(self, text: str, primary_categories: List[str], 
                              threshold: float = 0.3) -> Dict[str, List[Dict[str, Any]]]:
        """Classify the document into subcategories for each primary category"""
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
        
        return subcategory_results
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process document to classify applications at multiple levels"""
        try:
            # Skip empty text
            if not text or len(text.strip()) == 0:
                return {"categories": [], "subcategories": {}}
            
            # First level: classify primary categories
            primary_results = self.classify_primary_category(text)
            primary_categories = [item["category"] for item in primary_results]
            
            # Second level: classify subcategories for each detected primary category
            subcategory_results = self.classify_subcategories(text, primary_categories)
            
            return {
                "categories": primary_results,
                "subcategories": subcategory_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in application classification: {str(e)}")
            # Return simple keyword-based results as a fallback
            fallback_results = self._keyword_based_classification(text, self.primary_categories)
            return {
                "categories": fallback_results,
                "subcategories": {},
                "error": str(e),
                "fallback": True
            }