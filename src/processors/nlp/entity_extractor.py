# src/processors/nlp/entity_extractor.py
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import re
import json
import os

from .base_processor import BaseNLPProcessor

class GrapheneEntityExtractor(BaseNLPProcessor):
    """
    Extract graphene-specific entities using a combination of:
    1. Domain-specific NER model
    2. Pattern-based extraction with more sophisticated patterns
    3. SpaCy's built-in entity recognition as fallback
    """
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        super().__init__(model_name=model_name, device=device)
        self.logger = self._setup_logging()
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize transformer-based NER model
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        if self.device == "mps":
            # HF pipeline doesn't support MPS directly — fallback to manual use
            self.ner_pipeline = None
            self.logger.info("Skipping HF pipeline for MPS — using manual NER inference")
        else:
            self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer,
                                        aggregation_strategy="simple",
                                        device=0 if self.device == "cuda" else -1)
        
        # Initialize regex patterns for different entity types
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for entity extraction"""
        # Pattern for graphene variant materials
        self.material_patterns = [
            r'\b(?:graphene oxide|GO|reduced graphene oxide|rGO|graphene nano(?:sheets|platelets|ribbons|particles)|GNP|graphene quantum dots|GQD|exfoliated graphene|CVD graphene|few-layer graphene|single-layer graphene|multi-layer graphene|functionalized graphene)\b',
            r'\b(?:graphene)-(?:based|doped|functionalized|modified)\s+(?:\w+\s+){0,3}(?:materials?|composite|structure|film)\b'
        ]
        
        # Pattern for fabrication methods
        self.fabrication_patterns = [
            r'\b(?:chemical vapor deposition|CVD|mechanical exfoliation|liquid phase exfoliation|electrochemical exfoliation|thermal exfoliation|Hummers method|modified Hummers|sonication|thermal reduction|chemical reduction|hydrothermal synthesis|solvothermal synthesis|epitaxial growth|plasma enhanced CVD|PECVD|screen printing|chemical exfoliation)\b',
            r'(?:prepared|synthesized|fabricated|manufactured|produced)\s+(?:via|using|by)\s+(?:\w+\s+){0,3}(?:method|process|technique|approach)'
        ]
        
        # Pattern for performance metrics with units and values
        self.metrics_patterns = [
            # Electrical properties
            r'(?:electrical|thermal)?\s*conductivity\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:S/m|S/cm|W/mK)',
            r'(?:sheet|surface)\s*resistance\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:Ω/□|Ω/sq|ohm/sq)',
            r'mobility\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:cm²/Vs|cm2/Vs)',
            
            # Mechanical properties
            r'(?:tensile|mechanical|yield)\s*strength\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:GPa|MPa|kPa)',
            r'(?:Young\'s|elastic)\s*modulus\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:GPa|MPa|TPa)',
            r'(?:specific)?\s*surface\s*area\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:m²/g|m2/g)',
            
            # Energy storage properties
            r'(?:specific)?\s*capacitance\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:F/g|mF/cm²)',
            r'energy\s*density\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:Wh/kg|Wh/L)',
            r'power\s*density\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:W/kg|kW/kg)',
            
            # Efficiency measurements
            r'(?:power\s*conversion|energy|quantum)\s*efficiency\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'figure\s*of\s*merit\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)',
            
            # Sensor properties
            r'sensitivity\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:\S+)', # Capture various sensitivity units
            r'(?:detection\s*limit|LOD)\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:\S+)', # Capture various LOD units
            r'(?:response|recovery)\s*time\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*(?:s|ms|min)',
            
            # Optical properties
            r'transmittance\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'transparency\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'reflectivity\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            
            # Catalytic properties
            r'conversion\s*(?:rate|efficiency)\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'selectivity\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'turnover\s*(?:frequency|number)\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)'
        ]
    
    def _extract_with_patterns(self, text, patterns):
        """Extract entities using regex patterns"""
        results = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the entire matched phrase
                entity_text = match.group(0)
                # Extract numerical value if it's in a capture group
                value = match.groups()[0] if match.groups() else None
                
                # Get the context around the match (sentence containing the match)
                # First try using SpaCy for proper sentence boundaries
                context = ""
                doc = self.nlp(text)
                for sent in doc.sents:
                    if entity_text in sent.text:
                        context = sent.text
                        break
                
                # If no context found (e.g., SpaCy failed to segment properly), get +/- 100 chars
                if not context:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                
                results.append({
                    "text": entity_text,
                    "value": float(value) if value and value.replace('.', '', 1).isdigit() else None,
                    "span": (match.start(), match.end()),
                    "context": context
                })
        
        return results
    
    def extract_materials(self, text):
        """Extract graphene material variants and descriptions"""
        return self._extract_with_patterns(text, self.material_patterns)
    
    def extract_fabrication_methods(self, text):
        """Extract fabrication and synthesis methods"""
        return self._extract_with_patterns(text, self.fabrication_patterns)
    
    def extract_metrics(self, text):
        """Extract performance metrics with values and units"""
        return self._extract_with_patterns(text, self.metrics_patterns)
    
    def extract_entities_with_spacy(self, text):
        """Extract generic named entities using SpaCy as fallback"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "ORG", "GPE", "DATE", "PRODUCT", "PERSON"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "span": (ent.start_char, ent.end_char),
                    "context": ent.sent.text if ent.sent else text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
                })
        
        return entities
    
    def process(self, text):
        """Process text to extract all relevant entities"""
        if not text or len(text.strip()) == 0:
            return {
                "materials": [],
                "fabrication_methods": [],
                "metrics": [],
                "entities": []
            }
        
        # Extract different types of entities
        materials = self.extract_materials(text)
        fabrication_methods = self.extract_fabrication_methods(text)
        metrics = self.extract_metrics(text)
        generic_entities = self.extract_entities_with_spacy(text)
        
        return {
            "materials": materials,
            "fabrication_methods": fabrication_methods,
            "metrics": metrics,
            "entities": generic_entities
        }

