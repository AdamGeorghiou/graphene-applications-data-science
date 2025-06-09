from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import re
import json
import os
import logging
from tqdm import tqdm

from .base_processor import BaseNLPProcessor

class GrapheneEntityExtractor(BaseNLPProcessor):
    """
    Extract graphene-specific entities using a combination of:
    1. Domain-specific NER model
    2. Pattern-based extraction with more sophisticated patterns
    3. SpaCy's built-in entity recognition as fallback
    
    Now with true GPU-accelerated batch processing support
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

    def _setup_logging(self):
        """Set up logging for the entity extractor"""
        logger = logging.getLogger("GrapheneEntityExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
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
    
    def _extract_with_patterns_optimized(self, text, patterns, spacy_doc=None):
        """
        Extract entities using regex patterns with optional SpaCy doc
        
        Args:
            text: Text to extract from
            patterns: List of regex patterns to use
            spacy_doc: Optional pre-processed SpaCy doc
            
        Returns:
            List of entity dictionaries
        """
        results = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the entire matched phrase
                entity_text = match.group(0)
                # Extract numerical value if it's in a capture group
                value = match.groups()[0] if match.groups() else None
                
                # Get the context around the match (sentence containing the match)
                context = ""
                
                # Use provided SpaCy doc if available
                if spacy_doc:
                    for sent in spacy_doc.sents:
                        # Check if the entity text is within the sentence
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
    
    def extract_materials(self, text, spacy_doc=None):
        """Extract graphene material variants and descriptions"""
        return self._extract_with_patterns_optimized(text, self.material_patterns, spacy_doc)
    
    def extract_fabrication_methods(self, text, spacy_doc=None):
        """Extract fabrication and synthesis methods"""
        return self._extract_with_patterns_optimized(text, self.fabrication_patterns, spacy_doc)
    
    def extract_metrics(self, text, spacy_doc=None):
        """Extract performance metrics with values and units"""
        return self._extract_with_patterns_optimized(text, self.metrics_patterns, spacy_doc)
    
    def extract_entities_with_spacy(self, spacy_doc, text):
        """
        Extract generic named entities using SpaCy
        
        Args:
            spacy_doc: Pre-processed SpaCy document
            text: Original text (for context extraction)
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        for ent in spacy_doc.ents:
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
        # Use batch processing with a single item for consistency
        return self.process_batch([text])[0]
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts to extract entities with true batch processing
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of entity dictionaries, one per input text
        """
        self.logger.info(f"Processing batch of {len(texts)} texts for entity extraction")
        
        # Skip empty batch
        if not texts:
            return []
        
        # Initialize results list
        batch_results = [{} for _ in range(len(texts))]
        
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                # Add empty results for empty texts
                batch_results[i] = {
                    "materials": [],
                    "fabrication_methods": [],
                    "metrics": [],
                    "entities": []
                }
        
        # Skip further processing if no valid texts
        if not valid_texts:
            return batch_results
        
        # Batch process with SpaCy
        self.logger.info("Batch processing with SpaCy")
        spacy_docs = list(self.nlp.pipe(valid_texts))
        
        # Batch process with HF NER pipeline if available
        transformer_entities = None
        if self.ner_pipeline and valid_texts:
            try:
                self.logger.info("Batch processing with Transformer NER")
                # Process entire batch at once through the pipeline
                # Use a reasonable batch size to avoid OOM errors
                ner_batch_size = 16  # Adjust based on GPU memory
                transformer_results = []
                
                # Process in sub-batches if needed
                for i in range(0, len(valid_texts), ner_batch_size):
                    sub_batch = valid_texts[i:i + ner_batch_size]
                    sub_results = self.ner_pipeline(sub_batch)
                    # Ensure results are always a list of lists
                    if isinstance(sub_results, list) and not isinstance(sub_results[0], list):
                        if len(sub_batch) == 1:
                            transformer_results.append(sub_results)
                        else:
                            # This shouldn't normally happen but handle just in case
                            for j in range(len(sub_batch)):
                                transformer_results.append([])
                    else:
                        transformer_results.extend(sub_results)
                
                transformer_entities = transformer_results
            except Exception as e:
                self.logger.error(f"Error in transformer NER batch processing: {str(e)}")
                transformer_entities = None
        
        # Now process each text with its SpaCy doc and transformer results
        for batch_idx, doc_idx in enumerate(valid_indices):
            spacy_doc = spacy_docs[batch_idx]
            text = valid_texts[batch_idx]
            
            # Extract different types of entities using the SpaCy doc
            materials = self.extract_materials(text, spacy_doc)
            fabrication_methods = self.extract_fabrication_methods(text, spacy_doc)
            metrics = self.extract_metrics(text, spacy_doc)
            generic_entities = self.extract_entities_with_spacy(spacy_doc, text)
            
            # Create result dictionary
            result = {
                "materials": materials,
                "fabrication_methods": fabrication_methods,
                "metrics": metrics,
                "entities": generic_entities
            }
            
            # Add transformer entities if available
            if transformer_entities and batch_idx < len(transformer_entities):
                result["transformer_entities"] = transformer_entities[batch_idx]
            
            # Store in the appropriate position in batch_results
            batch_results[doc_idx] = result
        
        return batch_results