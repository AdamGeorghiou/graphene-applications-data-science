from typing import Dict, List, Any, Tuple
import spacy
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging
from tqdm import tqdm

from .base_processor import BaseNLPProcessor

class RelationExtractor(BaseNLPProcessor):
    """
    Extract relationships between graphene properties and applications
    and between materials, fabrication methods, and performance metrics
    Now with optimized GPU-accelerated batch processing support
    """
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        super().__init__(model_name=model_name, device=device)
        self.logger = self._setup_logging()
        
        # Load SpaCy model with dependency parsing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define property categories
        self.property_categories = {
            "electrical": ["conductivity", "resistance", "mobility", "current", "voltage", "ohmic"],
            "thermal": ["thermal conductivity", "heat", "temperature", "thermal stability"],
            "mechanical": ["strength", "stiffness", "modulus", "elasticity", "flexibility", "strain", "stress"],
            "optical": ["transmittance", "transparency", "reflectivity", "absorbance", "photoluminescence"],
            "surface": ["surface area", "roughness", "porosity", "wettability", "hydrophobicity", "hydrophilicity"],
            "chemical": ["reactivity", "stability", "functional groups", "doping"]
        }
        
        # Application categories (derived from ApplicationClassifier)
        self.application_categories = [
            "energy storage", "battery", "supercapacitor", "fuel cell", "solar cell", "photovoltaic",
            "transistor", "semiconductor", "sensor", "biosensor", "display", "electronics",
            "composite", "coating", "membrane", "filter", "barrier",
            "catalyst", "photocatalyst", "electrocatalyst",
            "drug delivery", "biomedical", "tissue engineering",
            "water treatment", "water purification", "environmental remediation"
        ]
        
        # Initialize zero-shot relation classifier
        self.setup_relation_classifier()
    
    def _setup_logging(self):
        """Set up logging for the relation extractor"""
        logger = logging.getLogger("RelationExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def setup_relation_classifier(self):
        """Set up the relation classification model"""
        try:
            # Load a model fine-tuned for relationship classification or use zero-shot
            self.relation_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            self.relation_classes = [
                "improves", "enhances", "increases", "decreases", "enables",
                "causes", "prevents", "is used for", "depends on", "is related to"
            ]
        except Exception as e:
            self.logger.error(f"Error setting up relation classifier: {str(e)}")
            self.relation_classifier = None
    
    def find_property_application_relationships_candidates(self, text, spacy_doc=None):
        """
        Find potential property-application relationship candidates
        without classifying them yet (to enable batched classification)
        
        Args:
            text: Text to process
            spacy_doc: Optional pre-processed SpaCy doc
            
        Returns:
            List of (context, property_info, application) tuples for batch classification
        """
        candidates = []
        
        # Parse the text if doc not provided
        doc = spacy_doc if spacy_doc else self.nlp(text)
        
        # Process each sentence
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Skip sentences that are too short
            if len(sent_text.split()) < 5:
                continue
            
            # Find properties in the sentence
            properties_in_sent = []
            for category, terms in self.property_categories.items():
                for term in terms:
                    if term.lower() in sent_text:
                        properties_in_sent.append({"category": category, "property": term})
            
            # Find applications in the sentence
            applications_in_sent = []
            for app in self.application_categories:
                if app.lower() in sent_text:
                    applications_in_sent.append(app)
            
            # If both properties and applications are found, add as candidates
            if properties_in_sent and applications_in_sent:
                for prop in properties_in_sent:
                    for app in applications_in_sent:
                        candidates.append((
                            sent_text,  # context
                            prop,       # property info
                            app         # application
                        ))
        
        return candidates
    
    def find_material_fabrication_relationships(self, materials, fabrication_methods):
        """Find relationships between materials and fabrication methods"""
        relationships = []
        
        for material in materials:
            material_text = material["text"]
            material_context = material["context"].lower()
            
            for method in fabrication_methods:
                method_text = method["text"]
                
                # If the method appears in the material's context
                if method_text.lower() in material_context:
                    # Classify the relationship
                    relation_type = "is fabricated by"  # Default
                    
                    # Improve specificity with keywords
                    if re.search(r"(?:prepared|synthesized|created|made|produced)\s+(?:via|by|using|through)", material_context):
                        relation_type = "is synthesized by"
                    elif re.search(r"(?:modified|functionalized|treated)\s+(?:via|by|using|through)", material_context):
                        relation_type = "is modified by"
                    
                    relationships.append({
                        "material": material_text,
                        "fabrication_method": method_text,
                        "relation_type": relation_type,
                        "context": material["context"]
                    })
        
        return relationships
    
    def find_material_performance_relationships(self, materials, metrics):
        """Find relationships between materials and performance metrics"""
        relationships = []
        
        for material in materials:
            material_text = material["text"]
            material_context = material["context"].lower()
            
            for metric in metrics:
                metric_text = metric["text"]
                metric_value = metric.get("value")
                
                # If the metric appears in the material's context
                if metric_text.lower() in material_context:
                    relationships.append({
                        "material": material_text,
                        "metric": metric_text,
                        "value": metric_value,
                        "context": material["context"]
                    })
        
        return relationships
    
    def batch_classify_relationships(self, candidates_by_doc):
        """
        Efficiently batch-classify all relationship candidates
        
        Args:
            candidates_by_doc: List of lists of (context, property_info, application) tuples,
                               one list per document
        
        Returns:
            List of lists of classified relationship dictionaries
        """
        if not self.relation_classifier or not candidates_by_doc:
            # Return empty results for each document
            return [[] for _ in range(len(candidates_by_doc))]
        
        # Flatten all candidates for efficient batch processing
        all_contexts = []
        all_entity_pairs = []
        # Keep track of which document and position each candidate belongs to
        candidate_doc_indices = []  
        
        # Collect all candidates
        for doc_idx, doc_candidates in enumerate(candidates_by_doc):
            for context, prop_info, app in doc_candidates:
                all_contexts.append(context)
                all_entity_pairs.append((prop_info["property"], app))
                candidate_doc_indices.append(doc_idx)
        
        if not all_contexts:
            return [[] for _ in range(len(candidates_by_doc))]
        
        # Create classified relationships for each document
        doc_relationships = [[] for _ in range(len(candidates_by_doc))]
        
        try:
            # Create hypotheses for zero-shot classification
            hypotheses = []
            for prop, app in all_entity_pairs:
                hypotheses.append(f"The relationship between {prop} and {app} is that {prop} RELATION {app}.")
            
            # Batch size for classification (adjust based on available memory)
            batch_size = 16
            
            # Process in batches to avoid OOM errors
            for i in range(0, len(all_contexts), batch_size):
                # Get batch contexts and hypotheses
                batch_contexts = all_contexts[i:i + batch_size]
                batch_hypotheses = hypotheses[i:i + batch_size]
                batch_indices = range(i, min(i + batch_size, len(all_contexts)))
                
                # Process this batch - group by unique contexts to minimize pipeline calls
                context_to_indices = {}
                for j, ctx in enumerate(batch_contexts):
                    if ctx not in context_to_indices:
                        context_to_indices[ctx] = []
                    context_to_indices[ctx].append(j)
                
                # For each unique context, run zero-shot classification in a batch
                for context, indices in context_to_indices.items():
                    # Get hypotheses for this context
                    context_hypotheses = [batch_hypotheses[j] for j in indices]
                    
                    # Run classification in a single batch for this context
                    relation_results = self.relation_classifier(
                        context,
                        self.relation_classes,
                        multi_label=False,
                        hypothesis_template="This text describes that {}."
                    )
                    
                    # If only one result, wrap it
                    if not isinstance(relation_results, list):
                        relation_results = [relation_results]
                    
                    # Process results for each hypothesis in this context
                    for j, result in enumerate(relation_results):
                        if j < len(indices):
                            idx = indices[j]
                            global_idx = batch_indices[idx]
                            doc_idx = candidate_doc_indices[global_idx]
                            
                            # Get original property and application
                            prop_info = candidates_by_doc[doc_idx][global_idx - sum(len(c) for c in candidates_by_doc[:doc_idx])][1]
                            app = candidates_by_doc[doc_idx][global_idx - sum(len(c) for c in candidates_by_doc[:doc_idx])][2]
                            
                            # Create relationship dictionary
                            relationship = {
                                "property_category": prop_info["category"],
                                "property": prop_info["property"],
                                "application": app,
                                "relation_type": result["labels"][0],
                                "context": context
                            }
                            
                            # Add to the appropriate document's relationships
                            doc_relationships[doc_idx].append(relationship)
            
            return doc_relationships
            
        except Exception as e:
            self.logger.error(f"Error in batch relationship classification: {str(e)}")
            
            # Fallback: create relationships with default relation type
            fallback_relationships = [[] for _ in range(len(candidates_by_doc))]
            
            for doc_idx, doc_candidates in enumerate(candidates_by_doc):
                for context, prop_info, app in doc_candidates:
                    fallback_relationships[doc_idx].append({
                        "property_category": prop_info["category"],
                        "property": prop_info["property"],
                        "application": app,
                        "relation_type": "is related to",  # Default fallback
                        "context": context
                    })
            
            return fallback_relationships
    
    def process(self, text, extracted_entities=None):
        """
        Process text to extract relationships between entities
        Can accept previously extracted entities to avoid redundant extraction
        """
        # Use batch processing with a single item for consistency
        return self.process_batch([text], [extracted_entities] if extracted_entities else None)[0]
    
    def process_batch(self, texts: List[str], batch_entities: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of texts to extract relationships between entities
        with optimized batch processing for classification
        
        Args:
            texts: List of text strings to process
            batch_entities: Optional list of entity dictionaries for each text
            
        Returns:
            List of relation dictionaries, one per input text
        """
        self.logger.info(f"Processing batch of {len(texts)} texts for relation extraction")
        
        # Skip empty batch
        if not texts:
            return []
        
        # Initialize results list
        batch_results = []
        
        # Batch process with SpaCy for property-application relations
        self.logger.info("Batch processing texts with SpaCy")
        spacy_docs = list(self.nlp.pipe(texts))
        
        # Step 1: Collect candidate property-application relationships without classifying yet
        self.logger.info("Collecting property-application relationship candidates")
        property_app_candidates = []
        
        for text, doc in zip(texts, spacy_docs):
            if not text or len(text.strip()) == 0:
                property_app_candidates.append([])
            else:
                candidates = self.find_property_application_relationships_candidates(text, doc)
                property_app_candidates.append(candidates)
        
        # Step 2: Batch classify all property-application relationships
        self.logger.info("Batch classifying property-application relationships")
        property_app_relations = self.batch_classify_relationships(property_app_candidates)
        
        # Step 3: Process material and fabrication relationships for each document individually
        self.logger.info("Processing material relationships")
        for idx, (text, doc) in enumerate(zip(texts, spacy_docs)):
            # Skip empty text
            if not text or len(text.strip()) == 0:
                batch_results.append({
                    "property_application_relations": [],
                    "material_fabrication_relations": [],
                    "material_performance_relations": []
                })
                continue
            
            # Get entities for this text if provided
            entities = None
            if batch_entities and idx < len(batch_entities):
                entities = batch_entities[idx]
            
            # Initialize other relation types
            material_fabrication_relations = []
            material_performance_relations = []
            
            # If entities were provided, find relationships between them
            if entities:
                materials = entities.get("materials", [])
                fabrication_methods = entities.get("fabrication_methods", [])
                metrics = entities.get("metrics", [])
                
                # Find material-fabrication relationships
                material_fabrication_relations = self.find_material_fabrication_relationships(
                    materials, fabrication_methods
                )
                
                # Find material-performance relationships
                material_performance_relations = self.find_material_performance_relationships(
                    materials, metrics
                )
            
            # Combine results for this text
            text_results = {
                "property_application_relations": property_app_relations[idx],
                "material_fabrication_relations": material_fabrication_relations,
                "material_performance_relations": material_performance_relations
            }
            
            batch_results.append(text_results)
        
        return batch_results