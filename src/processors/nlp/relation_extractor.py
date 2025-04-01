# src/processors/nlp/relation_extractor.py
from typing import Dict, List, Any
import spacy
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .base_processor import BaseNLPProcessor

class RelationExtractor(BaseNLPProcessor):
    """
    Extract relationships between graphene properties and applications
    and between materials, fabrication methods, and performance metrics
    """
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        super().__init__(model_name=model_name, device=device)
        
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
    
    def find_property_application_relationships(self, text):
        """
        Find relationships between properties and applications
        using dependency parsing and sentence proximity
        """
        relationships = []
        
        # Parse the text
        doc = self.nlp(text)
        
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
            
            # If both properties and applications are found, determine the relationship
            if properties_in_sent and applications_in_sent:
                for prop in properties_in_sent:
                    for app in applications_in_sent:
                        # Try to classify relationship type
                        relation_type = self.classify_relationship(sent_text, prop["property"], app)
                        
                        relationships.append({
                            "property_category": prop["category"],
                            "property": prop["property"],
                            "application": app,
                            "relation_type": relation_type,
                            "context": sent.text
                        })
        
        return relationships
    
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
    
    def classify_relationship(self, context, entity1, entity2):
        """
        Classify the relationship between two entities
        using the transformer-based zero-shot classifier
        """
        if not self.relation_classifier:
            return "is related to"  # Default fallback
            
        try:
            # Create a hypothesis for classification
            hypothesis = f"The relationship between {entity1} and {entity2} is that {entity1} RELATION {entity2}."
            
            # Classify the relationship
            result = self.relation_classifier(
                context,
                self.relation_classes,
                hypothesis_template="This text describes that {}."
            )
            
            # Return the top predicted relation type
            return result["labels"][0]
        except Exception as e:
            self.logger.error(f"Error classifying relationship: {str(e)}")
            return "is related to"  # Default fallback
    
    def process(self, text, extracted_entities=None):
        """
        Process text to extract relationships between entities
        Can accept previously extracted entities to avoid redundant extraction
        """
        if not text or len(text.strip()) == 0:
            return {
                "property_application_relations": [],
                "material_fabrication_relations": [],
                "material_performance_relations": []
            }
        
        # Find property-application relationships directly from text
        property_application_relations = self.find_property_application_relationships(text)
        
        # If entities were provided, find relationships between them
        material_fabrication_relations = []
        material_performance_relations = []
        
        if extracted_entities:
            materials = extracted_entities.get("materials", [])
            fabrication_methods = extracted_entities.get("fabrication_methods", [])
            metrics = extracted_entities.get("metrics", [])
            
            material_fabrication_relations = self.find_material_fabrication_relationships(
                materials, fabrication_methods
            )
            
            material_performance_relations = self.find_material_performance_relationships(
                materials, metrics
            )
        
        return {
            "property_application_relations": property_application_relations,
            "material_fabrication_relations": material_fabrication_relations,
            "material_performance_relations": material_performance_relations
        }
