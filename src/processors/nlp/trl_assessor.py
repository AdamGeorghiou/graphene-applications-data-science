from typing import Dict, List, Any
import re
import torch
from transformers import pipeline
import logging
from tqdm import tqdm

from .base_processor import BaseNLPProcessor

class TRLAssessor(BaseNLPProcessor):
    """
    Assess Technology Readiness Level (TRL) of graphene applications
    based on context and specific indicators
    Now with GPU-accelerated batch processing support
    """
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        super().__init__(model_name=model_name, device=device)
        self.logger = self._setup_logging()
        
        # Set up zero-shot classification for TRL assessment
        if self.device == "mps":
            self.classifier = None
            self.logger.info("Skipping HF zero-shot pipeline for MPS — unsupported, using fallback.")
        else:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
        
        # Define TRL levels with descriptions and indicators
        self.trl_definitions = {
            1: {
                "name": "Technology prototype demonstrated in operational environment",
                "description": "Prototype near or at planned operational system. Represents a major step up from TRL 6 by requiring demonstration of an actual system prototype in an operational environment.",
                "indicators": [
                    "operational environment", "system prototype", "demonstration", 
                    "operational testing", "actual prototype", "real environment",
                    "field tested", "operational demonstration", "prototype near planned system"
                ]
            },
            8: {
                "name": "System complete and qualified",
                "description": "Technology has been proven to work in its final form and under expected conditions. In most cases, this TRL represents the end of true system development.",
                "indicators": [
                    "system complete", "qualified", "final form", "expected conditions", 
                    "certification", "qualification testing", "ready for commercialization",
                    "production ready", "commercial ready", "verification", "validation"
                ]
            },
            9: {
                "name": "Actual system proven in operational environment",
                "description": "Actual application of the technology in its final form and under mission conditions, such as those encountered in operational test and evaluation.",
                "indicators": [
                    "actual system", "operational environment", "mission conditions", 
                    "operational testing", "commercial application", "market", "industry",
                    "commercialized", "production", "deployed", "industrial use", "real-world"
                ]
            }
        }
    
    def _setup_logging(self):
        """Set up logging for the TRL assessor"""
        logger = logging.getLogger("TRLAssessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def extract_trl_indicators(self, text):
        """
        Extract phrases that indicate specific TRL levels
        """
        trl_evidence = []
        
        for trl_level, trl_info in self.trl_definitions.items():
            for indicator in trl_info["indicators"]:
                # Look for indicator phrases in the text
                pattern = r'(?i)\b' + re.escape(indicator) + r'\b'
                for match in re.finditer(pattern, text):
                    # Get the sentence containing the match for context
                    sentence = self._extract_sentence(text, match.start(), match.end())
                    
                    trl_evidence.append({
                        "trl_level": trl_level,
                        "trl_name": trl_info["name"],
                        "indicator": indicator,
                        "context": sentence,
                        "span": (match.start(), match.end())
                    })
        
        return trl_evidence
    
    def extract_trl_indicators_batch(self, texts):
        """
        Extract phrases that indicate specific TRL levels from multiple texts
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of lists of TRL evidence dictionaries, one list per text
        """
        batch_evidence = []
        
        for text in texts:
            # Process each text individually
            # (regex processing is inherently sequential)
            evidence = self.extract_trl_indicators(text)
            batch_evidence.append(evidence)
        
        return batch_evidence
    
    def _extract_sentence(self, text, start, end):
        """Extract the sentence containing a matched span"""
        # Find the beginning of the sentence (previous period + space or start of text)
        sentence_start = text.rfind('. ', 0, start) + 2
        if sentence_start == 1:  # No period found or period is the first character
            sentence_start = 0
            
        # Find the end of the sentence (next period + space or end of text)
        sentence_end = text.find('. ', end)
        if sentence_end == -1:  # No period found
            sentence_end = len(text)
        else:
            sentence_end += 1  # Include the period
            
        return text[sentence_start:sentence_end].strip()
    
    def assess_trl_with_zero_shot(self, text):
        """
        Assess TRL level using zero-shot classification on key passages
        """
        # Use batch processing with a single item for consistency
        return self.assess_trl_with_zero_shot_batch([text])[0]
    
    def assess_trl_with_zero_shot_batch(self, texts):
        """
        Assess TRL level using zero-shot classification for multiple texts
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of lists of TRL prediction dictionaries, one list per text
        """
        # Create labels for each TRL level
        trl_labels = [f"TRL {level}: {info['name']}" for level, info in self.trl_definitions.items()]
        
        # Initialize empty results for each text
        batch_predictions = [[] for _ in range(len(texts))]
        
        if self.classifier:
            try:
                # Determine appropriate batch size based on text length
                avg_length = sum(len(text.split()) for text in texts) / len(texts)
                batch_size = min(16, max(1, int(5000 / max(1, avg_length))))
                
                # Process in smaller sub-batches to avoid OOM errors
                for i in range(0, len(texts), batch_size):
                    sub_batch = texts[i:i + batch_size]
                    
                    # Run zero-shot classification on the sub-batch
                    results = self.classifier(sub_batch, trl_labels, multi_label=True)
                    
                    # Ensure results is a list of dictionaries (handle single item case)
                    if not isinstance(results, list):
                        results = [results]
                    
                    # Process each result
                    for j, result in enumerate(results):
                        predictions = []
                        for label, score in zip(result['labels'], result['scores']):
                            trl_level = int(label.split(':')[0].replace('TRL', '').strip())
                            predictions.append({
                                "trl_level": trl_level,
                                "trl_name": self.trl_definitions[trl_level]["name"],
                                "confidence": float(score),
                                "method": "zero-shot"
                            })
                        
                        # Sort by confidence score
                        predictions.sort(key=lambda x: x["confidence"], reverse=True)
                        batch_predictions[i + j] = predictions
                
                return batch_predictions
                
            except Exception as e:
                self.logger.error(f"Error in batch zero-shot TRL assessment: {str(e)}")
                
        # Fallback: empty predictions
        return batch_predictions
    
    def refine_trl_assessment(self, evidence, predictions, source_type=None):
        """
        Combine evidence from different methods to estimate the final TRL level
        with confidence scores and supporting evidence
        """
        if not evidence and not predictions:
            return {
                "trl_level": 0,
                "confidence": 0,
                "evidence": [],
                "explanation": "Insufficient information to determine TRL level"
            }
        
        # Count occurrences of each TRL level in the evidence
        trl_counts = {}
        for item in evidence:
            level = item["trl_level"]
            if level not in trl_counts:
                trl_counts[level] = 0
            trl_counts[level] += 1
        
        # Adjust for source type (e.g., patents usually indicate higher TRL)
        if source_type:
            source_type = source_type.lower()
            if 'patent' in source_type:
                # Patents typically indicate at least TRL 3-4
                for level in range(1, 3):
                    if level in trl_counts:
                        trl_counts[level] *= 0.5  # Reduce weight of lower TRLs
                for level in range(3, 10):
                    if level in trl_counts:
                        trl_counts[level] *= 1.5  # Increase weight of higher TRLs
            
            elif 'journal' in source_type or 'paper' in source_type:
                # Academic papers can range widely but often focus on earlier TRLs
                pass  # Use default weights
        
        # Combine with zero-shot predictions if available
        if predictions:
            for pred in predictions[:3]:  # Consider top 3 predictions
                level = pred["trl_level"]
                if level not in trl_counts:
                    trl_counts[level] = 0
                # Add weighted by confidence
                trl_counts[level] += pred["confidence"] * 3  # Multiply by 3 to give appropriate weight
        
        # Select the TRL level with the highest score
        max_count = 0
        estimated_trl = 0
        for level, count in trl_counts.items():
            if count > max_count:
                max_count = count
                estimated_trl = level
        
        # Calculate confidence based on evidence strength
        confidence = min(0.95, max_count / (sum(trl_counts.values()) + 0.001))
        
        # Prepare explanation
        if estimated_trl > 0:
            explanation = f"TRL {estimated_trl} ({self.trl_definitions[estimated_trl]['name']}) "
            explanation += f"with {confidence:.2f} confidence based on {len(evidence)} indicators"
        else:
            explanation = "Unable to determine TRL level with sufficient confidence"
        
        return {
            "trl_level": estimated_trl,
            "trl_name": self.trl_definitions[estimated_trl]["name"] if estimated_trl > 0 else "",
            "confidence": confidence,
            "evidence": evidence,
            "predictions": predictions[:3] if predictions else [],
            "explanation": explanation
        }
    
    def refine_trl_assessment_batch(self, batch_evidence, batch_predictions, batch_source_types=None):
        """
        Batch process TRL assessment refinement for multiple documents
        
        Args:
            batch_evidence: List of lists of evidence dictionaries
            batch_predictions: List of lists of prediction dictionaries
            batch_source_types: Optional list of source types
            
        Returns:
            List of assessment dictionaries, one per document
        """
        batch_assessments = []
        
        for i in range(len(batch_evidence)):
            # Get evidence and predictions for this document
            evidence = batch_evidence[i]
            predictions = batch_predictions[i] if i < len(batch_predictions) else []
            
            # Get source type if available
            source_type = None
            if batch_source_types and i < len(batch_source_types):
                source_type = batch_source_types[i]
            
            # Refine assessment for this document
            assessment = self.refine_trl_assessment(evidence, predictions, source_type)
            batch_assessments.append(assessment)
        
        return batch_assessments
    
    def process(self, text, source_type=None):
        """
        Process document to assess TRL level
        """
        # Use batch processing with a single item for consistency
        return self.process_batch([text], [source_type] if source_type else None)[0]
    
    def process_batch(self, texts: List[str], batch_source_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of documents to assess TRL levels
        
        Args:
            texts: List of text strings to process
            batch_source_types: Optional list of source types for each text
            
        Returns:
            List of TRL assessment dictionaries, one per text
        """
        self.logger.info(f"Processing batch of {len(texts)} texts for TRL assessment")
        
        # Skip empty batch
        if not texts:
            return []
        
        # Initialize results for empty texts
        batch_results = []
        valid_texts = []
        valid_indices = []
        valid_source_types = []
        
        # Filter out empty texts
        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                batch_results.append({
                    "trl_level": 0,
                    "confidence": 0,
                    "evidence": [],
                    "explanation": "Empty text"
                })
            else:
                valid_texts.append(text)
                valid_indices.append(i)
                
                # Get source type if available
                if batch_source_types and i < len(batch_source_types):
                    valid_source_types.append(batch_source_types[i])
                else:
                    valid_source_types.append(None)
        
        # Skip further processing if no valid texts
        if not valid_texts:
            return batch_results
        
        # Ensure batch_results has placeholders for all texts
        while len(batch_results) < len(texts):
            batch_results.append(None)
        
        # Extract TRL indicators from valid texts
        self.logger.info("Extracting TRL indicators")
        batch_evidence = self.extract_trl_indicators_batch(valid_texts)
        
        # Assess TRL using zero-shot classification for valid texts
        self.logger.info("Performing zero-shot TRL assessment")
        batch_predictions = self.assess_trl_with_zero_shot_batch(valid_texts)
        
        # Combine evidence to determine final TRL levels
        self.logger.info("Refining TRL assessments")
        batch_assessments = self.refine_trl_assessment_batch(
            batch_evidence, batch_predictions, valid_source_types
        )
        
        # Place results in the correct positions
        for i, assessment in enumerate(batch_assessments):
            batch_results[valid_indices[i]] = assessment
        
        return batch_results