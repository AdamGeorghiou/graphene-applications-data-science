# src/processors/nlp/trl_assessor.py
from typing import Dict, List, Any
import re
import torch
from transformers import pipeline

from .base_processor import BaseNLPProcessor

class TRLAssessor(BaseNLPProcessor):
    """
    Assess Technology Readiness Level (TRL) of graphene applications
    based on context and specific indicators
    """
    
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device=None):
        super().__init__(model_name=model_name, device=device)
        
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
        # Create labels for each TRL level
        trl_labels = [f"TRL {level}: {info['name']}" for level, info in self.trl_definitions.items()]
        
        try:
            # Run zero-shot classification
            result = self.classifier(text, trl_labels, multi_label=True)
            
            # Extract predicted TRL levels with confidence scores
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
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in zero-shot TRL assessment: {str(e)}")
            return []
    
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
    
    def process(self, text, source_type=None):
        """
        Process document to assess TRL level
        """
        if not text or len(text.strip()) == 0:
            return {
                "trl_level": 0,
                "confidence": 0,
                "evidence": [],
                "explanation": "Empty text"
            }
        
        # Extract TRL indicators from text
        trl_evidence = self.extract_trl_indicators(text)
        
        # Assess TRL using zero-shot classification
        zero_shot_predictions = self.assess_trl_with_zero_shot(text)
        
        # Combine evidence to determine final TRL level
        assessment = self.refine_trl_assessment(trl_evidence, zero_shot_predictions, source_type)
        
        return assessment

