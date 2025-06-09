# src/processors/nlp/trl_assessor.py

from typing import Dict, List, Any
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

from .base_processor import BaseNLPProcessor

class TRLAssessor(BaseNLPProcessor):
    """
    Assesses Technology Readiness Level (TRL) using a hybrid approach.

    This processor combines three signals for a robust 3-category TRL assessment:
    1. Supervised Classifier: A fine-tuned SciBERT model trained on labeled data (primary signal).
    2. Zero-Shot Classifier: A general-purpose model classifying text against 9 TRL descriptions.
    3. Rule-Based Indicators: Regex matching for specific TRL-related keywords.

    The final assessment is a weighted average of these three signals.
    """

    def __init__(self, model_name="facebook/bart-large-mnli", device=None, config=None,
                supervised_model_path="models/trl_scibert_3cat", use_supervised=True):
        
        super().__init__(model_name, device, config)
        self.logger = logging.getLogger(__name__) # Use standard Python logging

        # --- 1. Setup Zero-Shot Classifier ---
        if self.device == "mps":
            self.zeroshot_classifier = None
            self.logger.warning("Zero-shot pipeline is unsupported on MPS. Skipping initialization.")
        else:
            try:
                self.zeroshot_classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name, # Use the model from the base class init
                    device=0 if str(self.device).startswith("cuda") else -1
                )
                self.logger.info(f"Zero-shot pipeline loaded on device: {self.zeroshot_classifier.device}")
            except Exception as e:
                self.logger.error(f"Failed to load zero-shot pipeline: {e}")
                self.zeroshot_classifier = None
        
        # --- 2. Setup Rule-Based Indicators (mapped to 3 categories) ---
        self._setup_indicator_rules()

        # --- 3. Setup Supervised Classifier ---
        self.supervised_model = None
        self.supervised_tokenizer = None
        self.use_supervised = use_supervised
        if self.use_supervised:
            try:
                self.logger.info(f"Loading supervised TRL classifier from {supervised_model_path}")
                self.supervised_tokenizer = AutoTokenizer.from_pretrained(supervised_model_path)
                self.supervised_model = AutoModelForSequenceClassification.from_pretrained(supervised_model_path)
                self.supervised_model.to(self.device)
                self.supervised_model.eval()
                self.logger.info("Supervised TRL classifier loaded successfully.")
            except OSError:
                self.logger.warning(f"Could not find supervised TRL model at {supervised_model_path}. "
                                "TRL assessment will rely on zero-shot and rule-based methods only. "
                                "Please run the training script to enable the supervised model.")
                self.supervised_model = None

    def _setup_indicator_rules(self):
        """Define TRL indicator rules, mapping keywords to the 3 TRL categories."""
        self.indicator_rules = {
            # Category 0: Early (TRL 1-3)
            0: [
                "basic principles", "concept formulated", "proof of concept", "lab testing",
                "experimental", "fundamental research", "theoretical", "simulation", "feasibility"
            ],
            # Category 1: Mid (TRL 4-6)
            1: [
                "lab validation", "component validation", "lab environment", "relevant environment",
                "prototype", "system validation", "technology demonstration", "lab-scale"
            ],
            # Category 2: Late (TRL 7-9)
            2: [
                "operational environment", "system prototype", "field tested", "qualified",
                "system complete", "production ready", "commercial ready", "commercialization",
                "actual system", "mission conditions", "commercial application", "market",
                "industry", "production", "deployed", "industrial use", "real-world"
            ]
        }
        # Pre-compile regex patterns for efficiency
        self.compiled_rules = {
            cat: re.compile(r'(?i)\b(' + '|'.join(re.escape(kw) for kw in kws) + r')\b')
            for cat, kws in self.indicator_rules.items()
        }
        # Labels for the 9-level zero-shot model
        self.trl9_labels = [f"TRL {i} Description" for i in range(1, 10)]

    def _extract_indicators_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extracts TRL indicators using pre-compiled regex and maps them to 3 categories."""
        results = []
        for text in texts:
            votes = [0, 0, 0] # Votes for Early, Mid, Late
            found_indicators = []
            for category_idx, pattern in self.compiled_rules.items():
                matches = pattern.findall(text)
                if matches:
                    votes[category_idx] += len(matches)
                    found_indicators.extend(matches)
            
            total_votes = sum(votes)
            rule_probs = [v / total_votes if total_votes > 0 else 1/3 for v in votes]
            results.append({"rule_probs": rule_probs, "found_indicators": list(set(found_indicators))})
        return results

    def _classify_zeroshot_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Performs 9-level TRL zero-shot classification and returns raw probabilities."""
        if not self.zeroshot_classifier:
            return [{"zeroshot_probs_9": [1/9]*9}] * len(texts)
            
        results = []
        try:
            # The pipeline handles batching internally and is faster with multi_label=False for this task
            outputs = self.zeroshot_classifier(texts, self.trl9_labels, multi_label=False)
            # Ensure output is always a list
            if not isinstance(outputs, list):
                outputs = [outputs]

            for out in outputs:
                score_dict = {label: score for label, score in zip(out['labels'], out['scores'])}
                # Ensure order is correct from TRL 1 to 9
                ordered_scores = [score_dict.get(f"TRL {i} Description", 0.0) for i in range(1, 10)]
                results.append({"zeroshot_probs_9": ordered_scores})
        except Exception as e:
            self.logger.error(f"Error in zero-shot batch processing: {e}")
            results = [{"zeroshot_probs_9": [1/9]*9}] * len(texts)
        return results

    def _predict_supervised_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generates 3-category TRL predictions using the fine-tuned SciBERT model."""
        if not self.supervised_model:
            # If model isn't loaded, return a neutral (uninformative) probability
            return [{"supervised_probs": [1/3, 1/3, 1/3]}] * len(texts)
            
        inputs = self.supervised_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.supervised_model(**inputs).logits
            
        probabilities = F.softmax(logits, dim=-1).cpu().tolist()
        return [{"supervised_probs": probs} for probs in probabilities]

    def _refine_assessment(self, indicator_res, zeroshot_res, supervised_res, source_type):
        """Combines all signals using a weighted average to produce the final assessment."""
        # --- Define weights for combining signals. These can be tuned. ---
        W_SUPERVISED = 0.60
        W_ZEROSHOT = 0.25
        W_RULES = 0.15

        # 1. Get supervised probabilities (already 3-cat)
        supervised_probs = supervised_res.get("supervised_probs", [1/3]*3)

        # 2. Get zero-shot probabilities (9-cat) and map to 3 categories
        zeroshot_9 = zeroshot_res.get("zeroshot_probs_9", [1/9]*9)
        early_zs = sum(zeroshot_9[0:3]); mid_zs = sum(zeroshot_9[3:6]); late_zs = sum(zeroshot_9[6:9])
        total_zs = early_zs + mid_zs + late_zs
        zeroshot_3 = [p / total_zs for p in [early_zs, mid_zs, late_zs]] if total_zs > 0 else [1/3]*3
        
        # 3. Get rule-based probabilities (already 3-cat)
        rule_probs = indicator_res.get("rule_probs", [1/3]*3)
        
        # 4. Apply source-type heuristic adjustment (optional, a small nudge)
        if source_type and 'patent' in source_type.lower():
            # Slightly boost mid/late probabilities for patents
            rule_probs[1] *= 1.1; rule_probs[2] *= 1.2
            total = sum(rule_probs); rule_probs = [p / total for p in rule_probs]

        # 5. Calculate final weighted-average score for each category
        final_scores = [
            (W_SUPERVISED * sup) + (W_ZEROSHOT * zs) + (W_RULES * rule)
            for sup, zs, rule in zip(supervised_probs, zeroshot_3, rule_probs)
        ]

        # 6. Determine final category and confidence
        final_category_idx = int(np.argmax(final_scores))
        confidence = final_scores[final_category_idx]
        id2label = {0: "Early (TRL 1-3)", 1: "Mid (TRL 4-6)", 2: "Late (TRL 7-9)"}

        return {
            "trl_category": id2label[final_category_idx],
            "trl_category_id": final_category_idx,
            "confidence": float(confidence),
            "details": {
                "final_scores": final_scores,
                "supervised_probs": supervised_probs,
                "zeroshot_probs_3cat": zeroshot_3,
                "rule_probs_3cat": rule_probs,
                "source_type": source_type,
                "found_indicators": indicator_res.get("found_indicators", [])
            }
        }

    def process_batch(self, texts: List[str], batch_source_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Processes a batch of documents to assess TRL, handling empty texts.
        """
        if not texts:
            return []
            
        self.logger.info(f"Assessing TRL for a batch of {len(texts)} documents.")
        if batch_source_types is None:
            batch_source_types = ["unknown"] * len(texts)

        # Handle empty/invalid texts to avoid processing errors
        full_results = [None] * len(texts)
        valid_texts, valid_indices, valid_sources = [], [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
                valid_sources.append(batch_source_types[i])
            else:
                full_results[i] = {
                    "trl_category": "Unknown", "trl_category_id": -1, "confidence": 0.0,
                    "details": {"error": "Input text was empty."}
                }
        
        if not valid_texts:
            return full_results

        # Run all processors on the valid texts
        indicator_results = self._extract_indicators_batch(valid_texts)
        zeroshot_results = self._classify_zeroshot_batch(valid_texts)
        supervised_results = self._predict_supervised_batch(valid_texts)
        
        # Refine and combine results for each valid document
        refined_assessments = [
            self._refine_assessment(ind_res, zs_res, sup_res, src)
            for ind_res, zs_res, sup_res, src in zip(indicator_results, zeroshot_results, supervised_results, valid_sources)
        ]

        # Place refined results back into their original positions in the full list
        for i, assessment in enumerate(refined_assessments):
            original_index = valid_indices[i]
            full_results[original_index] = assessment
            
        return full_results