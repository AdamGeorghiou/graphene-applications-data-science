# src/processors/nlp/document_processor.py
from typing import Dict, List, Any, Optional
import os
import json
import pandas as pd
import numpy as np
import logging
import traceback
import torch
from tqdm import tqdm

# Assuming these imports point to the refactored versions below
from .application_classifier import ApplicationClassifier
from .entity_extractor import GrapheneEntityExtractor
from .relation_extractor import RelationExtractor
from .trl_assessor import TRLAssessor
# Base processor import might not be directly needed here but good for context
# from .base_processor import BaseNLPProcessor

class GrapheneDocumentProcessor:
    """
    Main processor that orchestrates the NLP pipeline for graphene applications
    using efficient batch processing.
    """

    def __init__(self, use_gpu=True, output_dir=None):
        self.logger = self._setup_logging()
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # Note: Check if sub-processors handle MPS correctly
                self.device = "mps"
                self.logger.warning("MPS device selected. Ensure sub-processors handle MPS appropriately (some pipelines might fallback).")
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.logger.info(f"Selected device: {self.device}")

        self.output_dir = output_dir or os.path.join(os.getcwd(), 'data', 'nlp_results')
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"Initializing NLP pipeline with device: {self.device}")

        # Initialize component processors
        self._initialize_processors()

    def _setup_logging(self):
        """Set up logging for the document processor"""
        logger = logging.getLogger("GrapheneDocumentProcessor")
        # Set level to DEBUG to capture detailed sub-processor logs if needed,
        # but control console output level via handler
        logger.setLevel(logging.DEBUG)
        logger.propagate = False # Prevent duplicate logging if root logger is configured

        if not logger.handlers:
            # File handler for persistent logs (DEBUG level)
            log_file = os.path.join(os.path.dirname(__file__), '../../../logs', 'document_processor.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG) # Log everything to file
            logger.addHandler(file_handler)

            # Stream handler for console output (INFO level)
            stream_handler = logging.StreamHandler()
            # Simpler format for console, focusing on INFO/WARN/ERROR
            stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - GrapheneDocumentProcessor - %(message)s')
            stream_handler.setFormatter(stream_formatter)
            # Set console level higher to avoid flooding during batch processing
            stream_handler.setLevel(logging.INFO) # Show INFO and above on console
            logger.addHandler(stream_handler)

        return logger

    def _initialize_processors(self):
        """Initialize all component processors"""
        # Ensure sub-processor logs are captured if needed by setting their level
        logging.getLogger("GrapheneEntityExtractor").setLevel(logging.INFO)
        logging.getLogger("ApplicationClassifier").setLevel(logging.INFO)
        logging.getLogger("RelationExtractor").setLevel(logging.INFO)
        logging.getLogger("TRLAssessor").setLevel(logging.INFO)

        try:
            self.logger.info("Initializing NLP component processors...")

            # Pass the determined device to each processor
            self.entity_extractor = GrapheneEntityExtractor(
                model_name="allenai/scibert_scivocab_uncased",
                device=self.device
            )
            # Use getattr for safety in case device attribute isn't set on error
            self.logger.info(f"Entity extractor initialized on {getattr(self.entity_extractor, 'device', 'N/A')}")

            self.application_classifier = ApplicationClassifier(
                model_name="facebook/bart-large-mnli",
                device=self.device
            )
            self.logger.info(f"Application classifier initialized on {getattr(self.application_classifier, 'device', 'N/A')}")

            self.relation_extractor = RelationExtractor(
                # Base model might be SciBERT, classifier model (BART) is internal
                model_name="allenai/scibert_scivocab_uncased",
                device=self.device
            )
            self.logger.info(f"Relation extractor initialized on {getattr(self.relation_extractor, 'device', 'N/A')}")

            self.trl_assessor = TRLAssessor(
                 # Classifier model (BART) is internal to TRLAssessor
                 model_name="facebook/bart-large-mnli", # Pass the classifier model name here if needed by base class
                 device=self.device
            )
            self.logger.info(f"TRL assessor initialized on {getattr(self.trl_assessor, 'device', 'N/A')}")

            self.logger.info("All NLP processors initialized successfully")

        except Exception as e:
            self.logger.error(f"Fatal error initializing processors: {str(e)}", exc_info=True)
            raise # Re-raise the exception to halt execution if initialization fails

    def process_batch(self, documents: List[Dict], batch_size: int = 32):
        """
        Process a list of documents in batches using the refactored sub-processors.

        Args:
            documents: List of document dictionaries.
            batch_size: Size of batches for processing.

        Returns:
            List of processed document results including potential errors.
        """
        total_docs = len(documents)
        self.logger.info(f"Starting processing of {total_docs} documents in batches of {batch_size}")
        results = [None] * total_docs  # Pre-allocate results list

        num_batches = (len(documents) + batch_size - 1) // batch_size
        batch_iterator = tqdm(range(0, total_docs, batch_size),
                              total=num_batches,
                              desc="Processing Batches",
                              unit="batch")

        processed_count = 0
        error_count = 0

        for i in batch_iterator:
            start_index = i
            end_index = min(i + batch_size, total_docs)
            current_batch_docs = documents[start_index:end_index]
            # Map batch indices back to original indices
            original_indices = list(range(start_index, end_index))

            # --- Prepare batch data ---
            batch_texts = []
            valid_indices_in_batch = [] # Indices relative to the start of the full documents list
            source_types_for_batch = [] # Source types ONLY for valid texts

            for doc_idx, doc in zip(original_indices, current_batch_docs):
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                full_text = f"{title} {abstract}".strip()

                if not full_text:
                    results[doc_idx] = { # Assign error directly to the correct index
                        "id": doc.get('id', f"unknown_{doc_idx}"),
                        "source": doc.get('source', 'unknown'),
                        "error": "Empty document (no title or abstract)"
                    }
                    error_count += 1
                    continue # Skip this document for batch processing

                batch_texts.append(full_text)
                source_types_for_batch.append(doc.get('source', ''))
                valid_indices_in_batch.append(doc_idx) # Store original index

            # If all docs in batch were empty or skipped
            if not valid_indices_in_batch:
                self.logger.debug(f"Skipping batch starting at {start_index} as no valid documents found.")
                continue

            # --- Execute batched NLP tasks ---
            try:
                self.logger.debug(f"Processing {len(batch_texts)} valid texts for batch starting at original index {start_index}")

                # 1. Entity Extraction
                self.logger.debug("Starting entity extraction batch...")
                entity_results_batch = self.entity_extractor.process_batch(batch_texts)
                self.logger.debug(f"Finished entity extraction batch (got {len(entity_results_batch)} results).")

                # 2. Application Classification
                self.logger.debug("Starting application classification batch...")
                application_results_batch = self.application_classifier.process_batch(batch_texts)
                self.logger.debug(f"Finished application classification batch (got {len(application_results_batch)} results).")

                # 3. Relation Extraction (needs extracted entities)
                self.logger.debug("Starting relation extraction batch...")
                # Ensure entity_results_batch length matches batch_texts length
                if len(entity_results_batch) != len(batch_texts):
                    self.logger.error(f"Entity results length mismatch ({len(entity_results_batch)} vs {len(batch_texts)}). Skipping relation extraction for batch.")
                    # Handle error - maybe mark docs as errored or provide empty relations
                    relation_results_batch = [{} for _ in batch_texts] # Provide empty results
                else:
                    relation_results_batch = self.relation_extractor.process_batch(batch_texts, entity_results_batch)
                self.logger.debug(f"Finished relation extraction batch (got {len(relation_results_batch)} results).")

                # 4. TRL Assessment
                self.logger.debug("Starting TRL assessment batch...")
                # Pass source types corresponding ONLY to the valid texts
                trl_results_batch = self.trl_assessor.process_batch(batch_texts, source_types_for_batch)
                self.logger.debug(f"Finished TRL assessment batch (got {len(trl_results_batch)} results).")

                # --- Combine results for the valid documents in this batch ---
                self.logger.debug(f"Combining results for {len(valid_indices_in_batch)} valid documents.")
                for batch_idx, original_doc_idx in enumerate(valid_indices_in_batch):
                    # Check if sub-processor results lengths match expected batch size
                    if batch_idx >= len(entity_results_batch) or \
                       batch_idx >= len(application_results_batch) or \
                       batch_idx >= len(relation_results_batch) or \
                       batch_idx >= len(trl_results_batch):
                        self.logger.error(f"Result length mismatch for document index {original_doc_idx} (batch index {batch_idx}). Marking as error.")
                        results[original_doc_idx] = {
                            "id": documents[original_doc_idx].get('id', f"unknown_{original_doc_idx}"),
                            "source": documents[original_doc_idx].get('source', 'unknown'),
                            "error": "Sub-processor result length mismatch"
                        }
                        error_count += 1
                        continue

                    doc = documents[original_doc_idx] # Get original doc for metadata
                    year = None
                    if 'published_date' in doc and doc['published_date'] is not None:
                         pub_date = doc['published_date']
                         if isinstance(pub_date, str) and len(pub_date) >= 4:
                             try: year = int(pub_date[:4])
                             except (ValueError, TypeError): pass # Ignore conversion errors
                         elif isinstance(pub_date, (int, float)):
                             # Ensure year is within a reasonable range
                              if 1900 <= int(pub_date) <= 2100:
                                   year = int(pub_date)

                    # Assign results to the correct pre-allocated slot
                    results[original_doc_idx] = {
                        "id": doc.get('id', f"unknown_{original_doc_idx}"),
                        "source": doc.get('source', 'unknown'),
                        "title": doc.get('title', ''),
                        # "abstract": doc.get('abstract', ''), # Optional: Exclude abstract
                        "year": year,
                        "applications": application_results_batch[batch_idx],
                        "entities": entity_results_batch[batch_idx],
                        "relations": relation_results_batch[batch_idx],
                        "trl_assessment": trl_results_batch[batch_idx]
                    }
                    processed_count +=1

            except Exception as batch_err:
                self.logger.error(f"Error processing batch starting at original index {start_index}: {batch_err}", exc_info=True)
                # Mark all valid docs attempted in this batch as errored
                for original_doc_idx in valid_indices_in_batch:
                     results[original_doc_idx] = { # Assign error directly
                         "id": documents[original_doc_idx].get('id', f"unknown_{original_doc_idx}"),
                         "source": documents[original_doc_idx].get('source', 'unknown'),
                         "error": f"Batch processing error: {str(batch_err)}",
                         "traceback": traceback.format_exc() # Include traceback for debugging
                     }
                     error_count += 1

        self.logger.info(f"Finished processing {total_docs} documents. Processed successfully: {processed_count}. Errors: {error_count}.")
        return results


    def process_collection(self, documents_df: pd.DataFrame, batch_size: int = 32):
        """
        Process an entire collection of documents from a DataFrame using batch processing.

        Args:
            documents_df: Pandas DataFrame containing document data.
            batch_size: The size of batches to process.

        Returns:
            Tuple of (results, summary)
        """
        self.logger.info(f"Processing collection of {len(documents_df)} documents using batch size {batch_size}")

        if documents_df.empty:
            self.logger.warning("Input DataFrame is empty. No documents to process.")
            return [], {}

        # Convert DataFrame to list of dictionaries
        # Handle potential NaN values gracefully during conversion
        try:
             documents = documents_df.replace({np.nan: None}).to_dict('records')
        except Exception as e:
             self.logger.error(f"Error converting DataFrame to dictionary list: {e}", exc_info=True)
             return [], {"error": "Failed to convert DataFrame to records"}


        # Process all documents using process_batch
        results = self.process_batch(documents, batch_size=batch_size)

        # Filter out None placeholders just in case (shouldn't be needed with direct assignment)
        # results = [res for res in results if res is not None]

        # Generate summary statistics
        summary = self.generate_analysis_summary(results)

        # Save results and summary
        self._save_results(results, summary)

        return results, summary

    def _safe_json_convert(self, obj):
        """Convert objects to JSON-safe types, handling NumPy 2.0+ and Pandas"""
        if isinstance(obj, dict):
            return {k: self._safe_json_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
             # Convert tuples to lists for JSON
            return [self._safe_json_convert(item) for item in obj]
        # Use np.generic to catch all numpy scalar types
        elif isinstance(obj, np.integer): # Catches np.int32, np.int64 etc.
            return int(obj)
        elif isinstance(obj, np.floating):# Catches np.float32, np.float64 etc.
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._safe_json_convert(obj.tolist())
        elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
             # Handle Pandas NaT (Not a Time) explicitly
            return obj.isoformat() if not pd.isna(obj) else None
        elif isinstance(obj, (np.bool_, bool)): # Handle numpy and python bool
            return bool(obj)
        # General Pandas NA check (covers None, np.nan, pd.NA, pd.NaT)
        elif pd.isna(obj):
            return None
        # Allow basic types that are JSON serializable
        elif isinstance(obj, (str, int, float)) or obj is None:
             return obj
        else:
             # Attempt to convert unknown types to string as a fallback
            self.logger.debug(f"Attempting string conversion for unknown type: {type(obj)}")
            try:
                return str(obj)
            except Exception as e:
                 self.logger.warning(f"Could not convert object of type {type(obj)} to string: {e}. Returning None.")
                 return None # Fallback for truly unserializable types


    def _save_results(self, results, summary):
        """Save processed results and summary to files"""
        if not results:
             self.logger.warning("No results to save.")
             return
        try:
            self.logger.info(f"Attempting to save {len(results)} results and summary.")
            # Ensure results are JSON-safe
            self.logger.debug("Starting JSON conversion for results...")
            json_safe_results = self._safe_json_convert(results)
            self.logger.debug("Finished JSON conversion for results.")

            # Save full results
            results_path = os.path.join(self.output_dir, 'graphene_nlp_results.json')
            self.logger.info(f"Saving full results to {results_path}")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved results ({os.path.getsize(results_path)} bytes)")

            # Save summary as JSON
            summary_path = os.path.join(self.output_dir, 'graphene_analysis_summary.json')
            self.logger.debug("Starting JSON conversion for summary...")
            json_safe_summary = self._safe_json_convert(summary)
            self.logger.debug("Finished JSON conversion for summary.")
            self.logger.info(f"Saving summary to {summary_path}")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved summary ({os.path.getsize(summary_path)} bytes)")

            # Save key summary parts as CSV
            self._save_summary_csv(summary)

        except TypeError as te:
             self.logger.error(f"TypeError during JSON serialization: {te}. Check _safe_json_convert.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}", exc_info=True)

    def _save_summary_csv(self, summary):
        """Helper to save parts of the summary to CSV files."""
        if not isinstance(summary, dict):
            self.logger.warning("Summary is not a dictionary, cannot save CSV files.")
            return
        try:
            # Save application categories as CSV
            app_cats = summary.get('application_categories')
            if isinstance(app_cats, list) and app_cats:
                app_df = pd.DataFrame(app_cats)
                app_path = os.path.join(self.output_dir, 'application_categories.csv')
                app_df.to_csv(app_path, index=False, encoding='utf-8')
                self.logger.info(f"Saved application categories to {app_path}")

             # Save subcategories as CSV
            sub_cats = summary.get('application_subcategories')
            if isinstance(sub_cats, list) and sub_cats:
                 subcat_df = pd.DataFrame(sub_cats)
                 subcat_path = os.path.join(self.output_dir, 'application_subcategories.csv')
                 subcat_df.to_csv(subcat_path, index=False, encoding='utf-8')
                 self.logger.info(f"Saved application subcategories to {subcat_path}")

            # Save materials and fabrication methods as CSV
            materials_dict = summary.get('materials')
            fab_methods_dict = summary.get('fabrication_methods')
            if isinstance(materials_dict, dict) and isinstance(fab_methods_dict, dict):
                materials_df = pd.DataFrame(materials_dict.items(), columns=['material', 'count'])
                materials_path = os.path.join(self.output_dir, 'materials_count.csv')
                materials_df.to_csv(materials_path, index=False, encoding='utf-8')

                fabrication_df = pd.DataFrame(fab_methods_dict.items(), columns=['method', 'count'])
                fabrication_path = os.path.join(self.output_dir, 'fabrication_methods.csv')
                fabrication_df.to_csv(fabrication_path, index=False, encoding='utf-8')
                self.logger.info(f"Saved materials and fabrication counts")

            # Save yearly trends if available
            trends_data = summary.get('yearly_trends')
            if isinstance(trends_data, list) and trends_data:
                 time_df = pd.DataFrame(trends_data)
                 # Fill NaN for potentially missing app/TRL columns per year
                 time_df = time_df.fillna(0)
                 # Convert float columns potentially created by fillna back to int if appropriate
                 for col in time_df.columns:
                      if col.startswith(('app_', 'trl_', 'document_count')):
                          try:
                              time_df[col] = time_df[col].astype(int)
                          except ValueError:
                              self.logger.warning(f"Could not convert column {col} to int in yearly trends.")
                 time_path = os.path.join(self.output_dir, 'yearly_trends.csv')
                 time_df.to_csv(time_path, index=False, encoding='utf-8')
                 self.logger.info(f"Saved yearly trends data to {time_path}")

        except Exception as e:
            self.logger.error(f"Error saving summary CSV files: {str(e)}", exc_info=True)


    def generate_analysis_summary(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive analysis summary from processed results.

        Args:
            results: List of processed document results (can include error dicts).

        Returns:
            Dict containing summary statistics and analysis.
        """
        self.logger.info("Generating analysis summary")
        if not results:
            return {"error": "No results list provided to summarize"}

        try:
            # Filter out documents with processing errors *before* analysis
            valid_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
            total_processed = len(results)
            error_count = total_processed - len(valid_results)
            self.logger.info(f"Summarizing {len(valid_results)} valid results (out of {total_processed} processed, {error_count} errors).")

            if not valid_results:
                return {
                    "total_documents_processed": total_processed,
                    "valid_documents_analyzed": 0,
                    "documents_with_errors": error_count,
                    "error": "No valid results to summarize after filtering errors."
                    }

            # Initialize counters and containers
            application_categories = {}
            subcategories = {}
            materials = {}
            fabrication_methods = {}
            metrics_by_application = {} # Structure: {app: {metric_key: [values]}}
            trl_distribution = {i: 0 for i in range(1, 10)} # TRL 1-9
            yearly_data = {} # Structure: {year: {'count': N, 'applications': {app: N}, 'trl_levels': {trl: N}}}
            source_counts = {}

            # Process each valid document result
            for doc in valid_results:
                doc_id = doc.get('id', 'unknown')
                source = doc.get('source', 'unknown')
                year = doc.get('year') # Already processed in process_batch

                # Count sources
                source_counts[source] = source_counts.get(source, 0) + 1

                # Add to yearly data if year is valid integer
                is_valid_year = isinstance(year, int) # Check for int specifically now
                if is_valid_year:
                    if year not in yearly_data:
                        yearly_data[year] = {
                            'count': 0,
                            'applications': {},
                            'trl_levels': {trl: 0 for trl in range(1, 10)}
                        }
                    yearly_data[year]['count'] += 1

                # Process applications
                apps_data = doc.get('applications', {})
                primary_cats = apps_data.get('categories', [])
                doc_primary_cats = [] # Store primary cats found in this doc
                for category_info in primary_cats:
                    cat_name = category_info.get('category')
                    confidence = category_info.get('confidence', 0)
                    if cat_name:
                        doc_primary_cats.append(cat_name)
                        if cat_name not in application_categories:
                            application_categories[cat_name] = {'count': 0, 'confidence_sum': 0.0, 'documents': []}
                        application_categories[cat_name]['count'] += 1
                        # Ensure confidence is float before summing
                        try:
                            application_categories[cat_name]['confidence_sum'] += float(confidence)
                        except (ValueError, TypeError): pass # Ignore if confidence isn't numeric
                        application_categories[cat_name]['documents'].append(doc_id)
                        # Add to yearly data
                        if is_valid_year:
                            yearly_data[year]['applications'][cat_name] = yearly_data[year]['applications'].get(cat_name, 0) + 1

                # Process subcategories
                subcats_data = apps_data.get('subcategories', {})
                for primary_cat, subcat_list in subcats_data.items():
                     if isinstance(subcat_list, list): # Ensure it's a list
                         for subcat_info in subcat_list:
                             subcat_name = subcat_info.get('subcategory')
                             if subcat_name:
                                 if subcat_name not in subcategories:
                                     subcategories[subcat_name] = {'primary_category': primary_cat, 'count': 0, 'documents': []}
                                 subcategories[subcat_name]['count'] += 1
                                 subcategories[subcat_name]['documents'].append(doc_id)

                # Process entities (Materials, Fabrication Methods)
                entities_data = doc.get('entities', {})
                # Aggregate materials
                for material in entities_data.get('materials', []):
                    material_text = material.get('text', '').lower().strip()
                    if material_text: materials[material_text] = materials.get(material_text, 0) + 1
                # Aggregate fabrication methods
                for method in entities_data.get('fabrication_methods', []):
                    method_text = method.get('text', '').lower().strip()
                    if method_text: fabrication_methods[method_text] = fabrication_methods.get(method_text, 0) + 1

                # Process performance metrics (simplified aggregation - using first primary category found)
                metrics_data = entities_data.get('metrics', [])
                related_app = doc_primary_cats[0] if doc_primary_cats else "Unknown" # Use first primary cat as heuristic

                for metric in metrics_data:
                    value = metric.get('value')
                    # Ensure value is a number before proceeding
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_text_orig = metric.get('text', 'unknown_metric')
                        # Try to normalize metric names (example)
                        metric_key = metric_text_orig.lower().split('(')[0].strip() # Basic normalization
                        if 'conductivity' in metric_key: metric_key = 'conductivity'
                        elif 'strength' in metric_key: metric_key = 'strength'
                        # ... add more normalization rules ...

                        if related_app not in metrics_by_application: metrics_by_application[related_app] = {}
                        if metric_key not in metrics_by_application[related_app]: metrics_by_application[related_app][metric_key] = []
                        metrics_by_application[related_app][metric_key].append(float(value))


                # Process TRL assessment
                trl_data = doc.get('trl_assessment', {})
                trl_level = trl_data.get('trl_level') # Can be 0 or None
                if isinstance(trl_level, int) and 1 <= trl_level <= 9:
                    trl_distribution[trl_level] += 1
                    # Add to yearly data
                    if is_valid_year:
                        yearly_data[year]['trl_levels'][trl_level] += 1

            # --- Aggregate and Format Summaries ---

            # Application Categories Summary
            app_categories_summary = []
            total_valid = len(valid_results)
            for cat, data in application_categories.items():
                avg_confidence = data['confidence_sum'] / data['count'] if data['count'] > 0 else 0
                app_categories_summary.append({
                    'category': cat,
                    'count': data['count'],
                    'avg_confidence': round(avg_confidence, 4),
                    'percentage': round((data['count'] / total_valid) * 100, 2) if total_valid > 0 else 0
                    # 'doc_ids': data['documents'][:10] # Optional: Sample doc IDs
                })
            app_categories_summary.sort(key=lambda x: x['count'], reverse=True)

            # Subcategories Summary
            subcategories_summary = []
            for subcat, data in subcategories.items():
                subcategories_summary.append({
                    'subcategory': subcat,
                    'primary_category': data['primary_category'],
                    'count': data['count'],
                    'percentage': round((data['count'] / total_valid) * 100, 2) if total_valid > 0 else 0
                })
            subcategories_summary.sort(key=lambda x: x['count'], reverse=True)

             # Metrics Summary (calculate stats)
            metrics_summary = {}
            for app, metrics_dict in metrics_by_application.items():
                metrics_summary[app] = {}
                for metric_key, values in metrics_dict.items():
                    if values:
                        try:
                            # np.mean/min/max handle lists directly
                            metrics_summary[app][metric_key] = {
                                'avg': round(float(np.mean(values)), 4),
                                'min': round(float(np.min(values)), 4),
                                'max': round(float(np.max(values)), 4),
                                'count': len(values)
                            }
                        except Exception as metric_err:
                            self.logger.warning(f"Could not compute stats for metric '{metric_key}' in app '{app}': {metric_err}")


            # Yearly Trends Summary
            yearly_trends = []
            for year in sorted(yearly_data.keys()):
                year_entry = {'year': year, 'document_count': yearly_data[year]['count']}
                 # Add app counts for this year
                for cat, count in yearly_data[year]['applications'].items():
                     year_entry[f'app_{cat}'] = count
                 # Add TRL counts for this year
                for trl, count in yearly_data[year]['trl_levels'].items():
                     if count > 0: # Only include TRLs with counts > 0 for this year
                         year_entry[f'trl_{trl}'] = count
                yearly_trends.append(year_entry)

            # Create final summary dictionary
            summary = {
                'total_documents_processed': total_processed,
                'valid_documents_analyzed': len(valid_results),
                'documents_with_errors': error_count,
                'source_distribution': dict(sorted(source_counts.items(), key=lambda item: item[1], reverse=True)),
                'application_categories': app_categories_summary,
                'application_subcategories': subcategories_summary,
                'materials': dict(sorted(materials.items(), key=lambda item: item[1], reverse=True)[:50]), # Top 50
                'fabrication_methods': dict(sorted(fabrication_methods.items(), key=lambda item: item[1], reverse=True)[:30]), # Top 30
                'performance_metrics_summary': metrics_summary,
                'trl_distribution': trl_distribution,
                'yearly_trends': yearly_trends
            }

            self.logger.info("Analysis summary generated successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Fatal error during analysis summary generation: {str(e)}", exc_info=True)
            # Return basic error info if summary fails completely
            return {
                    "total_documents_processed": len(results) if results else 0,
                    "error": f"Summary generation failed: {str(e)}"
                }