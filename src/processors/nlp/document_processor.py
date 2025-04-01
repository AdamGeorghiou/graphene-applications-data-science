# src/processors/nlp/document_processor.py
from typing import Dict, List, Any, Optional
import os
import json
import pandas as pd
import numpy as np
import logging
import traceback
import torch 

from .application_classifier import ApplicationClassifier
from .entity_extractor import GrapheneEntityExtractor
from .relation_extractor import RelationExtractor
from .trl_assessor import TRLAssessor
from .base_processor import BaseNLPProcessor

class GrapheneDocumentProcessor:
    """
    Main processor that orchestrates the NLP pipeline for graphene applications
    """
    
    def __init__(self, use_gpu=True, output_dir=None):
        self.logger = self._setup_logging()
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
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
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_processors(self):
        """Initialize all component processors"""
        try:
            self.logger.info("Initializing NLP component processors...")

            self.entity_extractor = GrapheneEntityExtractor(
                model_name="allenai/scibert_scivocab_uncased",
                device=self.device
            )
            self.logger.info(f"Entity extractor initialized on {self.entity_extractor.device}")

            self.application_classifier = ApplicationClassifier(
                model_name="facebook/bart-large-mnli",
                device=self.device
            )
            self.logger.info(f"Application classifier initialized on {self.application_classifier.device}")

            self.relation_extractor = RelationExtractor(
                model_name="allenai/scibert_scivocab_uncased",
                device=self.device
            )
            self.logger.info(f"Relation extractor initialized on {self.relation_extractor.device}")

            self.trl_assessor = TRLAssessor(
                model_name="allenai/scibert_scivocab_uncased",
                device=self.device
            )
            self.logger.info(f"TRL assessor initialized on {self.trl_assessor.device}")

            self.logger.info("All NLP processors initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing processors: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def process_document(self, document):
        """
        Process a single document through the entire NLP pipeline
        
        Args:
            document: Dict containing document fields (title, abstract, etc.)
            
        Returns:
            Dict containing all extracted information
        """
        try:
            # Create document ID if not present
            doc_id = document.get('id')
            if not doc_id:
                doc_id = f"{document['source']}_{abs(hash(document['title']))}"
            
            # Combine title and abstract for analysis
            title = document.get('title', '')
            abstract = document.get('abstract', '')
            
            if not title and not abstract:
                self.logger.warning(f"Empty document: {doc_id}")
                return {
                    "id": doc_id,
                    "source": document.get('source', 'unknown'),
                    "error": "Empty document"
                }
            
            # Combine text for full analysis
            full_text = f"{title} {abstract}".strip()
            
            # Extract publication year if available
            year = None
            if 'published_date' in document:
                pub_date = document['published_date']
                if isinstance(pub_date, str) and len(pub_date) >= 4:
                    # Try extracting year from date string
                    try:
                        year = int(pub_date[:4])
                    except ValueError:
                        pass
                elif isinstance(pub_date, (int, float)):
                    year = int(pub_date)
            
            # Get source type
            source_type = document.get('source', '')
            
            # Step 1: Entity extraction
            self.logger.info(f"Extracting entities for document {doc_id}")
            entity_results = self.entity_extractor.process(full_text)
            
            # Step 2: Application classification
            self.logger.info(f"Classifying applications for document {doc_id}")
            application_results = self.application_classifier.process(full_text)
            
            # Step 3: Relation extraction
            self.logger.info(f"Extracting relations for document {doc_id}")
            relation_results = self.relation_extractor.process(full_text, entity_results)
            
            # Step 4: TRL assessment
            self.logger.info(f"Assessing TRL for document {doc_id}")
            trl_results = self.trl_assessor.process(full_text, source_type)
            
            # Combine all results
            result = {
                "id": doc_id,
                "source": document.get('source', 'unknown'),
                "title": title,
                "abstract": abstract,
                "year": year,
                
                # Extracted information
                "applications": application_results,
                "entities": entity_results,
                "relations": relation_results,
                "trl_assessment": trl_results
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {document.get('id', 'unknown')}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "id": document.get('id', 'unknown'),
                "source": document.get('source', 'unknown'),
                "error": str(e)
            }
    
    def process_batch(self, documents, batch_size=10):
        """
        Process a batch of documents
        
        Args:
            documents: List of document dicts
            batch_size: Number of documents to process at once
            
        Returns:
            List of processed document results
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)+batch_size-1)//batch_size}")
            
            batch_results = []
            for doc in batch:
                result = self.process_document(doc)
                batch_results.append(result)
            
            results.extend(batch_results)
            self.logger.info(f"Completed batch {i//batch_size + 1}")
        
        return results
    
    def process_collection(self, documents_df):
        """
        Process an entire collection of documents from a DataFrame

        Args:
            documents_df: Pandas DataFrame containing document data

        Returns:
            Tuple of (results, summary)
        """
        self.logger.info(f"Processing collection of {len(documents_df)} documents")

        # Convert DataFrame to list of dictionaries for processing
        documents = documents_df.to_dict('records')

        # Process all documents
        results = self.process_batch(documents)

        # Generate summary statistics
        summary = self.generate_analysis_summary(results)

        # Save results and summary
        self._save_results(results, summary)

        return results, summary

    def _safe_json_convert(self, obj):
        """Convert objects to JSON-safe types"""
        if isinstance(obj, dict):
            return {k: self._safe_json_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_json_convert(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._safe_json_convert(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._safe_json_convert(obj.tolist())
        elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _save_results(self, results, summary):
        """Save processed results and summary to files"""
        try:
            # Ensure results are JSON-safe
            json_safe_results = self._safe_json_convert(results)

            # Save full results
            results_path = os.path.join(self.output_dir, 'graphene_nlp_results.json')
            with open(results_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            self.logger.info(f"Saved results to {results_path}")

            # Save summary as JSON
            summary_path = os.path.join(self.output_dir, 'graphene_analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(self._safe_json_convert(summary), f, indent=2)
            self.logger.info(f"Saved summary to {summary_path}")

            # Save application categories as CSV
            if 'application_categories' in summary:
                app_df = pd.DataFrame(summary['application_categories'])
                app_path = os.path.join(self.output_dir, 'application_categories.csv')
                app_df.to_csv(app_path, index=False)
                self.logger.info(f"Saved application categories to {app_path}")

            # Save materials and fabrication methods as CSV
            if 'materials' in summary and 'fabrication_methods' in summary:
                materials_df = pd.DataFrame(summary['materials'].items(), columns=['material', 'count'])
                materials_path = os.path.join(self.output_dir, 'materials_count.csv')
                materials_df.to_csv(materials_path, index=False)

                fabrication_df = pd.DataFrame(summary['fabrication_methods'].items(), columns=['method', 'count'])
                fabrication_path = os.path.join(self.output_dir, 'fabrication_methods.csv')
                fabrication_df.to_csv(fabrication_path, index=False)
                self.logger.info(f"Saved materials and fabrication data")

            # Save time series data if available
            if 'yearly_trends' in summary:
                time_df = pd.DataFrame(summary['yearly_trends'])
                time_path = os.path.join(self.output_dir, 'yearly_trends.csv')
                time_df.to_csv(time_path, index=False)
                self.logger.info(f"Saved time series data to {time_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())

    def generate_analysis_summary(self, results):
        """
        Generate comprehensive analysis summary from processed results

        Args:
            results: List of processed document results

        Returns:
            Dict containing summary statistics and analysis
        """
        self.logger.info("Generating analysis summary")

        try:
            # Filter out documents with errors
            valid_results = [r for r in results if 'error' not in r]

            if not valid_results:
                return {"error": "No valid results to summarize"}

            # Initialize counters and containers
            application_categories = {}
            subcategories = {}
            materials = {}
            fabrication_methods = {}
            metrics_by_application = {}
            trl_distribution = {i: 0 for i in range(1, 10)}
            yearly_data = {}

            # Process each document
            for doc in valid_results:
                # Get year information
                year = doc.get('year')
                if year and isinstance(year, (int, float)) and 1900 <= year <= 2100:
                    if year not in yearly_data:
                        yearly_data[year] = {
                            'count': 0,
                            'applications': {},
                            'trl_levels': {}
                        }
                    yearly_data[year]['count'] += 1

                # Process applications
                apps = doc.get('applications', {})
                for category in apps.get('categories', []):
                    cat_name = category.get('category')
                    confidence = category.get('confidence', 0)

                    if cat_name:
                        if cat_name not in application_categories:
                            application_categories[cat_name] = {
                                'count': 0,
                                'confidence_sum': 0,
                                'documents': []
                            }
                        application_categories[cat_name]['count'] += 1
                        application_categories[cat_name]['confidence_sum'] += confidence
                        application_categories[cat_name]['documents'].append(doc['id'])

                        # Add to yearly data if available
                        if year and 1900 <= year <= 2100:
                            if cat_name not in yearly_data[year]['applications']:
                                yearly_data[year]['applications'][cat_name] = 0
                            yearly_data[year]['applications'][cat_name] += 1

                # Process subcategories
                for primary_cat, subcats in apps.get('subcategories', {}).items():
                    for subcat in subcats:
                        subcat_name = subcat.get('subcategory')
                        if subcat_name:
                            if subcat_name not in subcategories:
                                subcategories[subcat_name] = {
                                    'primary_category': primary_cat,
                                    'count': 0,
                                    'documents': []
                                }
                            subcategories[subcat_name]['count'] += 1
                            subcategories[subcat_name]['documents'].append(doc['id'])

                # Process materials
                for material in doc.get('entities', {}).get('materials', []):
                    material_text = material.get('text', '').lower()
                    if material_text:
                        materials[material_text] = materials.get(material_text, 0) + 1

                # Process fabrication methods
                for method in doc.get('entities', {}).get('fabrication_methods', []):
                    method_text = method.get('text', '').lower()
                    if method_text:
                        fabrication_methods[method_text] = fabrication_methods.get(method_text, 0) + 1

                # Process performance metrics
                for metric in doc.get('entities', {}).get('metrics', []):
                    metric_text = metric.get('text', '')
                    metric_value = metric.get('value')

                    # Try to determine which application this metric relates to
                    related_app = None
                    for relation in doc.get('relations', {}).get('material_performance_relations', []):
                        if metric_text in relation.get('metric', ''):
                            # Look up which application this material is used for
                            material = relation.get('material', '').lower()
                            for app_relation in doc.get('relations', {}).get('property_application_relations', []):
                                if material in app_relation.get('context', '').lower():
                                    related_app = app_relation.get('application')
                                    break

                    # If we found a related application, store the metric
                    if related_app and metric_value is not None:
                        if related_app not in metrics_by_application:
                            metrics_by_application[related_app] = {}

                        metric_key = None
                        if 'conductivity' in metric_text.lower():
                            metric_key = 'conductivity'
                        elif 'strength' in metric_text.lower():
                            metric_key = 'strength'
                        elif 'sensitivity' in metric_text.lower():
                            metric_key = 'sensitivity'
                        elif 'detection' in metric_text.lower() or 'lod' in metric_text.lower():
                            metric_key = 'detection_limit'
                        elif 'capacity' in metric_text.lower() or 'capacitance' in metric_text.lower():
                            metric_key = 'capacitance'
                        elif 'efficiency' in metric_text.lower():
                            metric_key = 'efficiency'
                        elif 'area' in metric_text.lower():
                            metric_key = 'surface_area'

                        if metric_key:
                            if metric_key not in metrics_by_application[related_app]:
                                metrics_by_application[related_app][metric_key] = []
                            metrics_by_application[related_app][metric_key].append(metric_value)

                # Process TRL assessment
                trl_level = doc.get('trl_assessment', {}).get('trl_level', 0)
                if 1 <= trl_level <= 9:
                    trl_distribution[trl_level] += 1

                    # Add to yearly data if available
                    if year and 1900 <= year <= 2100:
                        if trl_level not in yearly_data[year]['trl_levels']:
                            yearly_data[year]['trl_levels'][trl_level] = 0
                        yearly_data[year]['trl_levels'][trl_level] += 1

            # Calculate average metrics
            metrics_summary = {}
            for app, metrics in metrics_by_application.items():
                metrics_summary[app] = {}
                for metric_name, values in metrics.items():
                    if values:
                        metrics_summary[app][metric_name] = {
                            'avg': np.mean(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }

            # Prepare application categories summary
            app_categories_summary = []
            for cat, data in application_categories.items():
                avg_confidence = data['confidence_sum'] / data['count'] if data['count'] > 0 else 0
                app_categories_summary.append({
                    'category': cat,
                    'count': data['count'],
                    'avg_confidence': avg_confidence,
                    'percentage': data['count'] / len(valid_results) * 100
                })

            # Sort categories by count
            app_categories_summary.sort(key=lambda x: x['count'], reverse=True)

            # Prepare subcategories summary
            subcategories_summary = []
            for subcat, data in subcategories.items():
                subcategories_summary.append({
                    'subcategory': subcat,
                    'primary_category': data['primary_category'],
                    'count': data['count'],
                    'percentage': data['count'] / len(valid_results) * 100
                })

            # Sort subcategories by count
            subcategories_summary.sort(key=lambda x: x['count'], reverse=True)

            # Prepare yearly trends
            yearly_trends = []
            for year in sorted(yearly_data.keys()):
                year_entry = {
                    'year': year,
                    'document_count': yearly_data[year]['count']
                }

                # Add top applications
                for cat, count in yearly_data[year]['applications'].items():
                    year_entry[f'app_{cat}'] = count

                # Add TRL distribution
                for trl, count in yearly_data[year]['trl_levels'].items():
                    year_entry[f'trl_{trl}'] = count

                yearly_trends.append(year_entry)

            # Create final summary
            summary = {
                'total_documents': len(valid_results),
                'application_categories': app_categories_summary,
                'application_subcategories': subcategories_summary,
                'materials': dict(sorted(materials.items(), key=lambda x: x[1], reverse=True)[:50]),
                'fabrication_methods': dict(sorted(fabrication_methods.items(), key=lambda x: x[1], reverse=True)[:20]),
                'performance_metrics': metrics_summary,
                'trl_distribution': trl_distribution,
                'yearly_trends': yearly_trends
            }

            self.logger.info("Analysis summary generated successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}