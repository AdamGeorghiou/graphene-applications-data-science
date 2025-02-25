import os
import sys
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import logging
from typing import Dict, List, Any, Optional
from transformers import pipeline
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

class GrapheneNLPProcessor:
    def __init__(self):
        # Set up paths
        self.processed_data_dir = os.path.join(project_root, 'data', 'processed')
        self.output_dir = os.path.join(project_root, 'data', 'nlp_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Set up NLP tools
        self._setup_nlp_tools()
        
        # Initialize application keywords and category mapping
        self._initialize_keywords()
        
        # Set up SpaCy Matcher for application extraction
        self._setup_application_matcher()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger('nlp_processor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, 'nlp_processing.log'))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        logger.propagate = False
        return logger
        
    def _setup_nlp_tools(self):
        """Initialize NLP tools and download required resources"""
        print("Setting up NLP tools...")
        
        # Download NLTK resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
        
        # Initialize SpaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("Loaded SpaCy model")
        except OSError:
            print("Downloading SpaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize NLTK tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentiment analyzer
        print("Loading sentiment analysis model...")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def _initialize_keywords(self):
        """Initialize application keywords and category mapping"""
        # Keywords used for fallback extraction and matcher pattern generation
        self.application_keywords = [
            # Energy applications
            'battery', 'supercapacitor', 'solar cell', 'energy storage',
            'fuel cell', 'photovoltaic', 'energy harvesting',
            
            # Electronic applications
            'transistor', 'semiconductor', 'conductor', 'electronic', 
            'circuit', 'sensor', 'biosensor', 'display', 'electrode',
            'touchscreen', 'flexible electronics',
            
            # Material applications
            'composite', 'coating', 'membrane', 'filter', 'barrier',
            'reinforcement', 'additive', 'nanomaterial',
            
            # Chemical applications
            'catalyst', 'photocatalyst', 'electrocatalyst', 'oxidation',
            'reduction', 'chemical conversion',
            
            # Biological applications
            'drug delivery', 'tissue engineering', 'biomedical',
            'biosensing', 'cellular', 'antibacterial',
            
            # Environmental applications
            'water treatment', 'gas separation', 'pollution control',
            'environmental remediation', 'water purification'
        ]
        # Mapping keywords to industry categories
        self.CATEGORY_MAPPING = {
            # Energy
            "battery": "Energy",
            "supercapacitor": "Energy",
            "solar cell": "Energy",
            "energy storage": "Energy",
            "fuel cell": "Energy",
            "photovoltaic": "Energy",
            "energy harvesting": "Energy",
            # Electronics
            "transistor": "Electronics",
            "semiconductor": "Electronics",
            "conductor": "Electronics",
            "electronic": "Electronics",
            "circuit": "Electronics",
            "sensor": "Electronics",
            "biosensor": "Electronics",
            "display": "Electronics",
            "electrode": "Electronics",
            "touchscreen": "Electronics",
            "flexible electronics": "Electronics",
            # Materials
            "composite": "Materials",
            "coating": "Materials",
            "membrane": "Materials",
            "filter": "Materials",
            "barrier": "Materials",
            "reinforcement": "Materials",
            "additive": "Materials",
            "nanomaterial": "Materials",
            # Chemicals
            "catalyst": "Chemical",
            "photocatalyst": "Chemical",
            "electrocatalyst": "Chemical",
            "oxidation": "Chemical",
            "reduction": "Chemical",
            "chemical conversion": "Chemical",
            # Biomedical
            "drug delivery": "Biomedical",
            "tissue engineering": "Biomedical",
            "biomedical": "Biomedical",
            "biosensing": "Biomedical",
            "cellular": "Biomedical",
            "antibacterial": "Biomedical",
            # Environmental
            "water treatment": "Environmental",
            "gas separation": "Environmental",
            "pollution control": "Environmental",
            "environmental remediation": "Environmental",
            "water purification": "Environmental"
        }
    
    def _setup_application_matcher(self):
        """Set up SpaCy Matcher to extract graphene application phrases."""
        from spacy.matcher import Matcher
        self.matcher = Matcher(self.nlp.vocab)
        patterns = []
        # Create patterns that look for "graphene" followed by one or more tokens matching a keyword
        for keyword in self.application_keywords:
            pattern = [
                {"LOWER": "graphene"},
                {"LOWER": keyword, "OP": "+"}
            ]
            patterns.append(pattern)
            # Also match "graphene-based" formulations
            pattern_based = [
                {"LOWER": "graphene"},
                {"LOWER": "based"},
                {"LOWER": keyword, "OP": "+"}
            ]
            patterns.append(pattern_based)
        self.matcher.add("GRAPHENE_APP", patterns)
    
    def categorize_application(self, text: str) -> str:
        """Assign an industry category to a given application phrase."""
        for keyword, category in self.CATEGORY_MAPPING.items():
            if keyword in text.lower():
                return category
        return "Other"
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        try:
            if pd.isna(text):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize and lemmatize
            words = word_tokenize(text)
            words = [self.lemmatizer.lemmatize(word) for word in words if word.isalnum()]
            
            # Remove stopwords
            words = [word for word in words if word not in self.stop_words]
            
            return ' '.join(words)
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing text: {str(e)}")
            return text
    
    def extract_applications(self, text: str) -> List[Dict[str, str]]:
        """Extract graphene application phrases and assign categories"""
        applications = []
        try:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                app_phrase = span.text.strip()
                category = self.categorize_application(app_phrase)
                # Use the sentence containing the match as context
                context = span.sent.text if span.sent is not None else text
                applications.append({
                    'application': app_phrase,
                    'category': category,
                    'context': context
                })
            # Fallback: also check for standalone keywords if not captured by matcher
            lower_text = text.lower()
            for keyword in self.application_keywords:
                if keyword in lower_text:
                    # Avoid duplicates from matcher
                    if not any(keyword in app['application'].lower() for app in applications):
                        # Find first sentence containing the keyword
                        for sent in doc.sents:
                            if keyword in sent.text.lower():
                                applications.append({
                                    'application': keyword,
                                    'category': self.categorize_application(keyword),
                                    'context': sent.text
                                })
                                break
            return applications
        except Exception as e:
            self.logger.error(f"Error in application extraction: {str(e)}")
            return []
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using SpaCy"""
        try:
            doc = self.nlp(text)
            return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            return []
    
    def perform_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on the text"""
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            return {
                'label': result['label'],
                'score': float(result['score'])
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'UNKNOWN', 'score': 0.0}
    
    def process_document(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single document"""
        try:
            # Combine title and abstract for analysis
            full_text = f"{row['title']} {row['abstract']}"
            
            # Process text components
            processed_text = self.preprocess_text(full_text)
            applications = self.extract_applications(full_text)
            sentiment = self.perform_sentiment_analysis(full_text)
            entities = self.extract_entities(full_text)
            
            return {
                'id': row['id'],
                'source': row['source'],
                'title': row['title'],
                'abstract': row['abstract'],
                'processed_text': processed_text,
                'applications': applications,
                'sentiment': sentiment,
                'entities': entities
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            print(f"Error processing document: {str(e)}")
            return None
    
    def process_all_documents(self):
        """Process all documents"""
        try:
            # Load cleaned data
            input_path = os.path.join(self.processed_data_dir, 'cleaned_graphene_data.csv')
            if not os.path.exists(input_path):
                print(f"No cleaned data found at {input_path}")
                return None, None
            
            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} documents for processing")
            
            # Process documents
            results = []
            for idx, row in df.iterrows():
                print(f"Processing document {idx+1}/{len(df)}")
                result = self.process_document(row)
                if result:
                    results.append(result)
            
            if results:
                # Save results
                output_path = os.path.join(self.output_dir, 'nlp_results.json')
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Generate summary
                summary = self.generate_analysis_summary(results)
                return results, summary
            else:
                print("No valid results to save")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            print(f"Error processing documents: {str(e)}")
            return None, None
            
    def generate_analysis_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of NLP analysis"""
        try:
            # Count raw application phrases and their categories
            all_apps = [app['application'] for result in results 
                        for app in result['applications']]
            app_counts = Counter(all_apps)
            
            # Count application categories
            all_categories = [app['category'] for result in results 
                              for app in result['applications']]
            category_counts = Counter(all_categories)
            
            # Count entity types
            all_entities = [ent['label'] for result in results 
                            for ent in result['entities']]
            entity_counts = Counter(all_entities)
            
            # Sentiment distribution
            sentiment_dist = Counter([result['sentiment']['label'] 
                                      for result in results])
            
            summary = {
                'total_documents': len(results),
                'total_applications_found': len(all_apps),
                'unique_applications': len(set(all_apps)),
                'top_applications': dict(app_counts.most_common(10)),
                'application_category_distribution': dict(category_counts),
                'entity_distribution': dict(entity_counts.most_common()),
                'sentiment_distribution': dict(sentiment_dist)
            }
            
            # Save summary
            summary_path = os.path.join(self.output_dir, 'nlp_summary.txt')
            with open(summary_path, 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}:\n{value}\n\n")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {'error': str(e), 'total_documents': len(results)}
    
def main():
    processor = GrapheneNLPProcessor()
    print("Starting NLP processing...")
    results, summary = processor.process_all_documents()
    
    if results and summary:
        print("\nProcessing complete!")
        print(f"Results saved to: {os.path.join(processor.output_dir, 'nlp_results.json')}")
        print(f"Summary saved to: {os.path.join(processor.output_dir, 'nlp_summary.txt')}")

if __name__ == "__main__":
    main()
