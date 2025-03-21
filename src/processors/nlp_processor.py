import os
import sys
import pandas as pd
import numpy as np
import spacy
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import logging
from typing import Dict, List, Any, Optional
from src.utils.normalisation import unify_fabrication_label

# With this code
try:
    from transformers import pipeline
    PIPELINE_AVAILABLE = True
except:
    PIPELINE_AVAILABLE = False
    print("Warning: transformers pipeline not available, some features may be limited")
    
import json
import traceback  # Add this import

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

def json_safe_dict(d):
    """Convert a dictionary to be JSON-safe (handle numpy types, etc.)"""
    if isinstance(d, dict):
        return {k: json_safe_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [json_safe_dict(item) for item in list(d)]
    elif isinstance(d, tuple):
        return [json_safe_dict(item) for item in list(d)]
    elif isinstance(d, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(d)
    elif isinstance(d, (np.float64, np.float16, np.float32)):  # Changed from np.float_
        return float(d)
    elif isinstance(d, (np.ndarray,)):
        return json_safe_dict(d.tolist())
    else:
        return d

class GrapheneNLPProcessor:
    def __init__(self, use_advanced_nlp=True):
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
        
        # Optional advanced NLP components
        self.use_advanced_nlp = use_advanced_nlp
        if use_advanced_nlp:
            self._setup_advanced_nlp()
        
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
        # In the _setup_nlp_tools method, change:
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

    def _setup_advanced_nlp(self):
        """Initialize advanced NLP components like BERTopic"""
        self.logger.info("Setting up advanced NLP components...")
        try:
            # Initialize embedding model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Loaded sentence transformer model")
            
            # Initialize BERTopic
            from bertopic import BERTopic
            self.topic_model = BERTopic(embedding_model=self.embedding_model)
            print("Initialized BERTopic model")
        except Exception as e:
            self.logger.error(f"Error setting up advanced NLP: {str(e)}")
            print(f"Error setting up advanced NLP: {str(e)}")
            self.use_advanced_nlp = False  # Disable if it fails

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
        
    def _get_sentence_containing(self, text, phrase):
        """Helper method to find a sentence containing a specific phrase with error handling"""
        try:
            # Try using NLTK's sent_tokenize
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if phrase.lower() in sentence.lower():
                    return sentence
        except:
            # First fallback: Use SpaCy for sentence segmentation
            try:
                doc = self.nlp(text)
                for sent in doc.sents:
                    if phrase.lower() in sent.text.lower():
                        return sent.text
            except:
                # Second fallback: Simple period-based splitting
                text_parts = text.split('. ')
                for part in text_parts:
                    if phrase.lower() in part.lower():
                        return part + '.'
        return ""
    
    def unify_application_label(self, label: str) -> str:
        # Merge "graphene composite" with "composite"
        if label.lower() in ['graphene composite']:
            return 'composite'
        return label

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
                                    'application': self.unify_application_label(keyword),
                                    'category': self.categorize_application(keyword),
                                    'context': sent.text
                                })
                                break
            # Add more weight to applications mentioned in titles
            if 'title' in text and len(applications) == 0:
                # The text is likely just a title or very short
                # Do more aggressive matching for applications in titles
                for keyword in self.application_keywords:
                    if keyword.lower() in text.lower():
                        applications.append({
                            'application': self.unify_application_label(keyword),
                            'category': self.categorize_application(keyword),
                            'context': text,
                            'confidence': 'high'  # Title mentions are usually central to the document
                        })
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
    
    def extract_performance_metrics(self, text: str, application_type: str = None) -> Dict[str, Any]:
        """Extract performance metrics for graphene applications"""
        metrics = {}
        
        # Generic patterns for numerical properties with units
        patterns = {
            'conductivity': r'(?:electrical|thermal)?\s*conductivity\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:S/m|S/cm|W/mK)',
            'strength': r'(?:tensile|mechanical)\s*strength\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:GPa|MPa)',
            'surface_area': r'(?:specific)?\s*surface\s*area\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:m²/g|m2/g)',
            'efficiency': r'(?:power\s*conversion|energy|quantum)\s*efficiency\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*%',
            'capacitance': r'(?:specific)?\s*capacitance\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:F/g|mF/cm²)'
        }
        
        # Application-specific patterns
        if application_type:
            if application_type.lower() == 'supercapacitor':
                patterns.update({
                    'energy_density': r'energy\s*density\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:Wh/kg|Wh/L)',
                    'power_density': r'power\s*density\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*(?:W/kg|kW/kg)'
                })
            elif application_type.lower() in ['sensor', 'biosensor']:
                patterns.update({
                    'sensitivity': r'sensitivity\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)',
                    'detection_limit': r'(?:detection\s*limit|LOD)\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)'
                })
        
        # Search for metrics
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[metric_name] = float(matches[0])
        
        return metrics
    
    def extract_fabrication_methods(self, text: str) -> List[Dict[str, str]]:
        fabrication_methods = []
        # Broader patterns to catch more mentions
        simple_patterns = [
            r"(\w+\s*exfoliation)",
            r"(chemical vapor deposition|CVD)",
            r"(reduced graphene oxide|rGO)",
            r"(graphene oxide|GO)",
            r"(Hummers method)",
            r"(epitaxial growth)",
            r"(sonication)",
            r"(thermal reduction)",
            r"(chemical reduction)"
        ]
        
        for pattern in simple_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize the extracted method using the imported function
                normalized_method = unify_fabrication_label(match)
                fabrication_methods.append({
                    'method': normalized_method,
                    'context': self._get_sentence_containing(text, match)
                })
        
        return fabrication_methods

    
    def extract_property_application_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships between graphene properties and applications"""
        relations = []
        
        # Define graphene properties
        properties = [
            'conductivity', 'strength', 'flexibility', 'transparency', 
            'surface area', 'thermal', 'mechanical', 'electrical', 'optical'
        ]
        
        # For each property, try to find its connection to applications
        for prop in properties:
            # Look for sentences that mention both property and application terms
            pattern = rf'(\b{prop}\b).{{0,100}}(?:{"|".join(self.application_keywords)})'
            alt_pattern = rf'(?:{"|".join(self.application_keywords)}).{{0,100}}(\b{prop}\b)'
            
            for p in [pattern, alt_pattern]:
                matches = re.findall(p, text, re.IGNORECASE)
                for match in matches:
                    # Extract the full sentence for context
                    try:
                        sentences = sent_tokenize(text)
                        context = next((s for s in sentences if prop in s.lower()), "")
                    except:
                        # Fallback if sent_tokenize fails
                        context = ""
                    
                    # Determine which application it relates to
                    app = next((app for app in self.application_keywords if app in context.lower()), "unknown")
                    
                    relations.append({
                        'property': prop,
                        'application': app,
                        'context': context.strip()
                    })
        
        return relations
    
    def extract_publication_timeline(self, text: str, year: Optional[int] = None) -> Dict[str, Any]:
        # Prioritize metadata year if available
        if year is not None:
            # Metadata year is available
            timeline_info = {'year': year, 'is_novel': False, 'advancement_type': None}
        else:
            # Try to extract from text as fallback
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            years = re.findall(year_pattern, text)
            if years:
                # Use the most recent year mentioned (often publication year)
                timeline_info = {'year': int(max(years)), 'is_novel': False, 'advancement_type': None}
            else:
                timeline_info = {'year': None, 'is_novel': False, 'advancement_type': None}
        
        # Check for novelty indicators
        novelty_patterns = [
            r'first\s+(?:time|report|demonstration)',
            r'novel\s+(?:approach|method|technique)',
            r'breakthrough',
            r'significant\s+advancement'
        ]
        
        for pattern in novelty_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                timeline_info['is_novel'] = True
                break
        
        # Determine type of advancement
        if 'performance' in text.lower() or 'improved' in text.lower():
            timeline_info['advancement_type'] = 'performance_improvement'
        elif 'synthesis' in text.lower() or 'fabrication' in text.lower():
            timeline_info['advancement_type'] = 'fabrication_method'
        elif 'application' in text.lower() or 'use case' in text.lower():
            timeline_info['advancement_type'] = 'new_application'
        
        return timeline_info
    
    def assess_trl_level(self, text: str, source: str = None) -> Dict[str, Any]:
        """Estimate Technology Readiness Level (TRL) based on text content"""
        trl_assessment = {'trl_score': 0, 'evidence': []}
        
        # TRL indicators (keywords mapping to approximate TRL levels)
        trl_indicators = {
            # Basic research (TRL 1-2)
            1: ['basic principle', 'concept', 'theoretical', 'fundamental research'],
            2: ['technology concept', 'analytical', 'paper study', 'formulated'],
            
            # Lab validation (TRL 3-4)
            3: ['proof of concept', 'experimental', 'laboratory', 'validated analytically'],
            4: ['component validation', 'lab environment', 'breadboard', 'low fidelity'],
            
            # Simulated environment (TRL 5-6)
            5: ['validated in relevant environment', 'simulated environment', 'high fidelity'],
            6: ['demonstrated in relevant environment', 'prototype', 'beta version'],
            
            # Real environment (TRL 7-9)
            7: ['demonstrated in operational environment', 'real environment', 'field tested'],
            8: ['system complete', 'qualified', 'test and demonstration', 'commercial ready'],
            9: ['proven in operational environment', 'commercial product', 'market', 'industrialization']
        }
        
        # Check for indicators of each TRL level
        max_trl = 0
        for trl, indicators in trl_indicators.items():
            for indicator in indicators:
                if re.search(r'\b' + indicator + r'\b', text, re.IGNORECASE):
                    max_trl = max(max_trl, trl)
                    try:
                        context = re.search(r'[^.]*\b' + indicator + r'\b[^.]*', text, re.IGNORECASE).group(0)
                    except:
                        context = "Context extraction failed"
                    
                    trl_assessment['evidence'].append({
                        'trl_level': trl, 
                        'indicator': indicator,
                        'context': context
                    })
        
        trl_assessment['trl_score'] = max_trl
        # Source-based TRL enhancement
        if 'patent' in source.lower() if source else False:
            # Patents generally indicate at least concept validation
            if trl_assessment['trl_score'] == 0:
                trl_assessment['trl_score'] = 2
                trl_assessment['evidence'].append({
                    'trl_level': 2,
                    'indicator': 'patent filing',
                    'context': 'Document is a patent'
                })
            
            # Check for commercialization indicators in patents
            if any(ind in text.lower() for ind in ['commercialization', 'product', 'manufacturing']):
                trl_assessment['trl_score'] = max(trl_assessment['trl_score'], 6)
                trl_assessment['evidence'].append({
                    'trl_level': 6,
                    'indicator': 'commercial intent',
                    'context': 'Patent indicates commercial readiness'
                })
        return trl_assessment
    
    def classify_research_type(self, text: str, source: str) -> str:
        """Classify if research is academic or industrial/applied"""
        # Default classification based on source
        if 'patent' in source.lower():
            classification = 'industrial'
        else:
            classification = 'academic'
        
        # Check for industrial indicators in text
        industrial_indicators = [
            'commercial', 'industry', 'product', 'manufacturing', 
            'production', 'market', 'scalable', 'cost-effective'
        ]
        
        academic_indicators = [
            'theoretical', 'fundamental', 'principle', 'model',
            'simulation', 'investigated', 'studied'
        ]
        
        # Count indicators
        industrial_count = sum(1 for ind in industrial_indicators if ind in text.lower())
        academic_count = sum(1 for ind in academic_indicators if ind in text.lower())
        
        # Decide based on both source and content
        if industrial_count > academic_count + 2:  # Strong industrial indicators
            classification = 'industrial'
        elif academic_count > industrial_count + 2:  # Strong academic indicators
            classification = 'academic'
        
        return classification
    
    def _json_sanitize(self, obj):
        """Convert NaN and None values to empty strings for JSON serialization"""
        if obj is None:
            return ""
        elif isinstance(obj, float) and np.isnan(obj):
            return ""
        elif isinstance(obj, dict):
            return {k: self._json_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._json_sanitize(item) for item in list(obj)]
        else:
            return obj

    def process_document(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single document with enhanced analysis"""
        try:
            # Use the 'id' if it exists; otherwise, generate one
            doc_id = row.get('id')
            if not doc_id:
                doc_id = f"{row['source']}_{abs(hash(row['title']))}"
            
            # Combine title and abstract for analysis (if abstract is missing, use an empty string)
            full_text = f"{row['title']} {row.get('abstract', '')}"
            
            # Special handling for patent documents with no abstract
            is_patent = 'patent' in row['source'].lower()
            has_abstract = row.get('abstract', '') != ''
            
            # Extract applications
            applications = self.extract_applications(full_text)
            
            # Special case for patents without abstracts
            if is_patent and not has_abstract:
                # For patents without abstracts, the title is especially important
                # Extract applications more aggressively from title
                title_only_apps = []
                for keyword in self.application_keywords:
                    if keyword.lower() in row['title'].lower():
                        title_only_apps.append({
                            'application': self.unify_application_label(keyword),
                            'category': self.categorize_application(keyword),
                            'context': row['title'],
                            'confidence': 'high'
                        })
                
                # If we found applications in title, use them
                if title_only_apps:
                    applications = title_only_apps
            
            app_types = [app['application'] for app in applications]
            
            # Use publication year if available
            year = row.get('published_date', None)
            if isinstance(year, (str, float)) and str(year).isdigit():
                year = int(str(year))
            
            # Enhanced processing
            result = {
                'id': doc_id,
                'source': row['source'],
                'title': row['title'],
                'abstract': row.get('abstract', ''),
                'applications': applications,
                # Enhanced analysis components
                'publication_info': self.extract_publication_timeline(full_text, year),
                'fabrication_methods': self.extract_fabrication_methods(full_text),
                'research_type': self.classify_research_type(full_text, row['source']),
                'trl_assessment': self.assess_trl_level(full_text, row['source']),
                'property_relations': self.extract_property_application_relations(full_text),
                # Get performance metrics for each identified application
                'performance_metrics': {
                    app_type: self.extract_performance_metrics(full_text, app_type) 
                    for app_type in set(app_types)
                }
            }
            
            # Sanitize the result to remove NaN values
            sanitized_result = self._json_sanitize(result)
            
            return sanitized_result
                    
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            print(f"Error processing document: {str(e)}")
            return None

    
    def process_collection(self, df):
        """Process an entire collection of documents"""
        print(f"Processing {len(df)} documents")
        self.logger.info(f"Starting to process {len(df)} documents")
        
        # Basic processing
        results = []
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Print progress every 10 documents
                print(f"Processing document {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)")
            result = self.process_document(row)
            if result:
                results.append(result)
        
        # Advanced processing if enabled
        if self.use_advanced_nlp and results:
            print("Applying advanced NLP techniques...")
            results = self._apply_advanced_nlp(df, results)
        
        # Save results
        if results:
            self._save_results(results)
            
            # Generate summary
            summary = self.generate_enhanced_analysis_summary(results)
            return results, summary
        else:
            print("No valid results to process")
            return None, None
    
    def _apply_advanced_nlp(self, df, results):
        """Apply advanced NLP techniques and integrate results"""
        try:
            # Extract documents for topic modeling
            documents = (df['title'].fillna('') + " " + df['abstract'].fillna('')).tolist()
            
            # Fit topic model
            print("Fitting BERTopic model. This may take some time...")
            topics, probs = self.topic_model.fit_transform(documents)
            
            # Save topic model
            model_path = os.path.join(self.output_dir, "bertopic_model")
            self.topic_model.save(model_path)
            print(f"Saved BERTopic model to {model_path}")
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Associate topics with documents
            for i, result in enumerate(results):
                topic_id = int(topics[i])
                result['topic'] = topic_id
                # Get the keywords for this topic if it's not -1 (outlier)
                if topic_id != -1:
                    result['topic_keywords'] = self.topic_model.get_topic(topic_id)
                else:
                    result['topic_keywords'] = []
                    
            print("Advanced NLP processing complete")
            return results
        except Exception as e:
            self.logger.error(f"Error in advanced NLP: {str(e)}")
            print(f"Error in advanced NLP: {str(e)}")
            return results  # Return original results if advanced processing fails


    def _save_results(self, results):
        """Save processed results to files"""
        try:
            # Make sure results are JSON-safe by using json_safe_dict
            json_safe_results = [json_safe_dict(result) for result in results]
            
            # Save main results
            output_path = os.path.join(self.output_dir, 'nlp_results.json')
            with open(output_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            print(f"Results saved to: {output_path}")
            
            # If we have topics, save documents with topics
            if self.use_advanced_nlp:
                # Create a DataFrame with document IDs and topics
                topics_df = pd.DataFrame([
                    {'id': result['id'], 
                    'topic': result['topic'], 
                    'keywords': result.get('topic_keywords', [])}
                    for result in results
                ])
                topics_path = os.path.join(self.output_dir, 'documents_with_topics.csv')
                topics_df.to_csv(topics_path, index=False)
                print(f"Topic assignments saved to: {topics_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")

    def generate_enhanced_analysis_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced summary of NLP analysis"""
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
                            for ent in result.get('entities', [])]
            entity_counts = Counter(all_entities)
            
            # TRL distribution
            trl_distribution = Counter([result['trl_assessment']['trl_score'] for result in results])
            
            # Research type distribution
            research_type_dist = Counter([result['research_type'] for result in results])
            
            # Fabrication methods
            all_methods = [method['method'] for result in results 
                        for method in result.get('fabrication_methods', [])]
            fabrication_counts = Counter(all_methods)
            
            # Performance metrics summary
            # For each app type, collect all values for each metric
            performance_summary = {}
            for result in results:
                for app_type, metrics in result.get('performance_metrics', {}).items():
                    if app_type not in performance_summary:
                        performance_summary[app_type] = {}
                    
                    for metric, value in metrics.items():
                        if metric not in performance_summary[app_type]:
                            performance_summary[app_type][metric] = []
                        performance_summary[app_type][metric].append(value)
            
            # Calculate average, min, max for each metric
            performance_stats = {}
            for app_type, metrics in performance_summary.items():
                performance_stats[app_type] = {}
                for metric, values in metrics.items():
                    if values:
                        try:
                            performance_stats[app_type][metric] = {
                                'avg': float(np.mean(values)),
                                'min': float(min(values)),
                                'max': float(max(values)),
                                'count': len(values)
                            }
                        except Exception as e:
                            self.logger.error(f"Error calculating stats for {app_type}.{metric}: {str(e)}")
            
            

            # Timeline analysis: Extract valid years only
            years = []
            for result in results:
                year = result['publication_info'].get('year')

                if isinstance(year, str):
                    # Extract the first 4 digits if it's a full date like "2022-08-10"
                    extracted_year = year[:4] if year[:4].isdigit() else None
                elif isinstance(year, (int, float)):
                    # Convert numerical years safely
                    extracted_year = int(year)
                else:
                    extracted_year = None  # Handle None or unexpected types

                if extracted_year:  # Only add valid years
                    years.append(str(extracted_year))  # Store as string for consistency

            # Count occurrences of each year
            year_distribution = Counter(years)

            # Convert all years to integers for sorting if possible
            sorted_years = []
            for year in year_distribution:
                try:
                    sorted_years.append((int(year), year))  # Convert to integer for sorting
                except (ValueError, TypeError):
                    sorted_years.append((999999, year))  # Push invalid years to end

            # Sort by the integer value and create a dictionary using the original year format
            year_dict = {original_year: year_distribution[original_year] 
                        for _, original_year in sorted(sorted_years)}

            # Try to extract property-application relationships
            try:
                property_app_pairs = []
                for result in results:
                    for rel in result.get('property_relations', []):
                        if isinstance(rel, dict):
                            prop = rel.get('property', '')
                            app = rel.get('application', '')
                            if prop and app:
                                pair_key = f"{prop}-{app}"
                                property_app_pairs.append(pair_key)
                property_app_counts = Counter(property_app_pairs)
                top_pairs = dict(property_app_counts.most_common(10))
            except Exception as e:
                self.logger.error(f"Error processing property relationships: {str(e)}")
                top_pairs = {}

            
            summary = {
                'total_documents': len(results),
                'total_applications_found': len(all_apps),
                'unique_applications': len(set(all_apps)),
                'top_applications': dict(app_counts.most_common(10)),
                'application_category_distribution': dict(category_counts),
                'trl_distribution': dict(trl_distribution),
                'research_type_distribution': dict(research_type_dist),
                'fabrication_methods': dict(fabrication_counts.most_common(10)),
                'performance_metrics_summary': performance_stats,
                'year_distribution': year_dict,
                'top_property_application_pairs': top_pairs
            }
            
            # Convert summary to JSON-safe values
            json_safe_summary = json_safe_dict(summary)
            
            # Save as text summary
            try:
                summary_path = os.path.join(self.output_dir, 'enhanced_nlp_summary.txt')
                with open(summary_path, 'w') as f:
                    for key, value in json_safe_summary.items():
                        f.write(f"{key}:\n{value}\n\n")
                
                # Also save as JSON for easier parsing
                summary_json_path = os.path.join(self.output_dir, 'enhanced_nlp_summary.json')
                with open(summary_json_path, 'w') as f:
                    json.dump(json_safe_summary, f, indent=2)
                    
                print(f"Summary saved to: {summary_path}")
                print(f"Summary JSON saved to: {summary_json_path}")
            except Exception as e:
                self.logger.error(f"Error saving summary: {str(e)}")
                print(f"Error saving summary: {str(e)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced summary: {str(e)}")
            print(f"Error generating enhanced summary: {str(e)}, {traceback.format_exc()}")
            return {'error': str(e), 'total_documents': len(results)}
    
def main():
    print("Starting GrapheneNLPProcessor initialization...") 
    processor = GrapheneNLPProcessor(use_advanced_nlp=True)
    
    print("Processor initialized, loading data...")
    input_path = os.path.join(processor.processed_data_dir, 'cleaned_graphene_data.csv')
    if not os.path.exists(input_path):
        print(f"No cleaned data found at {input_path}")
        return
    
        
    df = pd.read_csv(input_path)
    print(f"Data loaded: {len(df)} records")
    # Process all documents in a single call
    print("Starting NLP processing with integrated advanced features...")
    results, summary = processor.process_collection(df)
    
    if results and summary:
        print("\nProcessing complete!")
        print(f"Results saved successfully")

if __name__ == "__main__":
    main()