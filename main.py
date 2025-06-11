#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Beulah11 - Command Line Version
Advanced Narrative Analysis Tool for batch processing
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
# Force non-interactive backend for command line usage
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import io
import base64
from PIL import Image as PILImage
import tempfile
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import textwrap
import networkx as nx
import csv
from typing import Dict, Any, List, Tuple
import subprocess
warnings.filterwarnings('ignore')

# Configure matplotlib for better output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Download required NLTK resources with better error handling
def setup_nltk():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {e}")

# Check optional dependencies
FEATURES = {
    'spacy': False,
    'pdf': False,
    'docx': False,
    'xlsx': False,
    'plotly': False,
    'wordcloud': False
}

# Check spaCy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        FEATURES['spacy'] = True
    except OSError:
        try:
            print("Downloading spaCy model...")
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            nlp = spacy.load("en_core_web_sm")
            FEATURES['spacy'] = True
        except:
            print("○ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            nlp = None
except ImportError:
    print("○ spaCy not available. Install with: pip install spacy")
    nlp = None

# Check PDF support
try:
    import pdfplumber
    FEATURES['pdf'] = True
except ImportError:
    print("○ PDF support not available. Install with: pip install pdfplumber")

# Check DOCX support
try:
    import docx
    FEATURES['docx'] = True
except ImportError:
    print("○ DOCX support not available. Install with: pip install python-docx")

# Check Excel support
try:
    import openpyxl
    FEATURES['xlsx'] = True
except ImportError:
    print("○ Excel support not available. Install with: pip install openpyxl")

# Check Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    FEATURES['plotly'] = True
except ImportError:
    print("○ Plotly not available. Install with: pip install plotly")

# Check WordCloud
try:
    from wordcloud import WordCloud
    FEATURES['wordcloud'] = True
except ImportError:
    print("○ WordCloud not available. Install with: pip install wordcloud")

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"○ Sentiment analyzer error: {e}")
    sia = None


class NarrativeEventAnalyzer:
    """Advanced narrative analysis based on Vossen, Caselli & Segers frameworks"""
    
    def __init__(self, doc):
        self.doc = doc
        self.events = []
        self.timeline = []
        self.actors = {}
        self.locations = {}
        self.plotline = {}
        self.storyline = {}
        
    def extract_events(self) -> Dict[str, Any]:
        """Extract events from the text using dependency parsing"""
        events = []
        
        for sent in self.doc.sents:
            # Look for verbs as potential actions
            for token in sent:
                if token.pos_ == "VERB":
                    # Find the subject of the verb
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child.text
                            break
                            
                    # Find the object of the verb
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            break
                            
                    # Find location (simplified)
                    location = "unknown"
                    for child in token.children:
                        if child.dep_ == "prep" and child.text in ["in", "at", "on"]:
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    location = grandchild.text
                                    break
                                    
                    # Find time expressions
                    time_expr = "unknown"
                    for ent in sent.ents:
                        if ent.label_ in ["DATE", "TIME"]:
                            time_expr = ent.text
                            break
                            
                    # Create event
                    if subject:
                        events.append({
                            "sentence": sent.text,
                            "action": token.lemma_,
                            "subject": subject,
                            "object": obj,
                            "location": location,
                            "time": time_expr,
                            "position": token.i
                        })
        
        self.events = events
        return {"events": events, "count": len(events)}
    
    def create_timeline(self) -> Dict[str, Any]:
        """Create a timeline from extracted events"""
        if not self.events:
            return {"timeline": [], "actors": {}, "locations": {}}
            
        # Sort events by their position in text
        timeline = sorted(self.events, key=lambda x: x["position"])
        
        # Extract actors and locations
        actors = {}
        locations = {}
        
        for event in timeline:
            # Track actors
            subject = event["subject"]
            if subject not in actors:
                actors[subject] = {
                    "mentions": 1,
                    "actions": [event["action"]],
                    "relationships": {}
                }
            else:
                actors[subject]["mentions"] += 1
                actors[subject]["actions"].append(event["action"])
            
            # Track relationships
            if event["object"] and event["object"] not in ["it", "this", "that"]:
                obj = event["object"]
                if obj not in actors[subject]["relationships"]:
                    actors[subject]["relationships"][obj] = [event["action"]]
                else:
                    actors[subject]["relationships"][obj].append(event["action"])
            
            # Track locations
            location = event["location"]
            if location != "unknown":
                if location not in locations:
                    locations[location] = {
                        "mentions": 1,
                        "events": [event["action"]]
                    }
                else:
                    locations[location]["mentions"] += 1
                    locations[location]["events"].append(event["action"])
        
        self.timeline = timeline
        self.actors = actors
        self.locations = locations
        
        return {"timeline": timeline, "actors": actors, "locations": locations}
    
    def analyze_plotline(self) -> Dict[str, Any]:
        """Analyze the plotline based on event sequences and causality"""
        if not self.events:
            return {"causal_relations": [], "action_chains": [], "main_actors": [], "key_locations": []}
            
        # Simple causal relation analysis
        causal_markers = ["because", "due to", "as a result", "consequently", "thus", "therefore", "so", "since"]
        causal_relations = []
        
        sents = list(self.doc.sents)
        for i, sent in enumerate(sents):
            for marker in causal_markers:
                if marker in sent.text.lower():
                    if i > 0:
                        prev_sent = sents[i-1]
                        causal_relations.append({
                            "cause": prev_sent.text,
                            "effect": sent.text,
                            "marker": marker
                        })
        
        # Extract main event chains
        action_chains = []
        current_chain = []
        current_actor = None
        
        for event in self.timeline:
            if current_actor is None:
                current_actor = event["subject"]
                current_chain.append(event)
            elif event["subject"] == current_actor:
                current_chain.append(event)
            else:
                if len(current_chain) > 1:
                    action_chains.append({
                        "actor": current_actor,
                        "events": current_chain.copy()
                    })
                current_actor = event["subject"]
                current_chain = [event]
                
        # Add the last chain if it exists
        if current_chain and len(current_chain) > 1:
            action_chains.append({
                "actor": current_actor,
                "events": current_chain
            })
            
        # Build plotline structure
        self.plotline = {
            "causal_relations": causal_relations,
            "action_chains": action_chains,
            "main_actors": sorted(self.actors.keys(), key=lambda x: self.actors[x]["mentions"], reverse=True)[:5] if self.actors else [],
            "key_locations": sorted(self.locations.keys(), key=lambda x: self.locations[x]["mentions"], reverse=True)[:5] if self.locations else [],
        }
        
        return self.plotline
    
    def analyze_storyline(self) -> Dict[str, Any]:
        """Analyze the storyline using narratology frameworks"""
        if not self.events or not self.plotline:
            return {"perspectives": [], "narrative_structure": {}, "themes": []}
            
        # Identify perspectives based on speech verbs
        speech_verbs = ["say", "tell", "speak", "claim", "state", "mention", "report"]
        perspectives = []
        
        for sent in self.doc.sents:
            for token in sent:
                if token.lemma_ in speech_verbs:
                    # Find who is speaking
                    speaker = None
                    for child in token.children:
                        if child.dep_ == "nsubj":
                            speaker = child.text
                            break
                            
                    if speaker:
                        perspectives.append({
                            "speaker": speaker,
                            "content": sent.text,
                            "verb": token.text
                        })
                
        # Simple narrative structure analysis
        total_events = len(self.timeline)
        if total_events > 0:
            beginning = self.timeline[:int(total_events * 0.25)]
            middle = self.timeline[int(total_events * 0.25):int(total_events * 0.75)]
            end = self.timeline[int(total_events * 0.75):]
            
            # Identify potential climax
            intensity_scores = []
            window_size = max(5, int(total_events * 0.1))
            
            for i in range(max(1, total_events - window_size + 1)):
                window = self.timeline[i:i+window_size]
                action_verbs = sum(1 for event in window if event["action"] in [
                    "fight", "attack", "confront", "argue", "challenge", "shout", 
                    "scream", "kill", "destroy", "reveal", "discover", "realize"
                ])
                intensity_scores.append((i, action_verbs))
                
            # Find the window with the highest intensity score
            if intensity_scores:
                climax_index = max(intensity_scores, key=lambda x: x[1])[0]
                climax = self.timeline[climax_index:climax_index+window_size]
            else:
                climax = []
        else:
            beginning = middle = end = climax = []
            
        # Extract main themes
        noun_chunks = list(self.doc.noun_chunks)
        theme_candidates = Counter([chunk.text.lower() for chunk in noun_chunks])
        themes = [theme for theme, count in theme_candidates.most_common(10)]
        
        # Build storyline structure
        self.storyline = {
            "perspectives": perspectives,
            "narrative_structure": {
                "exposition": beginning,
                "rising_action": middle[:len(middle)//2] if middle else [],
                "climax": climax,
                "falling_action": middle[len(middle)//2:] if middle else [],
                "resolution": end
            },
            "themes": themes
        }
        
        return self.storyline


class EnhancedBeulahNarrativeAnalyzer:
    """Enhanced analyzer with advanced narrative analysis and network visualization"""
    
    def __init__(self):
        self.narratives = {}
        self.current_narrative_id = None
        self._current_filename = ""
        self._id_counter = {}
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            self.lemmatizer = None
            self.stop_words = set()
        
        self.sia = sia
        self.nlp = nlp
        
    def load_file(self, file_path):
        """Load a file based on its extension"""
        try:
            # Store current filename
            self._current_filename = os.path.basename(file_path)
            ext = file_path.split('.')[-1].lower()
            print(f"\nLoading {ext.upper()} file: {self._current_filename}")
            
            # Read file content
            if ext in ['txt', 'text', 'json', 'csv', 'vtt']:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            else:
                # For binary files
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            
            # Call appropriate loader
            if ext in ['txt', 'text']:
                return self._load_text(file_content, self._current_filename)
            elif ext == 'json':
                return self._load_json(file_content, self._current_filename)
            elif ext == 'csv':
                return self._load_csv(file_content, self._current_filename)
            elif ext == 'pdf' and FEATURES['pdf']:
                return self._load_pdf(file_content, self._current_filename)
            elif ext == 'docx' and FEATURES['docx']:
                return self._load_docx(file_content, self._current_filename)
            elif ext in ['xls', 'xlsx'] and FEATURES['xlsx']:
                return self._load_excel(file_content, self._current_filename)
            elif ext == 'vtt':
                return self._load_vtt(file_content, self._current_filename)
            else:
                print(f"Unsupported file type: {ext}")
                return None
                
        except Exception as e:
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_unique_id(self, base_id):
        """Generate unique ID with counter"""
        if base_id not in self._id_counter:
            self._id_counter[base_id] = 0
            if base_id not in self.narratives:
                return base_id
        
        self._id_counter[base_id] += 1
        unique_id = f"{base_id}_{self._id_counter[base_id]}"
        
        # Ensure uniqueness
        while unique_id in self.narratives:
            self._id_counter[base_id] += 1
            unique_id = f"{base_id}_{self._id_counter[base_id]}"
        
        return unique_id
    
    def _create_narrative(self, narrative_id, title, text):
        """Create narrative structure"""
        return {
            'id': narrative_id,
            'title': title,
            'text': text,
            'sentences': [],
            'processed_text': "",
            'analysis': {},
            'metadata': {
                'source': self._current_filename.split('.')[-1].lower(),
                'filename': self._current_filename,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _load_text(self, content, file_name):
        """Load plain text file"""
        try:
            # Decode content
            if isinstance(content, bytes):
                # Try multiple encodings
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                text = None
                used_encoding = None
                
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        used_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                
                if text is None:
                    text = content.decode('utf-8', errors='ignore')
                    used_encoding = 'utf-8 (with errors ignored)'
                
                print(f"  Decoded with {used_encoding}")
            else:
                text = content
            
            # Generate unique ID
            base_name = os.path.splitext(file_name)[0]
            base_id = f"txt_{base_name.replace(' ', '_').lower()}"
            narrative_id = self._generate_unique_id(base_id)
            
            # Create narrative
            narrative = self._create_narrative(narrative_id, base_name, text)
            self.narratives[narrative_id] = narrative
            
            print(f"  ✓ Created narrative ID: {narrative_id}")
            print(f"  ✓ Text length: {len(text)} characters")
            
            return narrative_id
            
        except Exception as e:
            print(f"Text loading error: {e}")
            return None
    
    def _load_json(self, content, file_name):
        """Load JSON file"""
        try:
            # Decode content
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            data = json.loads(content)
            print(f"  JSON structure: {type(data).__name__}")
            
            # Extract text based on structure
            text = ""
            title = os.path.splitext(file_name)[0]
            
            if isinstance(data, dict):
                # Look for text fields
                text = data.get('text', '')
                if not text:
                    # Try other common fields
                    text = data.get('content', data.get('body', data.get('message', '')))
                if not text:
                    # Convert entire dict to string
                    text = json.dumps(data, indent=2)
                
                # Look for title
                title = data.get('title', data.get('name', title))
                
            elif isinstance(data, list):
                # Combine text from list items
                text_parts = []
                for item in data:
                    if isinstance(item, dict):
                        item_text = item.get('text', item.get('content', str(item)))
                        text_parts.append(item_text)
                    else:
                        text_parts.append(str(item))
                text = '\n\n'.join(text_parts)
            else:
                text = str(data)
            
            if text.strip():
                # Generate unique ID
                base_id = f"json_{title.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, title, text)
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text content found in JSON")
                return None
                
        except json.JSONDecodeError as e:
            print(f"  JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"  JSON loading error: {e}")
            return None
    
    def _load_csv(self, content, file_name):
        """Load CSV file"""
        try:
            # Handle different content types
            if isinstance(content, (bytes, memoryview)):
                content = bytes(content)
                text_content = content.decode('utf-8', errors='ignore')
            else:
                text_content = content
            
            # Detect delimiter
            sample_lines = text_content.split('\n')[:5]
            has_tabs = any('\t' in line for line in sample_lines)
            has_commas = any(',' in line for line in sample_lines)
            
            delimiter = '\t' if has_tabs and not has_commas else ','
            
            # Read CSV
            df = pd.read_csv(io.StringIO(text_content), sep=delimiter)
            print(f"  CSV shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"  Columns: {', '.join(df.columns[:5])}" + (" ..." if len(df.columns) > 5 else ""))
            
            # Extract text content
            text_parts = []
            text_columns = []
            
            # Look for text columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains substantial text
                    sample_texts = df[col].dropna().astype(str).head(10)
                    if len(sample_texts) > 0:
                        avg_length = sample_texts.str.len().mean()
                        # Look for columns with substantial text
                        if avg_length > 20 or col.lower() in ['text', 'transcript', 'response', 
                                                               'content', 'description', 'narrative',
                                                               'comment', 'message', 'story']:
                            text_columns.append(col)
                            texts = df[col].dropna().astype(str).tolist()
                            text_parts.extend(texts)
            
            # Fallback: use all string columns if no text columns found
            if not text_parts:
                print("  No text columns found, using all string columns")
                for col in df.columns:
                    if df[col].dtype == 'object':
                        texts = df[col].dropna().astype(str).tolist()
                        text_parts.extend(texts)
            
            # Final fallback: convert entire dataframe
            if not text_parts:
                print("  Converting entire dataframe to text")
                text = df.to_string()
            else:
                text = '\n\n'.join(text_parts)
                if text_columns:
                    print(f"  Found text in columns: {', '.join(text_columns)}")
            
            if text.strip():
                # Generate unique ID
                base_name = os.path.splitext(file_name)[0]
                base_id = f"csv_{base_name.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, base_name, text)
                narrative['metadata'].update({
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'text_columns': text_columns,
                    'delimiter': 'tab' if delimiter == '\t' else 'comma'
                })
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text content found in CSV")
                return None
                
        except Exception as e:
            print(f"  CSV loading error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_pdf(self, content, file_name):
        """Load PDF file"""
        try:
            import pdfplumber
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            text = ""
            page_count = 0
            
            with pdfplumber.open(tmp_path) as pdf:
                print(f"  PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        page_count += 1
            
            os.unlink(tmp_path)
            
            if text.strip():
                # Generate unique ID
                base_name = os.path.splitext(file_name)[0]
                base_id = f"pdf_{base_name.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, base_name, text)
                narrative['metadata']['pages'] = page_count
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Extracted text from {page_count} pages")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text found in PDF")
                return None
                
        except Exception as e:
            print(f"  PDF loading error: {e}")
            return None
    
    def _load_docx(self, content, file_name):
        """Load DOCX file"""
        try:
            import docx
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            doc = docx.Document(tmp_path)
            paragraphs = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            os.unlink(tmp_path)
            
            if paragraphs:
                text = "\n\n".join(paragraphs)
                
                # Generate unique ID
                base_name = os.path.splitext(file_name)[0]
                base_id = f"docx_{base_name.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, base_name, text)
                narrative['metadata']['paragraphs'] = len(paragraphs)
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Extracted {len(paragraphs)} paragraphs")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text found in DOCX")
                return None
                
        except Exception as e:
            print(f"  DOCX loading error: {e}")
            return None
    
    def _load_excel(self, content, file_name):
        """Load Excel file"""
        try:
            # Save to temporary file
            ext = file_name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Read Excel file
            df = pd.read_excel(tmp_path, sheet_name=None)  # Read all sheets
            
            os.unlink(tmp_path)
            
            # Extract text from all sheets
            all_text_parts = []
            sheet_info = []
            
            for sheet_name, sheet_df in df.items():
                print(f"  Sheet '{sheet_name}': {sheet_df.shape}")
                text_parts = []
                
                # Find text columns in this sheet
                for col in sheet_df.columns:
                    if sheet_df[col].dtype == 'object':
                        texts = sheet_df[col].dropna().astype(str).tolist()
                        if texts:
                            text_parts.extend(texts)
                
                if text_parts:
                    all_text_parts.extend(text_parts)
                    sheet_info.append(f"{sheet_name} ({len(text_parts)} text items)")
            
            if all_text_parts:
                text = "\n\n".join(all_text_parts)
                
                # Generate unique ID
                base_name = os.path.splitext(file_name)[0]
                base_id = f"excel_{base_name.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, base_name, text)
                narrative['metadata']['sheets'] = sheet_info
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Extracted text from sheets: {', '.join(sheet_info)}")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text found in Excel file")
                return None
                
        except Exception as e:
            print(f"  Excel loading error: {e}")
            return None
    
    def _load_vtt(self, content, file_name):
        """Load VTT (WebVTT subtitle) file"""
        try:
            # Decode content
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='ignore')
            else:
                text_content = content
            
            lines = text_content.split('\n')
            text_parts = []
            
            # Parse VTT format
            for line in lines:
                line = line.strip()
                # Skip timestamps and metadata
                if line and '-->' not in line and not line.startswith('WEBVTT') and not line.isdigit():
                    # Remove speaker labels if present
                    if ':' in line and len(line.split(':')[0].split()) <= 2:
                        line = ':'.join(line.split(':')[1:]).strip()
                    text_parts.append(line)
            
            if text_parts:
                text = ' '.join(text_parts)
                
                # Generate unique ID
                base_name = os.path.splitext(file_name)[0]
                base_id = f"vtt_{base_name.replace(' ', '_').lower()}"
                narrative_id = self._generate_unique_id(base_id)
                
                # Create narrative
                narrative = self._create_narrative(narrative_id, base_name, text)
                self.narratives[narrative_id] = narrative
                
                print(f"  ✓ Created narrative ID: {narrative_id}")
                print(f"  ✓ Extracted {len(text_parts)} text segments")
                print(f"  ✓ Text length: {len(text)} characters")
                
                return narrative_id
            else:
                print("  ✗ No text content found in VTT")
                return None
                
        except Exception as e:
            print(f"  VTT loading error: {e}")
            return None
    
    def analyze_narrative(self, narrative_id):
        """Run comprehensive analysis on a narrative"""
        if narrative_id not in self.narratives:
            print(f"Narrative ID not found: {narrative_id}")
            return False
        
        narrative = self.narratives[narrative_id]
        text = narrative['text']
        
        print(f"\nAnalyzing narrative: {narrative['title']}")
        
        try:
            # Initialize analysis structure
            narrative['analysis'] = {
                'basic_stats': {},
                'word_analysis': {},
                'sentiment': {},
                'themes': {},
                'entities': {},
                'topics': [],
                'summary': {},
                'events': {},
                'network': {},
                'problem_formulations': {}
            }
            
            # Basic preprocessing
            print("  Preprocessing text...", end=" ")
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            filtered_words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
            
            narrative['sentences'] = sentences
            print(f"✓ ({len(sentences)} sentences)")
            
            # Basic statistics
            narrative['analysis']['basic_stats'] = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'unique_words': len(set(filtered_words)),
                'vocabulary_richness': len(set(filtered_words)) / len(filtered_words) if filtered_words else 0,
                'avg_word_length': sum(len(w) for w in filtered_words) / len(filtered_words) if filtered_words else 0
            }
            
            # Word frequency analysis
            print("  Analyzing word frequency...", end=" ")
            word_freq = Counter(filtered_words).most_common(50)
            narrative['analysis']['word_analysis'] = {
                'word_frequency': word_freq,
                'top_words': word_freq[:20]
            }
            print("✓")
            
            # Theme analysis
            print("  Extracting themes...", end=" ")
            themes = {
                'relationships': ['relationship', 'family', 'friend', 'community', 'together', 'people', 'social'],
                'emotions': ['happy', 'sad', 'angry', 'fear', 'love', 'hate', 'joy', 'anxious', 'worried'],
                'time': ['time', 'day', 'year', 'month', 'week', 'hour', 'past', 'future', 'present', 'today'],
                'places': ['place', 'location', 'city', 'country', 'home', 'house', 'street', 'building'],
                'work': ['work', 'job', 'business', 'office', 'career', 'profession', 'employment', 'company'],
                'nature': ['nature', 'tree', 'water', 'sky', 'earth', 'animal', 'plant', 'weather'],
                'technology': ['technology', 'computer', 'internet', 'digital', 'software', 'data', 'system'],
                'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'medicine', 'care', 'treatment'],
                'conflict': ['problem', 'issue', 'challenge', 'conflict', 'struggle', 'fight', 'battle', 'war'],
                'solution': ['solution', 'solve', 'fix', 'answer', 'resolve', 'help', 'improve', 'better']
            }
            
            theme_counts = {}
            for theme, keywords in themes.items():
                count = sum(1 for word in filtered_words if any(kw in word for kw in keywords))
                if count > 0:
                    theme_counts[theme] = count
            
            narrative['analysis']['themes'] = theme_counts
            print(f"✓ ({len(theme_counts)} themes)")
            
            # Sentiment analysis
            if self.sia:
                print("  Analyzing sentiment...", end=" ")
                sentiment_scores = []
                overall_sentiment = self.sia.polarity_scores(text)
                
                # Analyze first 100 sentences for performance
                for sent in sentences[:100]:
                    scores = self.sia.polarity_scores(sent)
                    sentiment_scores.append(scores['compound'])
                
                narrative['analysis']['sentiment'] = {
                    'overall': overall_sentiment,
                    'progression': sentiment_scores,
                    'statistics': {
                        'mean': np.mean(sentiment_scores) if sentiment_scores else 0,
                        'std': np.std(sentiment_scores) if sentiment_scores else 0,
                        'min': np.min(sentiment_scores) if sentiment_scores else 0,
                        'max': np.max(sentiment_scores) if sentiment_scores else 0
                    },
                    'emotional_arc': self._detect_emotional_arc(sentiment_scores)
                }
                print("✓")
            
            # Extract entities if spacy available
            if self.nlp and FEATURES['spacy']:
                print("  Extracting entities...", end=" ")
                # Process first 10000 characters for performance
                doc = self.nlp(text[:10000])
                
                entities = defaultdict(list)
                for ent in doc.ents:
                    entities[ent.label_].append(ent.text)
                
                # Count and sort entities
                entity_counts = {}
                for label, ents in entities.items():
                    entity_counts[label] = Counter(ents).most_common(10)
                
                narrative['analysis']['entities'] = entity_counts
                print(f"✓ ({sum(len(e) for e in entity_counts.values())} entities)")
                
                # Advanced narrative analysis
                if FEATURES['spacy']:
                    print("  Performing advanced narrative analysis...")
                    event_analyzer = NarrativeEventAnalyzer(doc)
                    
                    # Extract events
                    print("    - Extracting events...", end=" ")
                    events_data = event_analyzer.extract_events()
                    print(f"✓ ({events_data['count']} events)")
                    
                    # Create timeline
                    print("    - Creating timeline...", end=" ")
                    timeline_data = event_analyzer.create_timeline()
                    print("✓")
                    
                    # Analyze plotline
                    print("    - Analyzing plotline...", end=" ")
                    plotline_data = event_analyzer.analyze_plotline()
                    print("✓")
                    
                    # Analyze storyline
                    print("    - Analyzing storyline...", end=" ")
                    storyline_data = event_analyzer.analyze_storyline()
                    print("✓")
                    
                    narrative['analysis']['events'] = {
                        'events': events_data,
                        'timeline': timeline_data,
                        'plotline': plotline_data,
                        'storyline': storyline_data
                    }
                    
                    # Create entity network
                    print("    - Building entity network...", end=" ")
                    network_data = self._create_entity_network(doc, entities)
                    narrative['analysis']['network'] = network_data
                    print("✓")
            
            # Topic modeling
            if len(sentences) > 10:
                print("  Discovering topics...", end=" ")
                try:
                    vectorizer = TfidfVectorizer(
                        max_features=50,
                        stop_words='english',
                        min_df=2,
                        max_df=0.8
                    )
                    
                    # Use first 100 sentences for performance
                    doc_term_matrix = vectorizer.fit_transform(sentences[:100])
                    
                    # LDA with fewer topics for stability
                    n_topics = min(5, len(sentences) // 10)
                    lda = LatentDirichletAllocation(
                        n_components=n_topics,
                        random_state=42,
                        max_iter=10
                    )
                    lda.fit(doc_term_matrix)
                    
                    # Extract topics
                    feature_names = vectorizer.get_feature_names_out()
                    topics = []
                    
                    for topic_idx, topic in enumerate(lda.components_):
                        top_words_idx = topic.argsort()[-10:][::-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        topics.append({
                            'id': topic_idx,
                            'words': top_words[:5],
                            'weight': float(topic[top_words_idx].sum())
                        })
                    
                    narrative['analysis']['topics'] = topics
                    print(f"✓ ({len(topics)} topics)")
                except Exception as e:
                    print(f"✗ ({str(e)[:50]})")
            
            # Extract problem formulations
            if 'conflict' in theme_counts or 'solution' in theme_counts:
                print("  Extracting problem formulations...", end=" ")
                problem_formulations = self._extract_problem_formulations(text, sentences)
                narrative['analysis']['problem_formulations'] = problem_formulations
                print(f"✓ ({len(problem_formulations)} problems)")
            
            # Generate summary statistics
            narrative['analysis']['summary'] = {
                'description': self._generate_narrative_description(narrative),
                'key_insights': self._extract_key_insights(narrative)
            }
            
            print("\n✓ Analysis complete!")
            return True
            
        except Exception as e:
            print(f"\n✗ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_entity_network(self, doc, entities):
        """Create network data from entities for visualization"""
        try:
            G = nx.Graph()
            node_id = 0
            node_mapping = {}
            
            # Add entity nodes
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity not in node_mapping:
                        G.add_node(node_id, label=entity, type=entity_type)
                        node_mapping[entity] = node_id
                        node_id += 1
            
            # Create edges between entities that appear in the same sentence
            for sent in doc.sents:
                sentence_entities = []
                for entity, entity_id in node_mapping.items():
                    if entity.lower() in sent.text.lower():
                        sentence_entities.append(entity_id)
                
                # Create edges between all pairs of entities in this sentence
                for i in range(len(sentence_entities)):
                    for j in range(i+1, len(sentence_entities)):
                        source = sentence_entities[i]
                        target = sentence_entities[j]
                        
                        # Add edge or increment weight if it already exists
                        if G.has_edge(source, target):
                            G[source][target]['weight'] += 1
                        else:
                            G.add_edge(source, target, weight=1)
            
            return {
                'nodes': list(G.nodes(data=True)),
                'edges': list(G.edges(data=True)),
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
        except Exception as e:
            print(f"Network creation error: {e}")
            return {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0}
    
    def _extract_problem_formulations(self, text, sentences):
        """Extract objective problem formulations from the narrative"""
        problem_formulations = []
        
        # Problem indicators
        problem_indicators = ['problem', 'issue', 'challenge', 'difficulty', 'obstacle', 
                            'conflict', 'struggle', 'concern', 'trouble', 'dilemma']
        
        # Solution indicators
        solution_indicators = ['solution', 'solve', 'resolve', 'answer', 'fix', 
                             'address', 'tackle', 'overcome', 'deal with', 'handle']
        
        # Goal indicators
        goal_indicators = ['goal', 'objective', 'aim', 'purpose', 'target', 
                          'want', 'need', 'desire', 'hope', 'wish']
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            
            # Check if sentence contains problem indicators
            if any(indicator in sent_lower for indicator in problem_indicators):
                problem = {
                    'sentence': sent,
                    'type': 'problem',
                    'context': sentences[max(0, i-1):min(len(sentences), i+2)]
                }
                
                # Look for associated solutions
                for j in range(max(0, i-2), min(len(sentences), i+3)):
                    if j != i and any(indicator in sentences[j].lower() for indicator in solution_indicators):
                        problem['potential_solution'] = sentences[j]
                        break
                
                problem_formulations.append(problem)
            
            # Check for goals
            elif any(indicator in sent_lower for indicator in goal_indicators):
                problem_formulations.append({
                    'sentence': sent,
                    'type': 'goal',
                    'context': sentences[max(0, i-1):min(len(sentences), i+2)]
                })
        
        return problem_formulations
    
    def _detect_emotional_arc(self, sentiment_values):
        """Detect the emotional arc pattern"""
        if len(sentiment_values) < 3:
            return "insufficient_data"
        
        try:
            # Divide into thirds
            third_size = len(sentiment_values) // 3
            first_third = np.mean(sentiment_values[:third_size])
            middle_third = np.mean(sentiment_values[third_size:2*third_size])
            last_third = np.mean(sentiment_values[2*third_size:])
            
            # Classify arc type
            if first_third < middle_third < last_third:
                return "Rising (Rags to Riches)"
            elif first_third > middle_third > last_third:
                return "Falling (Tragedy)"
            elif middle_third > first_third and middle_third > last_third:
                return "Rise-Fall (Icarus)"
            elif middle_third < first_third and middle_third < last_third:
                return "Fall-Rise (Man in a Hole)"
            elif abs(first_third - last_third) < 0.1:
                return "Stable (Steady State)"
            else:
                return "Complex Pattern"
        except:
            return "Unknown"
    
    def _generate_narrative_description(self, narrative):
        """Generate a brief description of the narrative"""
        analysis = narrative.get('analysis', {})
        stats = analysis.get('basic_stats', {})
        themes = analysis.get('themes', {})
        
        # Find dominant theme
        dominant_theme = max(themes.items(), key=lambda x: x[1])[0] if themes else "general"
        
        description = f"A {stats.get('word_count', 0):,}-word narrative "
        description += f"with {stats.get('sentence_count', 0)} sentences, "
        description += f"focusing primarily on {dominant_theme} themes."
        
        return description
    
    def _extract_key_insights(self, narrative):
        """Extract key insights from the analysis"""
        insights = []
        analysis = narrative.get('analysis', {})
        
        # Vocabulary richness insight
        richness = analysis.get('basic_stats', {}).get('vocabulary_richness', 0)
        if richness > 0.7:
            insights.append("Highly diverse vocabulary usage")
        elif richness < 0.3:
            insights.append("Repetitive vocabulary pattern")
        
        # Sentiment insight
        sentiment = analysis.get('sentiment', {})
        if sentiment:
            overall = sentiment.get('overall', {}).get('compound', 0)
            if overall > 0.5:
                insights.append("Strongly positive overall sentiment")
            elif overall < -0.5:
                insights.append("Strongly negative overall sentiment")
            
            arc = sentiment.get('emotional_arc', '')
            if arc and arc != 'Unknown':
                insights.append(f"Follows a {arc} emotional pattern")
        
        # Theme insights
        themes = analysis.get('themes', {})
        if themes:
            top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_themes:
                insights.append(f"Major themes: {', '.join([t[0] for t in top_themes])}")
        
        # Event-based insights
        events = analysis.get('events', {})
        if events and 'plotline' in events:
            plotline = events['plotline']
            if plotline.get('main_actors'):
                insights.append(f"Key characters: {', '.join(plotline['main_actors'][:3])}")
            if plotline.get('causal_relations'):
                insights.append(f"Contains {len(plotline['causal_relations'])} causal relationships")
        
        # Problem formulation insights
        problems = analysis.get('problem_formulations', {})
        if problems:
            problem_count = len([p for p in problems if p.get('type') == 'problem'])
            goal_count = len([p for p in problems if p.get('type') == 'goal'])
            if problem_count > 0:
                insights.append(f"Identifies {problem_count} problems/conflicts")
            if goal_count > 0:
                insights.append(f"Contains {goal_count} stated goals/objectives")
        
        return insights
    
    def create_visualizations(self, narrative_id, output_prefix=""):
        """Create comprehensive visualizations and save to files"""
        if narrative_id not in self.narratives:
            return None
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        
        if not analysis:
            print("No analysis data available for visualization")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16), facecolor='white')
        
        # Add main title
        fig.suptitle(f'Narrative Analysis: {narrative["title"]}', fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid for more visualizations
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Theme Distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        themes = analysis.get('themes', {})
        if themes:
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            theme_names = [t[0].capitalize() for t in sorted_themes]
            theme_values = [t[1] for t in sorted_themes]
            
            bars = ax1.bar(theme_names, theme_values, color='skyblue', edgecolor='navy')
            ax1.set_xlabel('Themes')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Theme Distribution')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No theme data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Theme Distribution')
        
        # 2. Top Words (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        word_freq = analysis.get('word_analysis', {}).get('top_words', [])
        if word_freq:
            words = [w[0] for w in word_freq[:10]]
            counts = [w[1] for w in word_freq[:10]]
            
            bars = ax2.barh(words[::-1], counts[::-1], color='lightgreen', edgecolor='darkgreen')
            ax2.set_xlabel('Frequency')
            ax2.set_title('Top 10 Most Frequent Words')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        str(counts[::-1][i]), ha='left', va='center')
        else:
            ax2.text(0.5, 0.5, 'No word frequency data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top 10 Most Frequent Words')
        
        # 3. Sentiment Progression (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        sentiment_data = analysis.get('sentiment', {})
        if sentiment_data and sentiment_data.get('progression'):
            sentiments = sentiment_data['progression']
            x = range(len(sentiments))
            
            # Create color map for points
            colors = ['red' if val < -0.1 else 'green' if val > 0.1 else 'yellow' for val in sentiments]
            
            ax3.plot(x, sentiments, 'b-', linewidth=2, alpha=0.7, label='Sentiment')
            ax3.scatter(x, sentiments, c=colors, s=30, alpha=0.8, edgecolors='black')
            
            # Add rolling average
            if len(sentiments) > 10:
                window = min(10, len(sentiments) // 5)
                rolling_avg = pd.Series(sentiments).rolling(window=window, center=True).mean()
                ax3.plot(x, rolling_avg, 'r--', linewidth=2, alpha=0.7, label=f'{window}-pt average')
            
            ax3.set_xlabel('Sentence Index')
            ax3.set_ylabel('Sentiment Score')
            ax3.set_title('Sentiment Progression')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.legend()
            
            # Add emotional arc annotation
            arc = sentiment_data.get('emotional_arc', 'Unknown')
            stats = sentiment_data.get('statistics', {})
            info_text = f"Arc: {arc}\nMean: {stats.get('mean', 0):.3f}\nStd: {stats.get('std', 0):.3f}"
            ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    verticalalignment='top', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No sentiment data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Sentiment Progression')
        
        # 4. Named Entities (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        entities = analysis.get('entities', {})
        if entities:
            # Count total entities by type
            entity_counts = {}
            for ent_type, ent_list in entities.items():
                if ent_list:
                    entity_counts[ent_type] = sum(count for _, count in ent_list)
            
            if entity_counts:
                # Sort by count and take top types
                sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:6]
                labels = [e[0] for e in sorted_entities]
                sizes = [e[1] for e in sorted_entities]
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                    colors=colors_pie, startangle=90)
                
                # Adjust text size
                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                
                ax4.set_title('Named Entity Distribution')
            else:
                ax4.text(0.5, 0.5, 'No entities found', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Named Entity Distribution')
        else:
            ax4.text(0.5, 0.5, 'Entity extraction not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Named Entity Distribution')
        
        # 5. Event Timeline (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        events_data = analysis.get('events', {})
        if events_data and 'timeline' in events_data:
            timeline = events_data['timeline']
            if timeline.get('timeline'):
                event_list = timeline['timeline'][:10]  # Show first 10 events
                y_pos = range(len(event_list))
                
                ax5.barh(y_pos, [1] * len(event_list), color='steelblue', alpha=0.6)
                
                for i, event in enumerate(event_list):
                    label = f"{event['subject']} {event['action']}"
                    if event['object']:
                        label += f" {event['object']}"
                    ax5.text(0.5, i, label[:40] + "..." if len(label) > 40 else label,
                            ha='center', va='center', fontsize=8)
                
                ax5.set_yticks([])
                ax5.set_xlim(0, 1)
                ax5.set_title('Event Timeline (First 10 Events)')
                ax5.set_xlabel('Narrative Progression')
            else:
                ax5.text(0.5, 0.5, 'No events extracted', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Event Timeline')
        else:
            ax5.text(0.5, 0.5, 'Event extraction not available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Event Timeline')
        
        # 6. Character Network (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        if events_data and 'timeline' in events_data:
            actors = timeline.get('actors', {})
            if actors:
                # Create a simple network visualization
                G = nx.Graph()
                
                # Add nodes for top actors
                top_actors = sorted(actors.items(), key=lambda x: x[1]['mentions'], reverse=True)[:7]
                for actor, data in top_actors:
                    G.add_node(actor, mentions=data['mentions'])
                
                # Add edges based on relationships
                for actor, data in top_actors:
                    for related, actions in data['relationships'].items():
                        if related in [a[0] for a in top_actors]:
                            G.add_edge(actor, related, weight=len(actions))
                
                # Draw network
                pos = nx.spring_layout(G, k=1, iterations=50)
                node_sizes = [G.nodes[node]['mentions'] * 100 for node in G.nodes()]
                
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                                     edgecolors='navy', linewidths=2, ax=ax6)
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax6)
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, ax=ax6)
                
                ax6.set_title('Character Relationship Network')
                ax6.axis('off')
            else:
                ax6.text(0.5, 0.5, 'No character data', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Character Network')
        else:
            ax6.text(0.5, 0.5, 'Character analysis not available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Character Network')
        
        # 7. Topic Distribution (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        topics = analysis.get('topics', [])
        if topics:
            topic_labels = [f"Topic {t['id']+1}" for t in topics]
            topic_weights = [t['weight'] for t in topics]
            
            bars = ax7.bar(topic_labels, topic_weights, color='coral', edgecolor='darkred')
            ax7.set_xlabel('Topics')
            ax7.set_ylabel('Weight')
            ax7.set_title('Topic Distribution (LDA)')
            
            # Add topic words as labels
            for i, (bar, topic) in enumerate(zip(bars, topics)):
                height = bar.get_height()
                words = ', '.join(topic['words'][:3])
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        words, ha='center', va='bottom', fontsize=7, rotation=15)
        else:
            ax7.text(0.5, 0.5, 'No topic modeling data', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Topic Distribution')
        
        # 8. Problem/Solution Analysis (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        problems = analysis.get('problem_formulations', [])
        if problems:
            problem_count = len([p for p in problems if p.get('type') == 'problem'])
            goal_count = len([p for p in problems if p.get('type') == 'goal'])
            solution_count = len([p for p in problems if p.get('potential_solution')])
            
            categories = ['Problems', 'Goals', 'Solutions']
            values = [problem_count, goal_count, solution_count]
            colors_bar = ['red', 'orange', 'green']
            
            bars = ax8.bar(categories, values, color=colors_bar, alpha=0.7)
            ax8.set_ylabel('Count')
            ax8.set_title('Problem Formulation Analysis')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom')
        else:
            ax8.text(0.5, 0.5, 'No problem formulations found', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Problem Formulation Analysis')
        
        # 9. Narrative Structure (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        if events_data and 'storyline' in events_data:
            storyline = events_data['storyline']
            structure = storyline.get('narrative_structure', {})
            if structure:
                phases = ['Exposition', 'Rising Action', 'Climax', 'Falling Action', 'Resolution']
                phase_lengths = [
                    len(structure.get('exposition', [])),
                    len(structure.get('rising_action', [])),
                    len(structure.get('climax', [])),
                    len(structure.get('falling_action', [])),
                    len(structure.get('resolution', []))
                ]
                
                # Create a stacked bar chart
                ax9.bar(range(len(phases)), phase_lengths, color=plt.cm.viridis(np.linspace(0, 1, len(phases))))
                ax9.set_xticks(range(len(phases)))
                ax9.set_xticklabels(phases, rotation=45, ha='right')
                ax9.set_ylabel('Number of Events')
                ax9.set_title('Narrative Structure')
                
                # Add value labels
                for i, (phase, length) in enumerate(zip(phases, phase_lengths)):
                    if length > 0:
                        ax9.text(i, length + 0.1, str(length), ha='center', va='bottom')
            else:
                ax9.text(0.5, 0.5, 'No narrative structure data', ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Narrative Structure')
        else:
            ax9.text(0.5, 0.5, 'Narrative structure not analyzed', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Narrative Structure')
        
        # 10. Summary Statistics (bottom row spanning all columns)
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')
        
        # Create comprehensive summary
        stats = analysis.get('basic_stats', {})
        summary_data = analysis.get('summary', {})
        
        # Create three-column layout for statistics
        left_text = [
            "**Document Statistics**",
            f"Total Words: {stats.get('word_count', 0):,}",
            f"Unique Words: {stats.get('unique_words', 0):,}",
            f"Sentences: {stats.get('sentence_count', 0):,}",
            f"Avg Sentence Length: {stats.get('avg_sentence_length', 0):.1f} words",
            f"Avg Word Length: {stats.get('avg_word_length', 0):.1f} chars",
            f"Vocabulary Richness: {stats.get('vocabulary_richness', 0):.3f}",
        ]
        
        middle_text = ["**Key Insights**"]
        insights = summary_data.get('key_insights', [])
        if insights:
            for insight in insights[:6]:
                middle_text.append(f"• {insight}")
        else:
            middle_text.append("• No specific insights available")
        
        right_text = ["**Advanced Analysis**"]
        if events_data:
            if 'events' in events_data and events_data['events'].get('count', 0) > 0:
                right_text.append(f"• Events extracted: {events_data['events']['count']}")
            if 'plotline' in events_data:
                plotline = events_data['plotline']
                if plotline.get('causal_relations'):
                    right_text.append(f"• Causal relations: {len(plotline['causal_relations'])}")
                if plotline.get('action_chains'):
                    right_text.append(f"• Action chains: {len(plotline['action_chains'])}")
        
        if 'network' in analysis and analysis['network'].get('node_count', 0) > 0:
            right_text.append(f"• Network nodes: {analysis['network']['node_count']}")
            right_text.append(f"• Network edges: {analysis['network']['edge_count']}")
        
        # Format and display text
        left_str = '\n'.join(left_text)
        middle_str = '\n'.join(middle_text)
        right_str = '\n'.join(right_text)
        
        # Use monospace font for better alignment
        ax10.text(0.05, 0.95, left_str, transform=ax10.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='gray', alpha=0.8))
        
        ax10.text(0.35, 0.95, middle_str, transform=ax10.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                         edgecolor='gray', alpha=0.8))
        
        ax10.text(0.68, 0.95, right_str, transform=ax10.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', 
                         edgecolor='gray', alpha=0.8))
        
        # Add description at bottom
        description = summary_data.get('description', '')
        if description:
            ax10.text(0.5, 0.05, description, transform=ax10.transAxes,
                    fontsize=11, ha='center', style='italic',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure
        safe_title = narrative['title'].replace(' ', '_')[:30]
        filename = f"{output_prefix}visualizations_{safe_title}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualizations to: {filename}")
        plt.close(fig)
        
        return filename
    
    def create_interactive_visualizations(self, narrative_id, output_prefix=""):
        """Create interactive visualizations using Plotly"""
        if not FEATURES['plotly']:
            return None
        
        if narrative_id not in self.narratives:
            return None
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        
        if not analysis:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sentiment Progression', 'Word Frequency', 
                          'Theme Distribution', 'Entity Network',
                          'Topic Weights', 'Narrative Timeline'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Interactive Sentiment Progression
        sentiment_data = analysis.get('sentiment', {})
        if sentiment_data and sentiment_data.get('progression'):
            sentiments = sentiment_data['progression']
            x = list(range(len(sentiments)))
            
            # Add sentiment trace
            fig.add_trace(
                go.Scatter(
                    x=x, y=sentiments,
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='royalblue', width=2),
                    marker=dict(
                        size=8,
                        color=sentiments,
                        colorscale='RdYlGn',
                        cmin=-1, cmax=1,
                        showscale=True
                    ),
                    hovertemplate='Sentence %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # 2. Word Frequency
        word_freq = analysis.get('word_analysis', {}).get('top_words', [])
        if word_freq:
            words = [w[0] for w in word_freq[:15]]
            counts = [w[1] for w in word_freq[:15]]
            
            fig.add_trace(
                go.Bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    name='Word Frequency',
                    marker_color='lightgreen',
                    hovertemplate='%{y}: %{x} occurrences<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Theme Distribution
        themes = analysis.get('themes', {})
        if themes:
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            theme_names = [t[0].capitalize() for t in sorted_themes]
            theme_values = [t[1] for t in sorted_themes]
            
            fig.add_trace(
                go.Bar(
                    x=theme_names,
                    y=theme_values,
                    name='Themes',
                    marker_color='skyblue',
                    hovertemplate='%{x}: %{y} mentions<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Entity Network (simplified scatter plot)
        network_data = analysis.get('network', {})
        if network_data and network_data.get('nodes'):
            # Create a simple 2D projection of the network
            nodes = network_data['nodes'][:20]  # Limit to 20 nodes
            
            # Simple circular layout
            n_nodes = len(nodes)
            if n_nodes > 0:
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                x_pos = np.cos(angles)
                y_pos = np.sin(angles)
                
                node_labels = [node[1]['label'] for node in nodes]
                node_types = [node[1]['type'] for node in nodes]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_pos,
                        y=y_pos,
                        mode='markers+text',
                        text=node_labels,
                        textposition="top center",
                        name='Entities',
                        marker=dict(
                            size=20,
                            color=[hash(t) % 10 for t in node_types],
                            colorscale='Viridis',
                            showscale=False
                        ),
                        hovertemplate='%{text}<br>Type: %{customdata}<extra></extra>',
                        customdata=node_types
                    ),
                    row=2, col=2
                )
        
        # 5. Topic Distribution
        topics = analysis.get('topics', [])
        if topics:
            topic_labels = [f"Topic {t['id']+1}" for t in topics]
            topic_weights = [t['weight'] for t in topics]
            topic_words = ['<br>'.join(t['words'][:3]) for t in topics]
            
            fig.add_trace(
                go.Bar(
                    x=topic_labels,
                    y=topic_weights,
                    name='Topics',
                    marker_color='coral',
                    text=topic_words,
                    hovertemplate='%{x}<br>Weight: %{y:.3f}<br>Keywords:<br>%{text}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Narrative Timeline
        events_data = analysis.get('events', {})
        if events_data and 'timeline' in events_data:
            timeline = events_data['timeline'].get('timeline', [])[:15]  # First 15 events
            
            if timeline:
                x_timeline = list(range(len(timeline)))
                y_timeline = [1] * len(timeline)
                event_texts = []
                
                for event in timeline:
                    text = f"{event['subject']} {event['action']}"
                    if event['object']:
                        text += f" {event['object']}"
                    event_texts.append(text)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_timeline,
                        y=y_timeline,
                        mode='markers+text',
                        text=event_texts,
                        textposition="top center",
                        name='Events',
                        marker=dict(size=15, color='steelblue'),
                        hovertemplate='Event %{x}: %{text}<extra></extra>'
                    ),
                    row=3, col=2
                )
                
                fig.update_yaxes(visible=False, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=f"Interactive Analysis: {narrative['title']}",
            showlegend=False,
            height=1200,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Sentence Index", row=1, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Theme", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Topic", row=3, col=1)
        fig.update_yaxes(title_text="Weight", row=3, col=1)
        fig.update_xaxes(title_text="Event Sequence", row=3, col=2)
        
        # Save interactive visualization
        safe_title = narrative['title'].replace(' ', '_')[:30]
        filename = f"{output_prefix}interactive_{safe_title}.html"
        fig.write_html(filename)
        print(f"  ✓ Saved interactive visualization to: {filename}")
        
        return filename
    
    def export_network_for_gephi(self, narrative_id, export_dir="network_exports"):
        """Export network data in formats compatible with Gephi"""
        if narrative_id not in self.narratives:
            return None
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        network_data = analysis.get('network', {})
        
        if not network_data or not network_data.get('nodes'):
            return None
        
        # Create export directory if it doesn't exist
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        filename_prefix = narrative['title'].replace(' ', '_')[:30]
        
        # Export nodes to CSV
        nodes_file = f"{export_dir}/{filename_prefix}_nodes.csv"
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'Label', 'Type'])
            for node in network_data['nodes']:
                writer.writerow([node[0], node[1]['label'], node[1]['type']])
        
        # Export edges to CSV
        edges_file = f"{export_dir}/{filename_prefix}_edges.csv"
        with open(edges_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight'])
            for edge in network_data['edges']:
                writer.writerow([edge[0], edge[1], edge[2]['weight']])
        
        # Create NetworkX graph for GEXF export
        G = nx.Graph()
        for node in network_data['nodes']:
            G.add_node(node[0], label=node[1]['label'], type=node[1]['type'])
        for edge in network_data['edges']:
            G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
        
        # Export as GEXF
        gexf_file = f"{export_dir}/{filename_prefix}_network.gexf"
        nx.write_gexf(G, gexf_file)
        
        # Export as GraphML
        graphml_file = f"{export_dir}/{filename_prefix}_network.graphml"
        nx.write_graphml(G, graphml_file)
        
        return {
            'nodes_csv': nodes_file,
            'edges_csv': edges_file,
            'gexf': gexf_file,
            'graphml': graphml_file
        }
    
    def export_all_data(self, narrative_id, export_dir="data_exports"):
        """Export all analysis data in various formats"""
        if narrative_id not in self.narratives:
            return None
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        
        if not analysis:
            return None
        
        # Create export directory if it doesn't exist
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        filename_prefix = narrative['title'].replace(' ', '_')[:30]
        exported_files = {}
        
        # Export word frequency data
        word_freq = analysis.get('word_analysis', {}).get('word_frequency', [])
        if word_freq:
            word_freq_file = f"{export_dir}/{filename_prefix}_word_frequency.csv"
            with open(word_freq_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Word', 'Frequency'])
                for word, freq in word_freq:
                    writer.writerow([word, freq])
            exported_files['word_frequency'] = word_freq_file
        
        # Export sentiment data
        sentiment_data = analysis.get('sentiment', {})
        if sentiment_data and sentiment_data.get('progression'):
            sentiment_file = f"{export_dir}/{filename_prefix}_sentiment.csv"
            with open(sentiment_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Sentence_Index', 'Sentiment_Score'])
                for i, score in enumerate(sentiment_data['progression']):
                    writer.writerow([i, score])
            exported_files['sentiment'] = sentiment_file
        
        # Export themes
        themes = analysis.get('themes', {})
        if themes:
            themes_file = f"{export_dir}/{filename_prefix}_themes.csv"
            with open(themes_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Theme', 'Count'])
                for theme, count in themes.items():
                    writer.writerow([theme, count])
            exported_files['themes'] = themes_file
        
        # Export events
        events_data = analysis.get('events', {})
        if events_data and 'timeline' in events_data:
            timeline = events_data['timeline'].get('timeline', [])
            if timeline:
                events_file = f"{export_dir}/{filename_prefix}_events.csv"
                with open(events_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Subject', 'Action', 'Object', 'Location', 'Time'])
                    for event in timeline:
                        writer.writerow([
                            event['subject'],
                            event['action'],
                            event.get('object', ''),
                            event.get('location', ''),
                            event.get('time', '')
                        ])
                exported_files['events'] = events_file
        
        # Export full analysis as JSON
        analysis_file = f"{export_dir}/{filename_prefix}_full_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        exported_files['full_analysis'] = analysis_file
        
        return exported_files
    
    def generate_text_report(self, narrative_id, custom_title="", custom_text=""):
        """Generate comprehensive text report"""
        if narrative_id not in self.narratives:
            return None
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        
        report = []
        report.append(f"# {custom_title or 'Enhanced Narrative Analysis Report'}\n")
        report.append(f"**Document:** {narrative['title']}")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**Source Type:** {narrative['metadata']['source'].upper()}")
        
        # Add file-specific metadata
        if 'shape' in narrative['metadata']:
            report.append(f"**Data Shape:** {narrative['metadata']['shape'][0]} rows × {narrative['metadata']['shape'][1]} columns")
        if 'pages' in narrative['metadata']:
            report.append(f"**Pages:** {narrative['metadata']['pages']}")
        if 'sheets' in narrative['metadata']:
            report.append(f"**Sheets:** {', '.join(narrative['metadata']['sheets'])}")
        
        report.append("")
        
        if custom_text:
            report.append(f"## Executive Summary\n{custom_text}\n")
        
        # Document Statistics
        stats = analysis.get('basic_stats', {})
        report.append("## Document Statistics")
        report.append(f"- Total Words: {stats.get('word_count', 'N/A'):,}")
        report.append(f"- Unique Words: {stats.get('unique_words', 'N/A'):,}")
        report.append(f"- Sentences: {stats.get('sentence_count', 'N/A'):,}")
        report.append(f"- Average Sentence Length: {stats.get('avg_sentence_length', 0):.1f} words")
        report.append(f"- Average Word Length: {stats.get('avg_word_length', 0):.1f} characters")
        report.append(f"- Vocabulary Richness: {stats.get('vocabulary_richness', 0):.3f}")
        report.append("")
        
        # Sentiment Analysis
        sentiment = analysis.get('sentiment', {})
        if sentiment:
            report.append("## Sentiment Analysis")
            
            overall = sentiment.get('overall', {})
            report.append(f"- Overall Sentiment Score: {overall.get('compound', 0):.3f}")
            report.append(f"- Positive: {overall.get('pos', 0):.2f} ({overall.get('pos', 0)*100:.1f}%)")
            report.append(f"- Negative: {overall.get('neg', 0):.2f} ({overall.get('neg', 0)*100:.1f}%)")
            report.append(f"- Neutral: {overall.get('neu', 0):.2f} ({overall.get('neu', 0)*100:.1f}%)")
            
            stats_sent = sentiment.get('statistics', {})
            if stats_sent:
                report.append(f"\n### Sentiment Statistics")
                report.append(f"- Mean Sentiment: {stats_sent.get('mean', 0):.3f}")
                report.append(f"- Standard Deviation: {stats_sent.get('std', 0):.3f}")
                report.append(f"- Min Sentiment: {stats_sent.get('min', 0):.3f}")
                report.append(f"- Max Sentiment: {stats_sent.get('max', 0):.3f}")
                report.append(f"- Emotional Arc: {sentiment.get('emotional_arc', 'Unknown')}")
            report.append("")
        
        # Theme Analysis
        themes = analysis.get('themes', {})
        if themes:
            report.append("## Theme Analysis")
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            total_theme_mentions = sum(themes.values())
            
            for theme, count in sorted_themes:
                percentage = (count / total_theme_mentions * 100) if total_theme_mentions > 0 else 0
                report.append(f"- **{theme.capitalize()}**: {count} mentions ({percentage:.1f}%)")
            report.append("")
        
        # Narrative Events Analysis
        events_data = analysis.get('events', {})
        if events_data:
            report.append("## Narrative Event Analysis")
            
            if 'events' in events_data and events_data['events'].get('count', 0) > 0:
                report.append(f"- Total Events Extracted: {events_data['events']['count']}")
            
            if 'plotline' in events_data:
                plotline = events_data['plotline']
                
                if plotline.get('main_actors'):
                    report.append("\n### Main Characters")
                    for actor in plotline['main_actors']:
                        report.append(f"- {actor}")
                
                if plotline.get('key_locations'):
                    report.append("\n### Key Locations")
                    for location in plotline['key_locations']:
                        report.append(f"- {location}")
                
                if plotline.get('causal_relations'):
                    report.append(f"\n### Causal Relations: {len(plotline['causal_relations'])} identified")
                
                if plotline.get('action_chains'):
                    report.append(f"\n### Action Chains: {len(plotline['action_chains'])} sequences")
            
            if 'storyline' in events_data:
                storyline = events_data['storyline']
                
                if storyline.get('themes'):
                    report.append("\n### Narrative Themes")
                    for theme in storyline['themes'][:5]:
                        report.append(f"- {theme}")
                
                if storyline.get('perspectives'):
                    report.append(f"\n### Perspectives: {len(storyline['perspectives'])} viewpoints identified")
            
            report.append("")
        
        # Problem Formulations
        problems = analysis.get('problem_formulations', [])
        if problems:
            report.append("## Problem Formulations")
            
            problem_count = len([p for p in problems if p.get('type') == 'problem'])
            goal_count = len([p for p in problems if p.get('type') == 'goal'])
            solution_count = len([p for p in problems if p.get('potential_solution')])
            
            report.append(f"- Problems Identified: {problem_count}")
            report.append(f"- Goals Stated: {goal_count}")
            report.append(f"- Potential Solutions: {solution_count}")
            
            if problem_count > 0:
                report.append("\n### Example Problems")
                for i, problem in enumerate([p for p in problems if p.get('type') == 'problem'][:3], 1):
                    report.append(f"{i}. {problem['sentence'][:100]}...")
            
            report.append("")
        
        # Top Words
        word_freq = analysis.get('word_analysis', {}).get('top_words', [])
        if word_freq:
            report.append("## Top 20 Most Frequent Words")
            for i, (word, count) in enumerate(word_freq[:20], 1):
                report.append(f"{i:2d}. {word}: {count}")
            report.append("")
        
        # Named Entities
        entities = analysis.get('entities', {})
        if entities:
            report.append("## Named Entities")
            
            # Sort entity types by total count
            entity_totals = {}
            for ent_type, ent_list in entities.items():
                if ent_list:
                    total = sum(count for _, count in ent_list)
                    entity_totals[ent_type] = total
            
            sorted_types = sorted(entity_totals.items(), key=lambda x: x[1], reverse=True)
            
            for ent_type, _ in sorted_types:
                ent_list = entities[ent_type]
                if ent_list:
                    report.append(f"\n### {ent_type}")
                    for name, count in ent_list[:5]:
                        report.append(f"- {name}: {count}")
            report.append("")
        
        # Topics
        topics = analysis.get('topics', [])
        if topics:
            report.append("## Discovered Topics")
            for i, topic in enumerate(topics, 1):
                words = ', '.join(topic['words'])
                report.append(f"\n### Topic {i}")
                report.append(f"Keywords: {words}")
                report.append(f"Weight: {topic['weight']:.3f}")
            report.append("")
        
        # Network Analysis
        network = analysis.get('network', {})
        if network and network.get('node_count', 0) > 0:
            report.append("## Entity Network Analysis")
            report.append(f"- Total Nodes: {network['node_count']}")
            report.append(f"- Total Edges: {network['edge_count']}")
            report.append(f"- Network Density: {(2 * network['edge_count']) / (network['node_count'] * (network['node_count'] - 1)) if network['node_count'] > 1 else 0:.3f}")
            report.append("")
        
        # Key Insights
        summary = analysis.get('summary', {})
        if summary:
            report.append("## Key Insights")
            
            description = summary.get('description', '')
            if description:
                report.append(f"\n{description}")
            
            insights = summary.get('key_insights', [])
            if insights:
                report.append("\n### Analysis Highlights")
                for insight in insights:
                    report.append(f"- {insight}")
            report.append("")
        
        # Metadata
        report.append("## File Metadata")
        for key, value in narrative['metadata'].items():
            if key not in ['filename', 'source']:  # Already shown at top
                if isinstance(value, list):
                    report.append(f"- {key}: {', '.join(str(v) for v in value)}")
                else:
                    report.append(f"- {key}: {value}")
        
        return '\n'.join(report)
    
    def generate_pdf_report(self, narrative_id, filename="enhanced_narrative_analysis_report.pdf", 
                           custom_title="", custom_text=""):
        """Generate comprehensive PDF report with enhanced visualizations"""
        if narrative_id not in self.narratives:
            return False
        
        narrative = self.narratives[narrative_id]
        analysis = narrative.get('analysis', {})
        
        print(f"\nGenerating enhanced PDF report: {filename}")
        
        try:
            with PdfPages(filename) as pdf:
                # Page 1: Title Page
                print("  Creating title page...")
                fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                
                # Title
                fig.text(0.5, 0.8, custom_title or 'Enhanced Narrative Analysis Report', 
                        ha='center', fontsize=26, fontweight='bold')
                fig.text(0.5, 0.73, narrative['title'], 
                        ha='center', fontsize=20, style='italic')
                
                # Metadata box
                stats = analysis.get('basic_stats', {})
                sentiment = analysis.get('sentiment', {})
                events_data = analysis.get('events', {})
                
                metadata_lines = [
                    f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                    "",
                    f"Source: {narrative['metadata']['source'].upper()} file",
                    f"Filename: {narrative['metadata']['filename']}",
                    "",
                    f"Words: {stats.get('word_count', 0):,}",
                    f"Sentences: {stats.get('sentence_count', 0):,}",
                    f"Unique Words: {stats.get('unique_words', 0):,}",
                    f"Vocabulary Richness: {stats.get('vocabulary_richness', 0):.3f}",
                ]
                
                if sentiment:
                    overall = sentiment.get('overall', {})
                    metadata_lines.extend([
                        "",
                        f"Overall Sentiment: {overall.get('compound', 0):.3f}",
                        f"Emotional Arc: {sentiment.get('emotional_arc', 'Unknown')}"
                    ])
                
                if events_data and 'events' in events_data:
                    metadata_lines.extend([
                        "",
                        f"Events Extracted: {events_data['events'].get('count', 0)}"
                    ])
                
                metadata_text = '\n'.join(metadata_lines)
                
                fig.text(0.5, 0.4, metadata_text, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', 
                                 alpha=0.3, edgecolor='navy'),
                        fontsize=12, linespacing=1.5)
                
                # Footer
                fig.text(0.5, 0.05, 'Enhanced Beulah11 - Advanced Narrative Analysis System', 
                        ha='center', fontsize=10, style='italic', alpha=0.7)
                
                plt.axis('off')
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)
                
                # Page 2: Executive Summary
                print("  Creating executive summary...")
                fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                
                fig.text(0.5, 0.95, 'Executive Summary', 
                        ha='center', fontsize=18, fontweight='bold')
                
                # Generate summary content
                if custom_text:
                    summary_content = custom_text
                else:
                    summary_content = self._generate_auto_summary(narrative, analysis)
                
                # Wrap and display text
                wrapped_lines = []
                for paragraph in summary_content.split('\n\n'):
                    wrapped = textwrap.fill(paragraph, width=75)
                    wrapped_lines.append(wrapped)
                
                full_text = '\n\n'.join(wrapped_lines)
                
                fig.text(0.1, 0.85, full_text, ha='left', va='top', fontsize=11,
                        wrap=True, transform=fig.transFigure)
                
                plt.axis('off')
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)
                
                # Page 3-4: Comprehensive Visualizations
                print("  Creating visualizations...")
                viz_filename = self.create_visualizations(narrative_id)
                if viz_filename and os.path.exists(viz_filename):
                    # Load the saved visualization
                    viz_img = PILImage.open(viz_filename)
                    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.imshow(viz_img)
                    ax.axis('off')
                    pdf.savefig(fig, facecolor='white', bbox_inches='tight')
                    plt.close(fig)
                
                # Page 5: Interactive Visualization Preview (if available)
                if FEATURES['plotly']:
                    print("  Creating interactive visualization preview...")
                    interactive_path = self.create_interactive_visualizations(narrative_id)
                    
                    if interactive_path:
                        # Create a preview page
                        fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                        fig.text(0.5, 0.95, 'Interactive Visualizations', 
                                ha='center', fontsize=18, fontweight='bold')
                        
                        fig.text(0.5, 0.5, 
                                f"Interactive visualizations have been saved to:\n\n{interactive_path}\n\n" +
                                "Open this file in a web browser for full interactivity.",
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow'))
                        
                        plt.axis('off')
                        pdf.savefig(fig, facecolor='white')
                        plt.close(fig)
                
                # Page 6: Detailed Analysis
                print("  Creating detailed analysis...")
                fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                
                fig.text(0.5, 0.95, 'Detailed Analysis', 
                        ha='center', fontsize=18, fontweight='bold')
                
                # Create detailed analysis content
                detail_lines = []
                
                # Themes
                themes = analysis.get('themes', {})
                if themes:
                    detail_lines.append("THEME ANALYSIS")
                    detail_lines.append("-" * 40)
                    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
                    for theme, count in sorted_themes[:5]:
                        detail_lines.append(f"• {theme.capitalize()}: {count} occurrences")
                    detail_lines.append("")
                
                # Event Analysis
                events_data = analysis.get('events', {})
                if events_data and 'plotline' in events_data:
                    plotline = events_data['plotline']
                    detail_lines.append("NARRATIVE STRUCTURE")
                    detail_lines.append("-" * 40)
                    
                    if plotline.get('main_actors'):
                        detail_lines.append("Main Characters:")
                        for actor in plotline['main_actors'][:5]:
                            detail_lines.append(f"• {actor}")
                    
                    if plotline.get('causal_relations'):
                        detail_lines.append(f"\nCausal Relations: {len(plotline['causal_relations'])}")
                    
                    detail_lines.append("")
                
                # Problem Formulations
                problems = analysis.get('problem_formulations', [])
                if problems:
                    detail_lines.append("PROBLEM ANALYSIS")
                    detail_lines.append("-" * 40)
                    problem_count = len([p for p in problems if p.get('type') == 'problem'])
                    goal_count = len([p for p in problems if p.get('type') == 'goal'])
                    detail_lines.append(f"Problems Identified: {problem_count}")
                    detail_lines.append(f"Goals Stated: {goal_count}")
                    detail_lines.append("")
                
                # Top entities
                entities = analysis.get('entities', {})
                if entities:
                    detail_lines.append("KEY ENTITIES")
                    detail_lines.append("-" * 40)
                    for ent_type, ent_list in list(entities.items())[:3]:
                        if ent_list:
                            detail_lines.append(f"\n{ent_type}:")
                            for name, count in ent_list[:3]:
                                detail_lines.append(f"• {name} ({count})")
                    detail_lines.append("")
                
                # Topics
                topics = analysis.get('topics', [])
                if topics:
                    detail_lines.append("DISCOVERED TOPICS")
                    detail_lines.append("-" * 40)
                    for i, topic in enumerate(topics[:3], 1):
                        words = ', '.join(topic['words'])
                        detail_lines.append(f"Topic {i}: {words}")
                    detail_lines.append("")
                
                # Key insights
                summary_data = analysis.get('summary', {})
                insights = summary_data.get('key_insights', [])
                if insights:
                    detail_lines.append("KEY INSIGHTS")
                    detail_lines.append("-" * 40)
                    for insight in insights:
                        detail_lines.append(f"• {insight}")
                
                # Display details
                detail_text = '\n'.join(detail_lines)
                fig.text(0.1, 0.85, detail_text, ha='left', va='top', fontsize=10,
                        fontfamily='monospace', transform=fig.transFigure)
                
                plt.axis('off')
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)
                
                # Page 7: Word Cloud (if available)
                if FEATURES['wordcloud']:
                    print("  Creating word cloud...")
                    cleaned_text = narrative.get('text', '')
                    
                    if cleaned_text:
                        wordcloud = WordCloud(
                            width=800, 
                            height=600, 
                            background_color='white',
                            stopwords=self.stop_words,
                            max_words=100,
                            contour_width=3, 
                            contour_color='steelblue'
                        ).generate(cleaned_text)
                        
                        fig = plt.figure(figsize=(8.5, 11), facecolor='white')
                        
                        fig.text(0.5, 0.95, 'Word Cloud Visualization', 
                                ha='center', fontsize=18, fontweight='bold')
                        
                        ax = fig.add_subplot(111)
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        
                        plt.tight_layout()
                        pdf.savefig(fig, facecolor='white')
                        plt.close(fig)
                
                # Set PDF metadata
                d = pdf.infodict()
                d['Title'] = custom_title or f'Enhanced Narrative Analysis: {narrative["title"]}'
                d['Author'] = 'Enhanced Beulah11 - Advanced Narrative Analysis System'
                d['Subject'] = 'Comprehensive Narrative Analysis Report'
                d['Keywords'] = 'NLP, Sentiment Analysis, Text Analysis, Narrative, Event Extraction'
                d['CreationDate'] = datetime.now()
            
            print(f"\n✓ Enhanced PDF report successfully saved as: {filename}")
            
            # Export additional data files
            print("\n  Exporting supplementary data files...")
            
            # Export network data for Gephi
            if analysis.get('network', {}).get('node_count', 0) > 0:
                network_files = self.export_network_for_gephi(narrative_id)
                if network_files:
                    print(f"  ✓ Network data exported for Gephi")
            
            # Export all data in CSV/JSON formats
            exported_files = self.export_all_data(narrative_id)
            if exported_files:
                print(f"  ✓ Analysis data exported to {len(exported_files)} files")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_auto_summary(self, narrative, analysis):
        """Generate automatic summary when custom text not provided"""
        stats = analysis.get('basic_stats', {})
        sentiment = analysis.get('sentiment', {})
        themes = analysis.get('themes', {})
        events_data = analysis.get('events', {})
        
        summary = f"""This enhanced narrative analysis examines a {stats.get('word_count', 0):,}-word document 
containing {stats.get('sentence_count', 0):,} sentences. The text demonstrates a vocabulary richness 
score of {stats.get('vocabulary_richness', 0):.3f}, indicating {self._interpret_vocabulary_richness(stats.get('vocabulary_richness', 0))} 
vocabulary usage with {stats.get('unique_words', 0):,} unique words.

The sentiment analysis reveals an overall {self._interpret_sentiment(sentiment.get('overall', {}).get('compound', 0))} tone 
with a compound sentiment score of {sentiment.get('overall', {}).get('compound', 0):.3f}. The emotional 
trajectory follows a {sentiment.get('emotional_arc', 'complex')} pattern, suggesting 
{self._interpret_emotional_arc(sentiment.get('emotional_arc', 'complex'))}."""

        if themes:
            summary += f"""

Thematic analysis identifies {len(themes)} major themes, with particular emphasis on 
{self._get_top_themes(themes)}. These themes appear throughout the narrative, creating 
a cohesive focus on {self._interpret_thematic_focus(themes)}."""

        if events_data and 'events' in events_data:
            event_count = events_data['events'].get('count', 0)
            if event_count > 0:
                summary += f"""

Advanced narrative analysis extracted {event_count} distinct events from the text, 
revealing complex relationships between characters and locations."""
            
            if 'plotline' in events_data:
                plotline = events_data['plotline']
                if plotline.get('main_actors'):
                    summary += f" The narrative centers around {len(plotline['main_actors'])} main characters: {', '.join(plotline['main_actors'][:3])}."
                if plotline.get('causal_relations'):
                    summary += f" {len(plotline['causal_relations'])} causal relationships were identified, indicating a complex plot structure."

        problems = analysis.get('problem_formulations', [])
        if problems:
            problem_count = len([p for p in problems if p.get('type') == 'problem'])
            goal_count = len([p for p in problems if p.get('type') == 'goal'])
            if problem_count > 0 or goal_count > 0:
                summary += f"""

The narrative contains {problem_count} identified problems/conflicts and {goal_count} stated goals, 
suggesting a problem-oriented narrative structure with potential solution pathways."""

        summary += f"""

The narrative structure exhibits {self._analyze_structural_characteristics(stats)} characteristics, 
with an average sentence length of {stats.get('avg_sentence_length', 0):.1f} words and 
average word length of {stats.get('avg_word_length', 0):.1f} characters."""
        
        # Add entity information if available
        entities = analysis.get('entities', {})
        if entities:
            total_entities = sum(sum(count for _, count in ent_list) for ent_list in entities.values())
            entity_types = len(entities)
            summary += f"""

The text contains {total_entities} named entity references across {entity_types} categories, 
indicating {self._interpret_entity_density(total_entities, stats.get('word_count', 1))} entity density."""

        network = analysis.get('network', {})
        if network and network.get('node_count', 0) > 0:
            summary += f"""

Entity network analysis reveals {network['node_count']} distinct entities connected by 
{network['edge_count']} relationships, forming a {self._interpret_network_complexity(network)} network structure."""
        
        return summary
    
    def _interpret_sentiment(self, score):
        """Interpret sentiment score"""
        if score >= 0.5:
            return "strongly positive"
        elif score >= 0.1:
            return "positive"
        elif score <= -0.5:
            return "strongly negative"
        elif score <= -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _interpret_vocabulary_richness(self, score):
        """Interpret vocabulary richness score"""
        if score >= 0.7:
            return "highly diverse and sophisticated"
        elif score >= 0.5:
            return "moderately diverse"
        elif score >= 0.3:
            return "somewhat repetitive"
        else:
            return "highly repetitive"
    
    def _interpret_emotional_arc(self, arc):
        """Interpret emotional arc pattern"""
        interpretations = {
            "Rising (Rags to Riches)": "a progressive improvement in emotional tone",
            "Falling (Tragedy)": "a gradual decline in emotional tone",
            "Rise-Fall (Icarus)": "initial optimism followed by disappointment",
            "Fall-Rise (Man in a Hole)": "challenges overcome through perseverance",
            "Stable (Steady State)": "consistent emotional tone throughout",
            "Complex Pattern": "multiple emotional shifts and nuanced development"
        }
        return interpretations.get(arc, "varied emotional development")
    
    def _get_top_themes(self, themes):
        """Get top themes as a string"""
        if not themes:
            return "no specific themes identified"
        
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        top_themes = [theme for theme, count in sorted_themes[:3] if count > 0]
        
        if not top_themes:
            return "no prominent themes"
        elif len(top_themes) == 1:
            return top_themes[0]
        elif len(top_themes) == 2:
            return f"{top_themes[0]} and {top_themes[1]}"
        else:
            return f"{', '.join(top_themes[:-1])}, and {top_themes[-1]}"
    
    def _interpret_thematic_focus(self, themes):
        """Interpret the overall thematic focus"""
        if not themes:
            return "general topics"
        
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_themes[0][0] if sorted_themes else "general"
        
        focus_map = {
            "relationships": "interpersonal dynamics and social connections",
            "emotions": "emotional experiences and psychological states",
            "time": "temporal elements and chronological progression",
            "places": "spatial settings and geographical contexts",
            "work": "professional and occupational matters",
            "nature": "environmental and natural elements",
            "technology": "technological and digital aspects",
            "health": "medical and wellness-related topics",
            "conflict": "problems, challenges, and conflicts",
            "solution": "problem-solving and resolution strategies"
        }
        
        return focus_map.get(dominant, f"{dominant}-related concepts")
    
    def _analyze_structural_characteristics(self, stats):
        """Analyze structural characteristics"""
        avg_sent_len = stats.get('avg_sentence_length', 0)
        
        if avg_sent_len < 10:
            return "concise and direct"
        elif avg_sent_len < 20:
            return "balanced and readable"
        elif avg_sent_len < 30:
            return "complex and detailed"
        else:
            return "highly complex and elaborate"
    
    def _interpret_entity_density(self, entity_count, word_count):
        """Interpret entity density"""
        if word_count == 0:
            return "unknown"
        
        density = entity_count / word_count
        
        if density > 0.05:
            return "high"
        elif density > 0.02:
            return "moderate"
        else:
            return "low"
    
    def _interpret_network_complexity(self, network):
        """Interpret network complexity"""
        if network['node_count'] == 0:
            return "empty"
        
        density = (2 * network['edge_count']) / (network['node_count'] * (network['node_count'] - 1)) if network['node_count'] > 1 else 0
        
        if density > 0.5:
            return "highly interconnected"
        elif density > 0.2:
            return "moderately connected"
        else:
            return "sparsely connected"


def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description='Enhanced Beulah11 - Advanced Narrative Analysis Tool',
        epilog='Supports: TXT, JSON, CSV, PDF, DOCX, Excel, VTT files'
    )
    
    # Input arguments
    parser.add_argument('files', nargs='+', help='Input files to analyze')
    
    # Analysis options
    parser.add_argument('--no-analysis', action='store_true', 
                       help='Skip analysis (only load files)')
    parser.add_argument('--analyze-only', nargs='+', 
                       help='Analyze only specific files by index or name')
    
    # Output options
    parser.add_argument('--output-dir', default='beulah11_output',
                       help='Output directory for all generated files (default: beulah11_output)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive visualization generation')
    parser.add_argument('--no-pdf', action='store_true',
                       help='Skip PDF report generation')
    parser.add_argument('--no-text', action='store_true',
                       help='Skip text report generation')
    parser.add_argument('--no-export', action='store_true',
                       help='Skip data export')
    
    # Report customization
    parser.add_argument('--report-title', default='',
                       help='Custom title for reports')
    parser.add_argument('--report-summary', default='',
                       help='Custom executive summary for reports')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Process all files without prompts')
    parser.add_argument('--prefix', default='',
                       help='Prefix for all output files')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Initialize NLTK
    print("=== Enhanced Beulah11 - Advanced Narrative Analysis Tool ===")
    print("Initializing... Please wait...")
    setup_nltk()
    
    # Show feature status
    print("\nAvailable features:")
    for feature, status in FEATURES.items():
        print(f"  {'✓' if status else '✗'} {feature}")
    
    # Check if critical features are missing
    if not any([FEATURES['pdf'], FEATURES['docx'], FEATURES['xlsx']]):
        print("\nNote: To enable additional file format support, install optional dependencies:")
        print("  pip install pdfplumber python-docx openpyxl")
    
    if not FEATURES['spacy']:
        print("\nNote: To enable advanced narrative analysis features, install spaCy:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
    
    if not FEATURES['plotly']:
        print("\nNote: To enable interactive visualizations, install Plotly:")
        print("  pip install plotly")
    
    if not FEATURES['wordcloud']:
        print("\nNote: To enable word cloud generation, install WordCloud:")
        print("  pip install wordcloud")
    
    # Create analyzer
    print("\n✓ Creating enhanced analyzer...")
    analyzer = EnhancedBeulahNarrativeAnalyzer()
    
    # Load files
    print(f"\nLoading {len(args.files)} file(s)...")
    loaded_narratives = []
    
    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            continue
        
        narrative_id = analyzer.load_file(file_path)
        if narrative_id:
            loaded_narratives.append((narrative_id, file_path))
            print(f"✓ [{i}] Loaded: {os.path.basename(file_path)}")
        else:
            print(f"✗ [{i}] Failed to load: {os.path.basename(file_path)}")
    
    if not loaded_narratives:
        print("\n✗ No files were loaded successfully.")
        return 1
    
    print(f"\n✓ Successfully loaded {len(loaded_narratives)} file(s)")
    
    # Skip analysis if requested
    if args.no_analysis:
        print("\nSkipping analysis as requested.")
        return 0
    
    # Determine which files to analyze
    if args.analyze_only:
        # Filter based on user selection
        selected_narratives = []
        for selector in args.analyze_only:
            if selector.isdigit():
                # Select by index
                idx = int(selector)
                if 0 <= idx < len(loaded_narratives):
                    selected_narratives.append(loaded_narratives[idx])
                else:
                    print(f"Warning: Index {idx} out of range")
            else:
                # Select by filename pattern
                for narrative_id, file_path in loaded_narratives:
                    if selector in os.path.basename(file_path):
                        selected_narratives.append((narrative_id, file_path))
        
        if not selected_narratives:
            print("\nNo files matched the selection criteria.")
            return 1
        
        narratives_to_analyze = selected_narratives
    else:
        narratives_to_analyze = loaded_narratives
    
    # Analyze narratives
    print(f"\nAnalyzing {len(narratives_to_analyze)} narrative(s)...")
    analyzed_count = 0
    
    for narrative_id, file_path in narratives_to_analyze:
        if analyzer.analyze_narrative(narrative_id):
            analyzed_count += 1
        else:
            print(f"✗ Analysis failed for: {os.path.basename(file_path)}")
    
    print(f"\n✓ Successfully analyzed {analyzed_count}/{len(narratives_to_analyze)} narratives")
    
    if analyzed_count == 0:
        print("\n✗ No narratives were analyzed successfully.")
        return 1
    
    # Generate outputs for each analyzed narrative
    print(f"\nGenerating outputs in: {args.output_dir}")
    
    for narrative_id, file_path in narratives_to_analyze:
        if narrative_id not in analyzer.narratives or 'analysis' not in analyzer.narratives[narrative_id]:
            continue
        
        narrative = analyzer.narratives[narrative_id]
        safe_title = narrative['title'].replace(' ', '_')[:30]
        file_prefix = os.path.join(args.output_dir, f"{args.prefix}{safe_title}_")
        
        print(f"\nProcessing: {narrative['title']}")
        
        # Generate visualizations
        if not args.no_viz:
            print("  Generating visualizations...")
            viz_file = analyzer.create_visualizations(narrative_id, file_prefix)
            if viz_file:
                print(f"    ✓ Saved: {viz_file}")
        
        # Generate interactive visualizations
        if not args.no_interactive and FEATURES['plotly']:
            print("  Generating interactive visualizations...")
            interactive_file = analyzer.create_interactive_visualizations(narrative_id, file_prefix)
            if interactive_file:
                print(f"    ✓ Saved: {interactive_file}")
        
        # Generate text report
        if not args.no_text:
            print("  Generating text report...")
            text_report = analyzer.generate_text_report(
                narrative_id,
                args.report_title,
                args.report_summary
            )
            if text_report:
                text_filename = f"{file_prefix}report.md"
                with open(text_filename, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                print(f"    ✓ Saved: {text_filename}")
        
        # Generate PDF report
        if not args.no_pdf:
            print("  Generating PDF report...")
            pdf_filename = f"{file_prefix}report.pdf"
            if analyzer.generate_pdf_report(
                narrative_id,
                pdf_filename,
                args.report_title,
                args.report_summary
            ):
                print(f"    ✓ Saved: {pdf_filename}")
        
        # Export data
        if not args.no_export:
            print("  Exporting data...")
            
            # Export network data for Gephi
            network_export_dir = os.path.join(args.output_dir, "network_exports")
            network_files = analyzer.export_network_for_gephi(narrative_id, network_export_dir)
            if network_files:
                print("    ✓ Network data exported:")
                for file_type, filepath in network_files.items():
                    print(f"      - {file_type}: {filepath}")
            
            # Export all analysis data
            data_export_dir = os.path.join(args.output_dir, "data_exports")
            exported_files = analyzer.export_all_data(narrative_id, data_export_dir)
            if exported_files:
                print("    ✓ Analysis data exported:")
                for data_type, filepath in exported_files.items():
                    print(f"      - {data_type}: {filepath}")
    
    # Generate summary report if multiple files
    if len(narratives_to_analyze) > 1:
        print("\nGenerating batch summary report...")
        summary_filename = os.path.join(args.output_dir, f"{args.prefix}batch_summary.txt")
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("Enhanced Beulah11 - Batch Processing Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(narratives_to_analyze)}\n")
            f.write(f"Successfully analyzed: {analyzed_count}\n\n")
            
            f.write("Processed Files:\n")
            f.write("-" * 30 + "\n")
            
            for narrative_id, file_path in narratives_to_analyze:
                if narrative_id in analyzer.narratives and 'analysis' in analyzer.narratives[narrative_id]:
                    narrative = analyzer.narratives[narrative_id]
                    analysis = narrative['analysis']
                    stats = analysis.get('basic_stats', {})
                    
                    f.write(f"\nFile: {os.path.basename(file_path)}\n")
                    f.write(f"  Words: {stats.get('word_count', 0):,}\n")
                    f.write(f"  Sentences: {stats.get('sentence_count', 0):,}\n")
                    f.write(f"  Vocabulary Richness: {stats.get('vocabulary_richness', 0):.3f}\n")
                    
                    sentiment = analysis.get('sentiment', {})
                    if sentiment:
                        overall = sentiment.get('overall', {})
                        f.write(f"  Overall Sentiment: {overall.get('compound', 0):.3f}\n")
                    
                    themes = analysis.get('themes', {})
                    if themes:
                        top_theme = max(themes.items(), key=lambda x: x[1])[0] if themes else "none"
                        f.write(f"  Dominant Theme: {top_theme}\n")
        
        print(f"✓ Batch summary saved: {summary_filename}")
    
    print("\n✓ All processing complete!")
    print(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
USAGE EXAMPLES:

1. Analyze a single file:
   python main.py document.txt

2. Analyze multiple files:
   python main.py file1.txt file2.pdf file3.csv

3. Analyze specific files from a batch:
   python main.py *.txt --analyze-only 0 2 4

4. Generate only PDF reports:
   python main.py documents/*.txt --no-viz --no-interactive --no-text --no-export

5. Custom output directory and prefix:
   python main.py data.csv --output-dir results --prefix analysis_

6. Batch processing with custom report:
   python main.py *.txt --batch --report-title "Quarterly Analysis" --report-summary "Q4 2024 narrative analysis results"

7. Quick analysis without visualizations:
   python main.py document.docx --no-viz --no-interactive --no-pdf

8. Export data only (no reports):
   python main.py data.json --no-viz --no-interactive --no-pdf --no-text

SUPPORTED FILE TYPES:
- Text files (.txt, .text)
- JSON files (.json)
- CSV files (.csv)
- PDF files (.pdf) - requires pdfplumber
- Word documents (.docx) - requires python-docx
- Excel files (.xlsx, .xls) - requires openpyxl
- VTT subtitle files (.vtt)

OUTPUT FILES:
- Static visualizations: *_visualizations.png
- Interactive visualizations: *_interactive.html
- Text reports: *_report.md
- PDF reports: *_report.pdf
- Network data: network_exports/*_nodes.csv, *_edges.csv, *.gexf, *.graphml
- Analysis data: data_exports/*_word_frequency.csv, *_sentiment.csv, *_themes.csv, *_events.csv, *_full_analysis.json

REQUIREMENTS.TXT:
# Core dependencies (required)
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
scikit-learn>=0.24.0
networkx>=2.6.0
Pillow>=8.3.0

# Optional dependencies (for additional features)
spacy>=3.0.0          # Advanced narrative analysis
pdfplumber>=0.5.0     # PDF support
python-docx>=0.8.0    # DOCX support
openpyxl>=3.0.0       # Excel support
plotly>=5.0.0         # Interactive visualizations
wordcloud>=1.8.0      # Word cloud generation

# To install all dependencies:
# pip install -r requirements.txt
# python -m spacy download en_core_web_sm
"""
