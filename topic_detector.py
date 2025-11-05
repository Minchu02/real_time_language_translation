# topic_detector.py
"""
Advanced Topic Detection using BERTopic for intelligent topic modeling
Detects actual topics/themes rather than just keywords
"""

import re
import numpy as np
from collections import Counter, deque
import logging
from typing import List, Dict, Tuple
import threading

logger = logging.getLogger(__name__)

# Try to import BERTopic, fallback to simple method if not available
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
    logger.info("✅ BERTopic is available for advanced topic detection")
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    logger.warning(f"❌ BERTopic not available: {e}. Using fallback topic detection.")

class AdvancedTopicDetector:
    def __init__(self, use_bertopic=True, history_size=20):
        self.use_bertopic = use_bertopic and BERTOPIC_AVAILABLE
        self.conversation_history = deque(maxlen=history_size)
        self.topic_model = None
        self.embedding_model = None
        self.lock = threading.Lock()
        self.is_fitted = False  # Track if model is fitted
        
        if self.use_bertopic:
            self._initialize_bertopic()
        else:
            logger.info("Using fallback topic detection")
        
        # Topic categories for classification
        self.topic_categories = {
            "Technology & IT": ["software", "code", "programming", "ai", "machine learning", "data", "cloud", "server", "database"],
            "Business & Strategy": ["business", "strategy", "market", "sales", "revenue", "growth", "planning", "management"],
            "Health & Medicine": ["health", "medical", "patient", "treatment", "disease", "hospital", "doctor", "medicine"],
            "Education & Learning": ["education", "learning", "teaching", "student", "school", "university", "course", "training"],
            "Science & Research": ["research", "study", "analysis", "scientific", "experiment", "data analysis", "findings"],
            "Marketing & Sales": ["marketing", "sales", "customer", "campaign", "advertising", "brand", "social media"],
            "Operations & Logistics": ["operations", "logistics", "supply chain", "production", "manufacturing", "delivery"],
            "Finance & Accounting": ["finance", "accounting", "budget", "revenue", "profit", "investment", "financial"],
            "Human Resources": ["hr", "human resources", "employee", "team", "hiring", "recruitment", "workforce"],
            "General Discussion": ["meeting", "discussion", "update", "review", "plan", "agenda"]
        }
        
        # Context patterns
        self.context_patterns = {
            "problem_solving": [r"problem with", r"issue with", r"challenge with", r"how to solve", r"how to fix"],
            "planning": [r"plan for", r"strategy for", r"roadmap for", r"schedule for", r"timeline for"],
            "review": [r"review of", r"analysis of", r"evaluation of", r"report on", r"update on"],
            "decision": [r"decision about", r"choose between", r"select from", r"option for"],
            "brainstorming": [r"ideas for", r"suggestions for", r"brainstorm", r"what if", r"possibilities for"]
        }

    def _initialize_bertopic(self):
        """Initialize BERTopic model with optimized settings"""
        try:
            # Use lightweight embedding model for speed
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Reduce dimensionality
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
            
            # Cluster with HDBSCAN
            hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
            
            # Vectorizer
            vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))  # Changed min_df to 1
            
            # Initialize BERTopic
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True,
                verbose=False
            )
            
            logger.info("✅ BERTopic model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize BERTopic: {e}")
            self.use_bertopic = False

    def add_to_history(self, text: str):
        """Add text to conversation history for context"""
        if text and text not in ["No speech detected", "ASR error"] and not text.startswith("ASR error"):
            self.conversation_history.append(text)

    def detect_context(self, text: str) -> str:
        """Detect the context/type of discussion"""
        text_lower = text.lower()
        
        for context_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return context_type
        
        return "general"

    def classify_topic_category(self, text: str) -> str:
        """Classify text into predefined topic categories"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.topic_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    # Bonus for multi-word matches
                    if ' ' in keyword:
                        score += 0.5
            scores[category] = score
        
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return "General Discussion"

    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract meaningful key phrases from text"""
        # Remove common phrases and clean text
        common_phrases = ["let's talk about", "we need to", "i want to", "we should", "going to"]
        clean_text = text.lower()
        for phrase in common_phrases:
            clean_text = clean_text.replace(phrase, "")
        
        # Extract meaningful phrases (2-3 words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)
        
        if len(words) < 2:
            return []
        
        # Create bi-grams and tri-grams
        phrases = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            phrases.append(bigram)
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(trigram)
        
        # Score phrases based on relevance
        phrase_scores = {}
        for phrase in phrases:
            # Simple scoring: longer phrases and those containing important words get higher scores
            score = len(phrase.split())  # Prefer longer phrases
            if any(keyword in phrase for keyword in ['analysis', 'strategy', 'plan', 'report', 'system']):
                score += 2
            phrase_scores[phrase] = score
        
        # Get top phrases
        top_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:max_phrases]
        return [phrase for phrase, score in top_phrases]

    def detect_topic_bertopic(self, text: str) -> str:
        """Use BERTopic for advanced topic detection"""
        try:
            with self.lock:
                # Add current text to history for context
                self.add_to_history(text)
                
                # Use recent history for better topic modeling
                recent_texts = list(self.conversation_history)
                
                if len(recent_texts) < 1:
                    return self.detect_topic_fallback(text)
                
                # Fit or update topic model
                if not self.is_fitted:
                    # Initial fit with all available texts
                    topics, probabilities = self.topic_model.fit_transform(recent_texts)
                    self.is_fitted = True
                    logger.info("✅ BERTopic model fitted successfully")
                else:
                    # Update with new texts - use partial_fit if available, otherwise transform
                    try:
                        topics, probabilities = self.topic_model.transform(recent_texts)
                    except Exception as transform_error:
                        logger.warning(f"Transform failed, refitting: {transform_error}")
                        topics, probabilities = self.topic_model.fit_transform(recent_texts)
                
                # Get topic info
                topic_info = self.topic_model.get_topic_info()
                
                if not topic_info.empty and len(topics) > 0:
                    # Get the most probable topic for the latest text
                    latest_topic = topics[-1] if len(topics) > 0 else -1
                    
                    if latest_topic != -1:
                        topic_words = self.topic_model.get_topic(latest_topic)
                        if topic_words and len(topic_words) > 0:
                            # Get top 3 words for the topic
                            top_words = [word for word, score in topic_words[:3]]
                            topic_name = " | ".join(top_words)
                            return f"{topic_name.title()}"
                
                # Fallback if no clear topic
                return self.classify_topic_category(text)
                
        except Exception as e:
            logger.error(f"BERTopic detection failed: {e}")
            return self.detect_topic_fallback(text)

    def detect_topic_fallback(self, text: str) -> str:
        """Fallback topic detection using rule-based approach"""
        if not text or text in ["No speech detected", "ASR error"] or text.startswith("ASR error"):
            return "General Discussion"
        
        # Add to history for context
        self.add_to_history(text)
        
        # Try to extract explicit topic mentions
        explicit_patterns = [
            r"(?:talk about|discuss|discussion about|focus on|regarding)\s+([^.,!?]+)",
            r"(?:main topic|primary focus|key subject|agenda item)\s+(?:is|are)\s+([^.,!?]+)",
            r"(?:today we(?:'ll| will)\s+(?:talk|discuss|cover)\s+([^.,!?]+))"
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, text.lower())
            if match:
                topic = match.group(1).strip()
                if len(topic.split()) <= 4:  # Avoid too long phrases
                    return topic.title()
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(text)
        if key_phrases:
            return " | ".join([phrase.title() for phrase in key_phrases[:2]])
        
        # Classify into categories
        category = self.classify_topic_category(text)
        context = self.detect_context(text)
        
        # Combine context and category
        context_map = {
            "problem_solving": "Problem Solving",
            "planning": "Planning Session",
            "review": "Review Meeting", 
            "decision": "Decision Making",
            "brainstorming": "Brainstorming"
        }
        
        context_str = context_map.get(context, "")
        if context_str:
            return f"{context_str} - {category}"
        
        return category

    def detect_topic(self, text: str, use_advanced: bool = True) -> str:
        """Main topic detection function"""
        if not text or text in ["No speech detected", "ASR error"] or text.startswith("ASR error"):
            return "General Discussion"
        
        if use_advanced and self.use_bertopic:
            return self.detect_topic_bertopic(text)
        else:
            return self.detect_topic_fallback(text)
    
    def get_topic_breakdown(self, text: str) -> Dict:
        """Get detailed topic analysis"""
        topic = self.detect_topic(text)
        category = self.classify_topic_category(text)
        context = self.detect_context(text)
        key_phrases = self.extract_key_phrases(text)
        
        return {
            "main_topic": topic,
            "category": category,
            "context": context,
            "key_phrases": key_phrases,
            "method_used": "BERTopic" if (self.use_bertopic and use_advanced) else "Rule-Based"
        }

# Global instance
_topic_detector = AdvancedTopicDetector()

def detect_topic(text: str, use_advanced: bool = True) -> str:
    """Main function to detect topic - uses BERTopic if available"""
    return _topic_detector.detect_topic(text, use_advanced)

def get_topic_breakdown(text: str) -> Dict:
    """Get detailed topic analysis"""
    return _topic_detector.get_topic_breakdown(text)

def reset_topic_history():
    """Reset conversation history"""
    _topic_detector.conversation_history.clear()
    _topic_detector.is_fitted = False  # Reset fitted state

def is_bertopic_available() -> bool:
    """Check if BERTopic is available"""
    return BERTOPIC_AVAILABLE