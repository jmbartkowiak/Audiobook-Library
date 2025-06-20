"""
confidence_scoring.py
Version: 2.3.0
Last Update: 2025-06-20

This module provides comprehensive confidence scoring for audiobook organization, combining both grouping
and enrichment confidence calculations. It implements multiple fuzzy matching algorithms and a weighted
scoring system to determine if files should be grouped together and whether they need metadata enrichment.

Key Components:
1. Grouping Confidence: Determines if files should be grouped as chapters of the same book.
2. Enrichment Confidence: Decides when to use LLM-based enrichment vs. existing metadata.
3. Similarity Algorithms: Implements multiple string matching algorithms for robust comparisons.

Directory:
- GroupingResult:line 30 - Data class for grouping confidence results
- GroupingConfidence:line 40 - Main class for chapter grouping confidence
- _weighted_dl:line 220 - Weighted Damerau-Levenshtein distance
- _jw_ngram:line 250 - Jaro-Winkler with n-gram penalty
- _phonetic:line 280 - Phonetic similarity using Soundex
- _trio:line 300 - Combined similarity metric
- score_combined:line 320 - Main enrichment confidence scorer
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import os
import re
import jellyfish
import textdistance
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from rapidfuzz.distance import JaroWinkler

# Type aliases for better readability
SimilarityScore = float
ConfidenceScore = float

# Constants
_VOWELS: Set[str] = set("aeiou")

# ------------------- Data Classes -------------------

@dataclass
class GroupingResult:
    """
    Container for grouping confidence results with detailed scoring.
    
    Attributes:
        confidence: Float between 0.0 and 1.0 indicating grouping confidence
        is_chaptered: Boolean indicating if files should be grouped as chapters
        details: Dictionary with detailed scoring information
    """
    confidence: float
    is_chaptered: bool
    details: Dict[str, Any]

# ------------------- Similarity Algorithms -------------------

def _weighted_dl(a: str, b: str) -> SimilarityScore:
    """
    Calculate weighted Damerau-Levenshtein similarity between two strings.
    Vowel substitutions are penalized less than consonant substitutions.
    
    Usage: Used in _trio() for handling OCR errors and minor typos.
    
    Args:
        a: First input string
        b: Second input string
        
    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = 0 if a[i - 1] == b[j - 1] else (0.2 if a[i - 1] in _VOWELS else 1)
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + sub_cost
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 0.5)
    return 1 - dp[m][n] / max(m, n) if max(m, n) > 0 else 1.0

@lru_cache(maxsize=256)
def _jw_ngram(a: str, b: str, n: int = 2) -> SimilarityScore:
    """
    Calculate Jaro-Winkler similarity with n-gram penalty.
    
    Usage: Used in _trio() for matching names and titles with minor variations.
    
    Args:
        a: First input string
        b: Second input string
        n: N-gram size for penalty calculation
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    base = JaroWinkler.normalized_similarity(a, b)
    grams = lambda s: {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}
    decay = 0.1 * len((grams(a) | grams(b)) - (grams(a) & grams(b)))
    return max(base - decay, 0)

def _phonetic(a: str, b: str) -> SimilarityScore:
    """
    Calculate phonetic similarity using Soundex and Damerau-Levenshtein distance.
    
    Usage: Used in _trio() for matching names with different spellings but similar pronunciation.
    
    Args:
        a: First input string
        b: Second input string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    penalty = 0.5 if jellyfish.soundex(a) == jellyfish.soundex(b) else 1.0
    dl = textdistance.damerau_levenshtein.normalized_distance(a, b)
    return 1 - penalty * dl

@lru_cache(maxsize=256)
def _trio(a: str, b: str) -> SimilarityScore:
    """
    Combine three similarity metrics for robust fuzzy matching.
    
    Usage: Used as a general-purpose string similarity function throughout the application.
    
    Args:
        a: First input string
        b: Second input string
        
    Returns:
        Combined similarity score (average of three metrics)
    """
    return (_weighted_dl(a, b) + _jw_ngram(a, b) + _phonetic(a, b)) / 3

# ------------------- Grouping Confidence -------------------

class GroupingConfidence:
    """
    Calculates confidence scores for grouping files as chapters of the same book.
    Uses a weighted scoring system with penalties for negative indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration from settings.yaml.
        
        Args:
            config: Configuration dictionary with weights, penalties, and thresholds
        """
        self.config = config.get('grouping', {})
        self.weights = self.config.get('weights', {
            'duration_consistency': 0.4,
            'naming_pattern': 0.5,
            'title_similarity': 0.1
        })
        self.penalties = self.config.get('penalties', {
            'mixed_formats': 0.8,
            'low_title_similarity': 0.4,
            'suspicious_file_count': 0.3
        })
        self.thresholds = self.config.get('thresholds', {
            'min_title_similarity': 0.4,
            'max_files_before_penalty': 30,
            'max_total_penalty': 0.9
        })
    
    def calculate(self, files: List[Dict]) -> GroupingResult:
        """
        Calculate grouping confidence for a list of files.
        
        Args:
            files: List of file metadata dicts with keys:
                - path: Full path to the file
                - format: File format/extension (e.g., 'mp3', 'm4b')
                - duration_sec: Duration in seconds (optional)
                - title: Extracted title (optional)
                - size_mb: File size in MB (optional)
                
        Returns:
            GroupingResult with confidence, is_chaptered flag, and detailed scoring
        """
        if len(files) == 1:
            return GroupingResult(
                confidence=0.5,
                is_chaptered=False,
                details={"reason": "Single file, neutral confidence"}
            )
            
        scores = {}
        details = {}
        
        # 1. Duration Consistency
        durations = [f.get('duration_sec') for f in files if f.get('duration_sec') is not None]
        if durations and len(durations) > 1:
            avg_duration = sum(durations) / len(durations)
            duration_std = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            duration_cv = duration_std / avg_duration if avg_duration > 0 else 1
            scores['duration_consistency'] = 1 / (1 + duration_cv)
            details['duration'] = f"Duration CV: {duration_cv:.2f}"
        else:
            scores['duration_consistency'] = 0.5
            details['duration'] = "No duration data, using neutral score"

        # 2. Naming Pattern
        # Strip leading numbers and symbols to find a more meaningful common prefix
        filenames = [Path(f['path']).stem for f in files]
        cleaned_filenames = [re.sub(r'^\d+[\s._-]*', '', fn) for fn in filenames]
        base_name = os.path.commonprefix(cleaned_filenames).strip(' _-')
        
        # The score is based on the length of the common part relative to the average length
        avg_len = mean(len(fn) for fn in cleaned_filenames) if cleaned_filenames else 1
        pattern_score = len(base_name) / max(avg_len, 1)
        scores['naming_pattern'] = pattern_score
        details['naming'] = f"Common prefix: '{base_name[:30]}...'"

        # 3. Title Similarity
        titles = [f.get('title', '') for f in files if f.get('title')]
        if len(titles) >= 2:
            similarities = []
            for i in range(len(titles)):
                for j in range(i+1, len(titles)):
                    similarities.append(_trio(titles[i].lower(), titles[j].lower()))
            scores['title_similarity'] = mean(similarities) if similarities else 0
            details['title_sim'] = f"Avg similarity: {scores['title_similarity']:.2f}"
        else:
            scores['title_similarity'] = 0.5
            details['title_sim'] = "Insufficient titles for comparison"

        # Calculate base score
        base_score = sum(scores.get(f, 0) * w for f, w in self.weights.items())
        details['base_score'] = round(base_score, 3)
        
        # Apply penalties
        penalty = 0.0
        penalty_reasons = []
        
        # 1. Mixed formats (strong penalty)
        formats = {f.get('format', '').lower() for f in files if f.get('format')}
        if len(formats) > 1:
            penalty += self.penalties['mixed_formats']
            penalty_reasons.append(f"Mixed formats: {', '.join(formats)}")
        
        # 2. Low title similarity
        # Suppress this penalty if a strong naming pattern is detected, as chapters are expected to have different titles.
        is_strong_pattern = scores.get('naming_pattern', 0.0) > 0.9
        if not is_strong_pattern and scores.get('title_similarity', 1.0) < self.thresholds['min_title_similarity']:
            penalty += self.penalties['low_title_similarity']
            penalty_reasons.append(f"Low title similarity: {scores.get('title_similarity', 1.0):.2f}")
            
        # 3. Suspicious file count
        if len(files) > self.thresholds['max_files_before_penalty']:
            penalty += self.penalties['suspicious_file_count']
            penalty_reasons.append(f"High file count ({len(files)})")
        
        # Cap total penalty and calculate final score
        max_penalty = self.thresholds['max_total_penalty']
        penalty = min(penalty, max_penalty)
        final_score = base_score * (1 - penalty)
        
        # Prepare detailed reasons
        details.update({
            'penalty': round(penalty, 3),
            'penalty_reasons': penalty_reasons,
            'final_score': round(final_score, 3),
            'is_chaptered': final_score >= self.config.get('min_confidence', 0.6)
        })
        
        return GroupingResult(
            confidence=max(0.01, min(0.99, final_score)),
            is_chaptered=details['is_chaptered'],
            details=details
        )

# ------------------- Enrichment Confidence -------------------

class EnrichmentConfidence:
    """
    Calculates confidence scores for metadata enrichment decisions.
    Uses multiple fuzzy matching algorithms to determine if enrichment is needed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration from settings.yaml.
        
        Args:
            config: Configuration dictionary with weights and thresholds
        """
        self.config = config.get('enrichment', {})
        self.weights = self.config.get('weights', {
            'weighted_dl': 0.4,
            'jw_ngram': 0.4,
            'phonetic': 0.2
        })
        self.thresholds = self.config.get('thresholds', {
            'high_confidence': 0.85,
            'medium_confidence': 0.7,
            'min_length': 3,
            'max_length_ratio': 0.5
        })
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text for comparison.
        
        Args:
            text: Input string to preprocess
            
        Returns:
            Preprocessed string with standardized formatting
        """
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except spaces and hyphens
        return re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    def _get_length_penalty(self, a: str, b: str) -> float:
        """
        Calculate length-based penalty for very different length strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Penalty factor between 0.0 and 1.0
        """
        len_a, len_b = len(a), len(b)
        if not len_a or not len_b:
            return 0.0
            
        ratio = min(len_a, len_b) / max(len_a, len_b)
        if ratio < self.thresholds['max_length_ratio']:
            return 0.0
        return ratio ** 1.5
    
    def _get_combined_score(self, a: str, b: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined similarity score using multiple algorithms.
        
        Args:
            a: First string to compare
            b: Second string to compare
            
        Returns:
            Tuple of (combined_score, individual_scores)
        """
        if not a or not b:
            return 0.0, {}
            
        # Preprocess inputs
        a_clean = self._preprocess(a)
        b_clean = self._preprocess(b)
        
        # Calculate individual scores
        scores = {
            'weighted_dl': _weighted_dl(a_clean, b_clean),
            'jw_ngram': _jw_ngram(a_clean, b_clean),
            'phonetic': _phonetic(a_clean, b_clean)
        }
        
        # Apply length penalty
        length_penalty = self._get_length_penalty(a_clean, b_clean)
        if length_penalty < 1.0:
            for k in scores:
                scores[k] *= length_penalty
        
        # Calculate weighted average
        combined = sum(scores[k] * self.weights[k] for k in scores)
        
        return combined, scores
    
    def score_combined(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enrichment confidence for all fields.
        
        Args:
            existing: Dictionary of existing metadata
            new: Dictionary of new metadata to potentially enrich with
            
        Returns:
            Dictionary with confidence scores and details for each field
        """
        if not existing or not new:
            return {}
            
        results = {}
        all_fields = set(existing.keys()) | set(new.keys())
        
        for field in all_fields:
            existing_val = existing.get(field, '')
            new_val = new.get(field, '')
            
            # Skip if both values are empty or too short
            if not existing_val and not new_val:
                continue
                
            if len(str(existing_val)) < self.thresholds['min_length'] and \
               len(str(new_val)) < self.thresholds['min_length']:
                continue
            
            # Calculate scores
            combined_score, individual_scores = self._get_combined_score(
                str(existing_val), 
                str(new_val)
            )
            
            # Determine confidence level
            if combined_score >= self.thresholds['high_confidence']:
                confidence = 'high'
            elif combined_score >= self.thresholds['medium_confidence']:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            results[field] = {
                'confidence': confidence,
                'score': round(combined_score, 3),
                'existing': existing_val,
                'new': new_val,
                'details': {k: round(v, 3) for k, v in individual_scores.items()}
            }
        
        return results
    
    def calculate_confidence(self, title: str, author: str, folder: str, filename: str,
                           ol_ratio: float, file_count: int, mean_mb: float, total_mb: float,
                           bitrate_sigma: float, blacklist_hit: bool) -> float:
        """
        Calculate enrichment confidence score based on various metadata attributes.
        
        Args:
            title: Book title
            author: Author name
            folder: Parent folder name
            filename: Original filename
            ol_ratio: OpenLibrary match ratio (0.0 to 1.0)
            file_count: Number of files in group
            mean_mb: Average file size in MB
            total_mb: Total size of all files in MB
            bitrate_sigma: Standard deviation of bitrates (normalized)
            blacklist_hit: Whether any blacklisted terms were found
            
        Returns:
            Confidence score between 0.0 (low) and 1.0 (high)
        """
        # Initialize base score components
        scores = {
            'title_quality': 0.0,
            'author_quality': 0.0,
            'folder_match': 0.0,
            'ol_ratio': max(0.0, min(1.0, ol_ratio)),
            'file_count': min(1.0, file_count / 10.0),  # Cap at 10 files
            'size_consistency': 0.0,
            'bitrate_consistency': 1.0 - min(1.0, bitrate_sigma * 2),  # Convert sigma to penalty
            'blacklist_penalty': 0.5 if blacklist_hit else 1.0
        }
        
        # Calculate title quality score
        title = (title or '').strip()
        if len(title) >= 3:
            # Check for common patterns in bad titles
            bad_patterns = [
                r'^\d+$',  # Just numbers
                r'^[a-z0-9]{16,}$',  # Likely hash or ID
                r'^[a-z0-9_\-.]{20,}$',  # Long filename-like
                r'^track\s*\d+',  # Track number
                r'^disc\s*\d+',  # Disc number
            ]
            
            if any(re.search(p, title.lower()) for p in bad_patterns):
                scores['title_quality'] = 0.1
            else:
                # Score based on title length and content
                title_words = len(title.split())
                scores['title_quality'] = min(1.0, title_words / 5.0)  # 5+ words = max score
        
        # Calculate author quality score
        author = (author or '').strip()
        if author and author.lower() not in ('unknown', 'various', 'various artists', 'n/a'):
            author_parts = [p for p in re.split(r'[\s\-&,]', author) if len(p) > 1]
            if len(author_parts) >= 2:  # First and last name
                scores['author_quality'] = 1.0
            elif author_parts:
                scores['author_quality'] = 0.5
        
        # Check if folder name matches title/author
        folder = (folder or '').lower()
        if folder and title:
            title_lower = title.lower()
            if folder in title_lower or title_lower in folder:
                scores['folder_match'] = 1.0
            elif any(word in folder for word in title_lower.split() if len(word) > 3):
                scores['folder_match'] = 0.5
        
        # Calculate size consistency (penalize if mean size is too small for audiobooks)
        if mean_mb > 0:
            if mean_mb < 5:  # Very small files
                scores['size_consistency'] = 0.1
            elif mean_mb < 20:  # Small files
                scores['size_consistency'] = 0.3
            elif mean_mb > 500:  # Very large files
                scores['size_consistency'] = 0.7
            else:
                scores['size_consistency'] = 1.0
        
        # Weighted average of all components
        weights = {
            'title_quality': 0.25,
            'author_quality': 0.25,
            'folder_match': 0.15,
            'ol_ratio': 0.15,
            'file_count': 0.05,
            'size_consistency': 0.05,
            'bitrate_consistency': 0.05
        }
        
        # Calculate weighted score
        weighted_sum = sum(scores[comp] * weight for comp, weight in weights.items())
        total_weight = sum(weights.values())
        final_score = (weighted_sum / total_weight) * scores['blacklist_penalty']
        
        return max(0.0, min(1.0, final_score))

    def should_enrich(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if enrichment should be performed based on confidence scores.
        
        Args:
            existing: Dictionary of existing metadata
            new: Dictionary of new metadata to potentially enrich with
            
        Returns:
            Dictionary with enrichment decisions and confidence scores
        """
        if not existing:
            return {
                'should_enrich': True,
                'confidence': 'high',
                'reason': 'No existing metadata',
                'fields': {k: 'new' for k in new}
            }
            
        if not new:
            return {
                'should_enrich': False,
                'confidence': 'high',
                'reason': 'No new metadata',
                'fields': {}
            }
            
        scores = self.score_combined(existing, new)
        decisions = {}
        
        for field, score_data in scores.items():
            existing_val = existing.get(field, '')
            new_val = new.get(field, '')
            
            if not existing_val and new_val:
                decisions[field] = 'add_new'
            elif not new_val and existing_val:
                decisions[field] = 'keep_existing'
            elif score_data['confidence'] == 'high':
                decisions[field] = 'enrich_high_confidence'
            elif score_data['confidence'] == 'medium':
                decisions[field] = 'enrich_medium_confidence'
            else:
                decisions[field] = 'keep_existing_low_confidence'
        
        # Calculate overall confidence (lowest confidence of all fields)
        conf_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_confidence = min(
            (s['confidence'] for s in scores.values() if s['existing'] and s['new']),
            default='high',
            key=lambda x: conf_levels[x]
        )
        
        return {
            'should_enrich': any(v.startswith('enrich_') for v in decisions.values()),
            'confidence': min_confidence,
            'fields': decisions,
            'scores': scores
        }

# ------------------- Module Functions -------------------

def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration for both grouping and enrichment confidence scoring.
    
    Returns:
        Dictionary with default configuration that can be overridden by user settings
    """
    return {
        'grouping': {
            'weights': {
                'duration_consistency': 0.25,
                'naming_pattern': 0.35,
                'title_similarity': 0.40
            },
            'penalties': {
                'mixed_formats': 0.8,
                'low_title_similarity': 0.4,
                'suspicious_file_count': 0.3
            },
            'thresholds': {
                'min_title_similarity': 0.4,
                'max_files_before_penalty': 30,
                'max_total_penalty': 0.9,
                'min_confidence': 0.6
            }
        },
        'enrichment': {
            'weights': {
                'weighted_dl': 0.4,
                'jw_ngram': 0.4,
                'phonetic': 0.2
            },
            'thresholds': {
                'high_confidence': 0.85,
                'medium_confidence': 0.7,
                'min_length': 3,
                'max_length_ratio': 0.5
            }
        }
    }

def initialize_confidence_scorers(config: Optional[Dict[str, Any]] = None) -> Tuple[GroupingConfidence, EnrichmentConfidence]:
    """
    Initialize both confidence scorers with the given or default configuration.
    
    Args:
        config: Optional configuration dictionary. If None, uses default config.
        
    Returns:
        Tuple of (grouping_scorer, enrichment_scorer)
    """
    if config is None:
        config = get_default_config()
    return GroupingConfidence(config), EnrichmentConfidence(config)

# Example usage
if __name__ == "__main__":
    # Initialize scorers with default config
    grouping_scorer, enrichment_scorer = initialize_confidence_scorers()
    
    # Example 1: Check if files should be grouped
    files = [
        {
            'path': '/books/book1_ch1.mp3', 
            'title': 'The Great Adventure - Chapter 1',
            'duration_sec': 1800, 
            'format': 'mp3'
        },
        {
            'path': '/books/book1_ch2.mp3', 
            'title': 'The Great Adventure - Chapter 2', 
            'duration_sec': 1750, 
            'format': 'mp3'
        }
    ]
    grouping_result = grouping_scorer.calculate(files)
    print(f"Grouping confidence: {grouping_result.confidence:.2f}")
    print(f"Should group: {grouping_result.is_chaptered}")
    
    # Example 2: Check if metadata should be enriched
    existing_metadata = {
        'title': 'The Great Adventure',
        'author': 'John Smith',
        'narrator': 'Jane Doe'
    }
    new_metadata = {
        'title': 'The Great Adventure: Special Edition',
        'author': 'John A. Smith',
        'year': '2023',
        'genre': 'Adventure'
    }
    enrich_result = enrichment_scorer.should_enrich(existing_metadata, new_metadata)
    print(f"\nEnrichment decision: {enrich_result['should_enrich']}")
    print(f"Fields to update: {[k for k, v in enrich_result['fields'].items() if v.startswith('enrich_')]}")
