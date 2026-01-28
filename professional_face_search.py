"""
Professional Face Search System using DeepFace
===============================================

A robust, production-style script that detects which images contain 
the same person as in a reference photo.

Features:
- Extract embeddings once for better performance
- Support multiple faces per image
- Configurable similarity threshold
- Caching of embeddings
- Multiprocessing support
- Comprehensive error handling
- Professional code structure

Author: Claude Code
Version: 2.0.0
"""

import os
import json
import pickle
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import logging

import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FaceEmbedding:
    """Data class for storing face embedding information"""
    image_path: str
    face_index: int
    embedding: List[float]
    face_location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float
    model_used: str
    created_at: float

@dataclass
class SearchResult:
    """Data class for search results"""
    image_path: str
    face_index: int
    similarity_score: float
    distance: float
    face_location: Tuple[int, int, int, int]
    is_match: bool

class FaceEmbeddingCache:
    """Cache system for face embeddings"""
    
    def __init__(self, cache_dir: str = "face_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes"""
        with open(file_path, 'rb') as f:
            file_content = f.read(1024)  # Read first 1KB for speed
        file_stat = os.stat(file_path)
        content_hash = hashlib.md5(
            file_content + 
            str(file_stat.st_mtime).encode() + 
            str(file_stat.st_size).encode()
        ).hexdigest()
        return content_hash
    
    def get_cache_path(self, image_path: str, model_name: str) -> Path:
        """Get cache file path for an image"""
        file_hash = self._get_file_hash(image_path)
        cache_filename = f"{Path(image_path).stem}_{model_name}_{file_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def load_embeddings(self, image_path: str, model_name: str) -> Optional[List[FaceEmbedding]]:
        """Load cached embeddings if available"""
        try:
            cache_path = self.get_cache_path(image_path, model_name)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.debug(f"Loaded {len(embeddings)} embeddings from cache: {image_path}")
                return embeddings
        except Exception as e:
            logger.warning(f"Failed to load cache for {image_path}: {e}")
        return None
    
    def save_embeddings(self, image_path: str, model_name: str, embeddings: List[FaceEmbedding]):
        """Save embeddings to cache"""
        try:
            cache_path = self.get_cache_path(image_path, model_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.debug(f"Saved {len(embeddings)} embeddings to cache: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {image_path}: {e}")

class ProfessionalFaceSearch:
    """
    Professional Face Search System
    
    This class provides a robust and scalable face recognition system
    with caching, multiprocessing, and comprehensive error handling.
    """
    
    # Recommended models in order of accuracy vs speed
    RECOMMENDED_MODELS = {
        'ArcFace': 'High accuracy, moderate speed',
        'Facenet512': 'Good balance of accuracy and speed', 
        'Facenet': 'Fast, good accuracy',
        'VGG-Face': 'Legacy, reliable',
        'OpenFace': 'Fastest, lower accuracy'
    }
    
    def __init__(self, 
                 model_name: str = 'ArcFace',
                 similarity_threshold: float = 0.68,
                 cache_enabled: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize the face search system
        
        Args:
            model_name: DeepFace model to use ('ArcFace', 'Facenet512', etc.)
            similarity_threshold: Minimum cosine similarity for matches (0.0-1.0)
            cache_enabled: Whether to cache embeddings
            max_workers: Number of processes for multiprocessing (None = auto)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers or os.cpu_count()
        
        self.cache = FaceEmbeddingCache() if cache_enabled else None
        
        logger.info(f"Initialized FaceSearch with model: {model_name}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Cache enabled: {cache_enabled}")
        logger.info(f"Max workers: {self.max_workers}")
        
        # Validate model
        if model_name not in self.RECOMMENDED_MODELS:
            logger.warning(f"Model {model_name} not in recommended list")
    
    def extract_face_embeddings(self, image_path: str) -> List[FaceEmbedding]:
        """
        Extract face embeddings from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of FaceEmbedding objects for each detected face
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Check cache first
        if self.cache_enabled:
            cached_embeddings = self.cache.load_embeddings(image_path, self.model_name)
            if cached_embeddings:
                return cached_embeddings
        
        embeddings = []
        
        try:
            start_time = time.time()
            
            # Use DeepFace to extract face representations
            # enforce_detection=False allows processing images without clear faces
            face_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'  # Fastest detector for production
            )
            
            processing_time = time.time() - start_time
            
            for face_idx, face_obj in enumerate(face_objs):
                # Extract face region information
                region = face_obj.get('region', {})
                face_location = (
                    region.get('y', 0),
                    region.get('x', 0) + region.get('w', 0),
                    region.get('y', 0) + region.get('h', 0), 
                    region.get('x', 0)
                )
                
                embedding = FaceEmbedding(
                    image_path=image_path,
                    face_index=face_idx,
                    embedding=face_obj['embedding'],
                    face_location=face_location,
                    confidence=1.0,  # DeepFace doesn't provide confidence directly
                    model_used=self.model_name,
                    created_at=time.time()
                )
                
                embeddings.append(embedding)
            
            logger.info(f"Extracted {len(embeddings)} faces from {os.path.basename(image_path)} "
                       f"in {processing_time:.2f}s using {self.model_name}")
            
            # Cache the results
            if self.cache_enabled and embeddings:
                self.cache.save_embeddings(image_path, self.model_name, embeddings)
                
        except Exception as e:
            logger.error(f"Failed to extract embeddings from {image_path}: {e}")
            
        return embeddings
    
    def calculate_similarities(self, 
                             reference_embedding: List[float], 
                             target_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarities between reference and target embeddings
        
        Args:
            reference_embedding: Reference face embedding
            target_embeddings: List of target face embeddings
            
        Returns:
            List of similarity scores (0.0-1.0, higher = more similar)
        """
        if not target_embeddings:
            return []
            
        try:
            ref_array = np.array(reference_embedding).reshape(1, -1)
            target_array = np.array(target_embeddings)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(ref_array, target_array)[0]
            
            # Ensure values are between 0 and 1
            similarities = np.clip(similarities, 0, 1)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return [0.0] * len(target_embeddings)
    
    def search_faces(self, 
                     reference_image_path: str, 
                     gallery_paths: List[str],
                     return_all_results: bool = False) -> List[SearchResult]:
        """
        Search for faces matching the reference image in a gallery
        
        Args:
            reference_image_path: Path to reference image
            gallery_paths: List of paths to search in
            return_all_results: If True, return all faces with scores, not just matches
            
        Returns:
            List of SearchResult objects sorted by similarity score
        """
        logger.info(f"Starting face search with reference: {os.path.basename(reference_image_path)}")
        logger.info(f"Searching in {len(gallery_paths)} images")
        
        # Extract reference embeddings
        reference_embeddings = self.extract_face_embeddings(reference_image_path)
        
        if not reference_embeddings:
            logger.error("No faces found in reference image")
            return []
        
        # Use the first (usually best/largest) face as reference
        reference_embedding = reference_embeddings[0].embedding
        logger.info(f"Using reference face with {len(reference_embedding)} dimensions")
        
        all_results = []
        
        # Process gallery images
        if self.max_workers > 1:
            results = self._search_multiprocess(reference_embedding, gallery_paths)
        else:
            results = self._search_singleprocess(reference_embedding, gallery_paths)
        
        all_results.extend(results)
        
        # Filter results based on threshold
        if not return_all_results:
            all_results = [r for r in all_results if r.is_match]
        
        # Sort by similarity score (highest first)
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"Found {len(all_results)} matching faces")
        
        return all_results
    
    def _search_singleprocess(self, 
                            reference_embedding: List[float], 
                            gallery_paths: List[str]) -> List[SearchResult]:
        """Single-process search implementation"""
        results = []
        
        for i, image_path in enumerate(gallery_paths):
            if i % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(gallery_paths)}")
                
            results.extend(self._process_single_image(reference_embedding, image_path))
        
        return results
    
    def _search_multiprocess(self, 
                           reference_embedding: List[float], 
                           gallery_paths: List[str]) -> List[SearchResult]:
        """Multi-process search implementation"""
        results = []
        
        # Split gallery into chunks for processing
        chunk_size = max(1, len(gallery_paths) // self.max_workers)
        chunks = [gallery_paths[i:i + chunk_size] 
                 for i in range(0, len(gallery_paths), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self._process_image_chunk, reference_embedding, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        return results
    
    def _process_image_chunk(self, 
                           reference_embedding: List[float], 
                           image_paths: List[str]) -> List[SearchResult]:
        """Process a chunk of images (for multiprocessing)"""
        results = []
        for image_path in image_paths:
            results.extend(self._process_single_image(reference_embedding, image_path))
        return results
    
    def _process_single_image(self, 
                            reference_embedding: List[float], 
                            image_path: str) -> List[SearchResult]:
        """Process a single image and return results"""
        results = []
        
        try:
            # Extract embeddings from gallery image
            gallery_embeddings = self.extract_face_embeddings(image_path)
            
            if not gallery_embeddings:
                return results
            
            # Calculate similarities for all faces in the image
            gallery_embedding_vectors = [emb.embedding for emb in gallery_embeddings]
            similarities = self.calculate_similarities(reference_embedding, gallery_embedding_vectors)
            
            # Create results for each face
            for emb, similarity in zip(gallery_embeddings, similarities):
                distance = 1.0 - similarity
                is_match = similarity >= self.similarity_threshold
                
                result = SearchResult(
                    image_path=image_path,
                    face_index=emb.face_index,
                    similarity_score=similarity,
                    distance=distance,
                    face_location=emb.face_location,
                    is_match=is_match
                )
                
                results.append(result)
                
                if is_match:
                    logger.info(f"✅ MATCH: {os.path.basename(image_path)} "
                              f"(face {emb.face_index}) - similarity: {similarity:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
        
        return results
    
    def get_statistics(self, results: List[SearchResult]) -> Dict:
        """Get statistics about search results"""
        if not results:
            return {}
        
        similarities = [r.similarity_score for r in results]
        matches = [r for r in results if r.is_match]
        
        stats = {
            'total_faces_found': len(results),
            'total_matches': len(matches),
            'match_rate': len(matches) / len(results) if results else 0,
            'similarity_stats': {
                'mean': np.mean(similarities),
                'median': np.median(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            },
            'threshold_used': self.similarity_threshold,
            'model_used': self.model_name
        }
        
        return stats

def main():
    """
    Example usage of the Professional Face Search System
    """
    
    # Example usage
    reference_image = "faces/me.jpg"  # Your reference image
    images_folder = "images"  # Folder with gallery images
    
    # Initialize the face search system
    face_searcher = ProfessionalFaceSearch(
        model_name='ArcFace',           # Use the most accurate model
        similarity_threshold=0.68,       # Adjust based on your needs
        cache_enabled=True,             # Enable caching for better performance
        max_workers=4                   # Use 4 processes
    )
    
    # Get all images in the gallery folder
    if not os.path.exists(images_folder):
        logger.error(f"Gallery folder not found: {images_folder}")
        return
    
    gallery_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        gallery_paths.extend(Path(images_folder).glob(f"**/{ext}"))
    
    gallery_paths = [str(p) for p in gallery_paths]
    
    if not gallery_paths:
        logger.error(f"No images found in {images_folder}")
        return
    
    logger.info(f"Found {len(gallery_paths)} images to search")
    
    # Perform the search
    start_time = time.time()
    results = face_searcher.search_faces(reference_image, gallery_paths)
    search_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"FACE SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Reference: {reference_image}")
    print(f"Gallery: {len(gallery_paths)} images")
    print(f"Search time: {search_time:.2f} seconds")
    print(f"Model: {face_searcher.model_name}")
    print(f"Threshold: {face_searcher.similarity_threshold}")
    
    if results:
        print(f"\nFound {len(results)} matching images:")
        print(f"{'Image':<40} {'Face':<5} {'Similarity':<10} {'Status'}")
        print(f"{'-'*65}")
        
        for result in results:
            image_name = os.path.basename(result.image_path)
            status = "✅ MATCH" if result.is_match else "❌ No match"
            print(f"{image_name:<40} {result.face_index:<5} {result.similarity_score:<10.3f} {status}")
        
        # Get and print statistics
        stats = face_searcher.get_statistics(results)
        if stats:
            print(f"\n{'='*60}")
            print(f"STATISTICS")
            print(f"{'='*60}")
            print(f"Total faces found: {stats['total_faces_found']}")
            print(f"Total matches: {stats['total_matches']}")
            print(f"Match rate: {stats['match_rate']:.1%}")
            print(f"Average similarity: {stats['similarity_stats']['mean']:.3f}")
            print(f"Similarity range: {stats['similarity_stats']['min']:.3f} - {stats['similarity_stats']['max']:.3f}")
    else:
        print(f"\n❌ No matching faces found.")
        print(f"Try lowering the similarity threshold (current: {face_searcher.similarity_threshold})")

if __name__ == "__main__":
    main()