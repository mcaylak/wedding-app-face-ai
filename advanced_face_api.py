"""
Advanced Professional Face Recognition API
=========================================

FastAPI service that integrates the professional face search system
with the wedding photo backend.

Features:
- Professional face recognition using optimized DeepFace system
- Embedding caching for performance
- Multi-model support (ArcFace, Facenet512, etc.)
- Configurable similarity thresholds  
- Batch processing capabilities
- Comprehensive error handling
- Production-ready logging

Version: 2.0.0
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
import aiofiles
from typing import List, Dict, Optional
import logging
import asyncio
from pathlib import Path

# Import our professional face search system
from professional_face_search import ProfessionalFaceSearch, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Face Recognition API", 
    version="2.0.0",
    description="Professional face recognition service using optimized DeepFace"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global face search instances (one per model for performance)
FACE_SEARCHERS: Dict[str, ProfessionalFaceSearch] = {}

# Recommended configurations for different use cases
PRESET_CONFIGS = {
    "high_accuracy": {
        "model_name": "ArcFace",
        "similarity_threshold": 0.68,
        "description": "Most accurate results, slower processing"
    },
    "balanced": {
        "model_name": "Facenet512", 
        "similarity_threshold": 0.65,
        "description": "Good balance of speed and accuracy"
    },
    "fast": {
        "model_name": "Facenet",
        "similarity_threshold": 0.60,
        "description": "Fastest processing, good accuracy"
    },
    "legacy": {
        "model_name": "VGG-Face",
        "similarity_threshold": 0.70,
        "description": "Legacy model, very reliable"
    }
}

def get_face_searcher(model_name: str = "ArcFace", 
                     similarity_threshold: float = 0.68,
                     use_cache: bool = True) -> ProfessionalFaceSearch:
    """Get or create a face searcher instance"""
    cache_key = f"{model_name}_{similarity_threshold}_{use_cache}"
    
    if cache_key not in FACE_SEARCHERS:
        logger.info(f"Creating new face searcher: {cache_key}")
        FACE_SEARCHERS[cache_key] = ProfessionalFaceSearch(
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            cache_enabled=use_cache,
            max_workers=2  # Reduced for API usage
        )
    
    return FACE_SEARCHERS[cache_key]

@app.get("/")
async def root():
    return {
        "message": "Advanced Face Recognition API",
        "version": "2.0.0",
        "models_available": list(ProfessionalFaceSearch.RECOMMENDED_MODELS.keys()),
        "presets_available": list(PRESET_CONFIGS.keys())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Professional Face Recognition Service is running",
        "models_loaded": len(FACE_SEARCHERS),
        "cache_enabled": True
    }

@app.get("/models")
async def list_models():
    """Get available models and their descriptions"""
    return {
        "models": ProfessionalFaceSearch.RECOMMENDED_MODELS,
        "presets": PRESET_CONFIGS,
        "default_model": "ArcFace",
        "default_threshold": 0.68
    }

@app.post("/extract-embeddings")
async def extract_embeddings(
    file: UploadFile = File(...),
    model_name: str = Form("ArcFace"),
    use_cache: bool = Form(True)
):
    """
    Extract face embeddings from an uploaded image
    
    This endpoint extracts embeddings from all faces found in the image
    and can cache results for better performance.
    """
    try:
        logger.info(f"Extracting embeddings from: {file.filename} using {model_name}")
        
        # Validate model
        if model_name not in ProfessionalFaceSearch.RECOMMENDED_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Get face searcher instance
            face_searcher = get_face_searcher(model_name=model_name, use_cache=use_cache)
            
            # Extract embeddings
            embeddings = face_searcher.extract_face_embeddings(temp_path)
            
            if not embeddings:
                return {
                    "success": False,
                    "message": "No faces detected in the image",
                    "faces_found": 0,
                    "embeddings": []
                }
            
            # Convert embeddings to API response format
            api_embeddings = []
            for emb in embeddings:
                api_embeddings.append({
                    "face_index": emb.face_index,
                    "embedding": emb.embedding,
                    "face_location": emb.face_location,
                    "confidence": emb.confidence,
                    "model_used": emb.model_used,
                    "embedding_dimensions": len(emb.embedding)
                })
            
            return {
                "success": True,
                "message": f"Successfully extracted {len(embeddings)} face embeddings",
                "faces_found": len(embeddings),
                "embeddings": api_embeddings,
                "model_used": model_name,
                "cache_used": use_cache
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error extracting embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting embeddings: {str(e)}")

@app.post("/search-faces-advanced")
async def search_faces_advanced(
    reference_image: UploadFile = File(...),
    target_images: str = Form(...),  # JSON array of image info
    model_name: str = Form("ArcFace"),
    similarity_threshold: float = Form(0.68),
    preset: Optional[str] = Form(None),
    use_cache: bool = Form(True),
    return_all_results: bool = Form(False)
):
    """
    Advanced face search using the professional system
    
    This is the main endpoint that replaces the simple verify approach
    with a robust, production-ready face search system.
    """
    try:
        logger.info(f"Advanced face search request using {model_name}")
        
        # Apply preset configuration if specified
        if preset and preset in PRESET_CONFIGS:
            config = PRESET_CONFIGS[preset]
            model_name = config["model_name"]
            similarity_threshold = config["similarity_threshold"]
            logger.info(f"Using preset '{preset}': {config['description']}")
        
        # Parse target images
        try:
            target_image_list = json.loads(target_images)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid target_images JSON format")
        
        if not target_image_list:
            return {"matches": [], "message": "No target images provided"}
        
        # Save reference image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_temp:
            ref_content = await reference_image.read()
            ref_temp.write(ref_content)
            ref_path = ref_temp.name
        
        try:
            # Get face searcher instance
            face_searcher = get_face_searcher(
                model_name=model_name,
                similarity_threshold=similarity_threshold,
                use_cache=use_cache
            )
            
            # Prepare gallery paths (only include existing files)
            gallery_paths = []
            valid_images = []
            
            for img_data in target_image_list:
                img_path = img_data.get("path")
                if img_path and os.path.exists(img_path):
                    gallery_paths.append(img_path)
                    valid_images.append(img_data)
                else:
                    logger.warning(f"Image path not found: {img_path}")
            
            if not gallery_paths:
                return {
                    "matches": [],
                    "message": "No valid target images found",
                    "total_checked": 0
                }
            
            logger.info(f"Searching reference against {len(gallery_paths)} images")
            
            # Perform the search
            search_results = face_searcher.search_faces(
                reference_image_path=ref_path,
                gallery_paths=gallery_paths,
                return_all_results=return_all_results
            )
            
            # Convert results to API format
            api_results = []
            image_path_to_data = {img["path"]: img for img in valid_images}
            
            for result in search_results:
                img_data = image_path_to_data.get(result.image_path, {})
                
                api_results.append({
                    "photo_id": img_data.get("id"),
                    "photo_name": img_data.get("name", os.path.basename(result.image_path)),
                    "similarity_score": result.similarity_score,
                    "distance": result.distance,
                    "verified": result.is_match,
                    "face_index": result.face_index,
                    "face_location": result.face_location,
                    "threshold_used": similarity_threshold,
                    "model_used": model_name
                })
            
            # Get search statistics
            stats = face_searcher.get_statistics(search_results)
            
            return {
                "matches": api_results,
                "message": f"Advanced search completed: found {len(api_results)} results",
                "total_checked": len(gallery_paths),
                "statistics": stats,
                "configuration": {
                    "model_name": model_name,
                    "similarity_threshold": similarity_threshold,
                    "preset_used": preset,
                    "cache_enabled": use_cache
                }
            }
            
        finally:
            # Clean up reference image
            if os.path.exists(ref_path):
                os.unlink(ref_path)
                
    except Exception as e:
        logger.error(f"Error in advanced face search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced face search failed: {str(e)}")

@app.post("/compare-embeddings")  
async def compare_embeddings(request: Request):
    """
    Compare pre-computed embeddings using professional similarity calculation
    
    This endpoint accepts pre-computed embeddings and performs similarity
    calculations using the professional system.
    """
    try:
        body = await request.json()
        reference_embedding = body.get("reference_embedding", [])
        target_embeddings = body.get("target_encodings", [])  # Keep backend compatibility
        similarity_threshold = body.get("similarity_threshold", 0.68)
        model_name = body.get("model_name", "ArcFace")
        
        logger.info(f"Comparing embeddings: ref={len(reference_embedding)}D, targets={len(target_embeddings)}")
        
        if not reference_embedding or not target_embeddings:
            return {"matches": [], "message": "No embeddings provided"}
        
        # Get face searcher for similarity calculation
        face_searcher = get_face_searcher(model_name=model_name, similarity_threshold=similarity_threshold)
        
        # Calculate similarities using professional system
        similarities = face_searcher.calculate_similarities(reference_embedding, target_embeddings)
        
        # Generate results
        matches = []
        for i, similarity in enumerate(similarities):
            distance = 1.0 - similarity
            is_match = similarity >= similarity_threshold
            
            matches.append({
                "index": i,
                "similarity": float(similarity),
                "distance": float(distance), 
                "is_match": bool(is_match),
                "threshold_used": similarity_threshold
            })
            
            if is_match:
                logger.info(f"✅ MATCH: index {i} - similarity: {similarity:.3f}")
        
        match_count = sum(1 for m in matches if m["is_match"])
        
        return {
            "matches": matches,
            "message": f"Professional embedding comparison: {match_count} matches found",
            "configuration": {
                "model_name": model_name,
                "similarity_threshold": similarity_threshold,
                "total_comparisons": len(target_embeddings)
            }
        }
        
    except Exception as e:
        logger.error(f"Error comparing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding comparison failed: {str(e)}")

@app.post("/batch-extract")
async def batch_extract_embeddings(
    files: List[UploadFile] = File(...),
    model_name: str = Form("ArcFace"),
    use_cache: bool = Form(True)
):
    """
    Extract embeddings from multiple images in batch
    
    This endpoint processes multiple images efficiently and returns
    all embeddings for batch operations.
    """
    try:
        logger.info(f"Batch extracting embeddings from {len(files)} files using {model_name}")
        
        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
        
        # Get face searcher instance
        face_searcher = get_face_searcher(model_name=model_name, use_cache=use_cache)
        
        batch_results = []
        temp_files = []
        
        try:
            # Save all files temporarily
            for file in files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append((temp_file.name, file.filename))
            
            # Process all files
            for temp_path, original_filename in temp_files:
                try:
                    embeddings = face_searcher.extract_face_embeddings(temp_path)
                    
                    file_result = {
                        "filename": original_filename,
                        "success": True,
                        "faces_found": len(embeddings),
                        "embeddings": [
                            {
                                "face_index": emb.face_index,
                                "embedding": emb.embedding,
                                "face_location": emb.face_location,
                                "confidence": emb.confidence
                            }
                            for emb in embeddings
                        ]
                    }
                except Exception as e:
                    file_result = {
                        "filename": original_filename,
                        "success": False,
                        "error": str(e),
                        "faces_found": 0,
                        "embeddings": []
                    }
                
                batch_results.append(file_result)
            
            total_faces = sum(r["faces_found"] for r in batch_results)
            successful_files = sum(1 for r in batch_results if r["success"])
            
            return {
                "success": True,
                "message": f"Batch processing completed: {total_faces} faces found in {successful_files}/{len(files)} files",
                "results": batch_results,
                "summary": {
                    "total_files": len(files),
                    "successful_files": successful_files,
                    "total_faces_found": total_faces,
                    "model_used": model_name
                }
            }
            
        finally:
            # Clean up all temp files
            for temp_path, _ in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")

# Keep backward compatibility endpoints
@app.post("/encode-face")
async def encode_face_legacy(file: UploadFile = File(...)):
    """Legacy endpoint for backward compatibility"""
    result = await extract_embeddings(file, model_name="Facenet", use_cache=True)
    
    if result["success"] and result["embeddings"]:
        return {
            "face_encoding": result["embeddings"][0]["embedding"],
            "has_face": True,
            "message": result["message"]
        }
    else:
        return {
            "face_encoding": [],
            "has_face": False,
            "message": result["message"]
        }

@app.post("/compare-faces")
async def compare_faces_legacy(request: Request):
    """Legacy endpoint for backward compatibility"""
    return await compare_embeddings(request)

if __name__ == "__main__":
    logger.info("Starting Advanced Face Recognition API...")
    uvicorn.run(app, host="0.0.0.0", port=8083)