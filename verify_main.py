from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uvicorn
import logging
import tempfile
import os
import json
import numpy as np
import requests
from typing import List

# Import the professional face search system if available
try:
    from professional_face_search import ProfessionalFaceSearch
    PROFESSIONAL_SYSTEM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Professional Face Search System loaded successfully")
except ImportError:
    PROFESSIONAL_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Professional Face Search System not available, using fallback")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Enhanced Face Recognition Service", version="2.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global face searcher instance for professional system
professional_searcher = None

if PROFESSIONAL_SYSTEM_AVAILABLE:
    try:
        professional_searcher = ProfessionalFaceSearch(
            model_name='ArcFace',          # Most accurate model
            similarity_threshold=0.68,      # Optimal threshold
            cache_enabled=True,            # Enable caching for performance
            max_workers=2                  # Suitable for API usage
        )
        logger.info("Professional Face Searcher initialized with ArcFace model")
    except Exception as e:
        logger.error(f"Failed to initialize professional searcher: {e}")
        professional_searcher = None

@app.get("/")
async def root():
    return {"message": "Face Recognition Service is running with DeepFace.verify"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "DeepFace.verify service is running"}

@app.post("/encode-face")
async def encode_face(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file for encoding: {file.filename}")
        
        # Read the image file
        image_data = await file.read()
        
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name
        
        try:
            # Try using professional system first
            if PROFESSIONAL_SYSTEM_AVAILABLE and professional_searcher:
                logger.info("Using Professional Face Search System for encoding")
                try:
                    embeddings = professional_searcher.extract_face_embeddings(temp_file_path)
                    
                    if embeddings:
                        # Use the first (best) face embedding
                        best_embedding = embeddings[0]
                        
                        return {
                            "face_encoding": best_embedding.embedding,
                            "has_face": True,
                            "message": f"Professional system extracted face encoding using {best_embedding.model_used} for {file.filename}",
                            "model_used": best_embedding.model_used,
                            "faces_found": len(embeddings),
                            "confidence": best_embedding.confidence,
                            "system": "professional"
                        }
                    else:
                        return {
                            "face_encoding": [],
                            "has_face": False,
                            "message": f"No face detected by professional system in {file.filename}",
                            "system": "professional"
                        }
                except Exception as prof_e:
                    logger.warning(f"Professional system failed, falling back: {prof_e}")
                    # Fall through to legacy system
            
            # Fallback to legacy DeepFace.represent
            logger.info("Using legacy DeepFace system for encoding")
            embedding = DeepFace.represent(temp_file_path, model_name='Facenet', enforce_detection=False)
            
            if embedding and len(embedding) > 0:
                face_encoding = embedding[0]['embedding']
                
                return {
                    "face_encoding": face_encoding,
                    "has_face": True,
                    "message": f"Legacy system extracted face encoding using Facenet for {file.filename}",
                    "model_used": "Facenet",
                    "faces_found": len(embedding),
                    "system": "legacy"
                }
            else:
                return {
                    "face_encoding": [],
                    "has_face": False,
                    "message": f"No face detected by legacy system in {file.filename}",
                    "system": "legacy"
                }
                
        except Exception as e:
            logger.warning(f"All face processing failed: {e}")
            return {
                "face_encoding": [],
                "has_face": False,
                "message": f"Face detection failed: {str(e)}",
                "system": "failed"
            }
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Error in encode_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/verify-faces-simple")
async def verify_faces_simple(
    reference_image: UploadFile = File(...),
    target_images: str = Form(...)  # JSON array of image info with R2 keys
):
    """
    Simple face verification using DeepFace.verify() - exactly like your example:
    
    my_face = "faces/me.jpg"
    images_folder = "images"
    
    for img in os.listdir(images_folder):
        path = os.path.join(images_folder, img)
        result = DeepFace.verify(my_face, path, model_name="Facenet")
        if result["verified"]:
            result_list.append(img)
    """
    try:
        logger.info("Starting simple face verification using DeepFace.verify()")
        
        # Save reference image to temporary file (like my_face)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_temp:
            ref_content = await reference_image.read()
            ref_temp.write(ref_content)
            my_face = ref_temp.name
        
        try:
            # Parse target images from JSON
            target_image_list = json.loads(target_images)
            result_list = []
            
            logger.info(f"Comparing reference image against {len(target_image_list)} target images")
            
            # Handle image URLs (for downloading from backend) or local paths
            for img_data in target_image_list:
                img_path = img_data.get("path")
                img_id = img_data.get("id")
                img_name = img_data.get("name", f"photo_{img_id}")
                img_url = img_data.get("url")  # New: support for URLs
                
                logger.info(f"Processing image: id={img_id}, name={img_name}, url={img_url}, path={img_path}")
                
                actual_path = img_path
                temp_file_to_delete = None
                
                # If URL is provided, download the image first
                if img_url:
                    try:
                        logger.info(f"Attempting to download image from URL: {img_url}")
                        response = requests.get(img_url, timeout=10)
                        logger.info(f"Download response status: {response.status_code}")
                        if response.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                                temp_file.write(response.content)
                                actual_path = temp_file.name
                                temp_file_to_delete = temp_file.name
                            logger.info(f"Successfully downloaded image to: {actual_path}")
                        else:
                            logger.warning(f"Failed to download image from URL: {img_url} - Status: {response.status_code}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error downloading image from URL {img_url}: {e}")
                        continue
                elif not img_path or not os.path.exists(img_path):
                    logger.warning(f"Image path not found: {img_path}")
                    continue
                
                try:
                    # This is exactly your working code pattern
                    result = DeepFace.verify(my_face, actual_path, model_name="Facenet")
                    
                    if result["verified"]:
                        result_list.append({
                            "photo_id": img_id,
                            "photo_name": img_name,
                            "distance": float(result["distance"]),
                            "threshold": float(result["threshold"]),
                            "verified": True
                        })
                        logger.info(f"✅ MATCH: {img_name} - distance: {result['distance']:.4f} (threshold: {result['threshold']:.4f})")
                    else:
                        logger.info(f"❌ NO MATCH: {img_name} - distance: {result['distance']:.4f} (threshold: {result['threshold']:.4f})")

                except Exception as e:
                    logger.warning(f"Error verifying {img_name}: {str(e)}")
                finally:
                    # Clean up temporary downloaded file
                    if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                        os.unlink(temp_file_to_delete)
                    continue
            
            # Exactly like your print statement
            logger.info(f"Benim olduğum görseller: {[r['photo_name'] for r in result_list]}")
            
            return {
                "matches": result_list,
                "message": f"Found {len(result_list)} verified matches using DeepFace.verify",
                "total_checked": len(target_image_list),
                "verified_images": [r['photo_name'] for r in result_list]
            }
            
        finally:
            # Clean up reference image
            if os.path.exists(my_face):
                os.unlink(my_face)
                
    except Exception as e:
        logger.error(f"Error in verify_faces_simple: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error verifying faces: {str(e)}")

@app.post("/compare-faces")
async def compare_faces_json(request: Request):
    """
    Enhanced face comparison using professional system when available
    """
    try:
        body = await request.json()
        reference_encoding = body.get("reference_encoding", [])
        target_encodings = body.get("target_encodings", [])
        
        logger.info(f"Received face comparison request: ref_encoding_len={len(reference_encoding)}, target_count={len(target_encodings)}")
        
        if not reference_encoding or not target_encodings:
            return {"matches": [], "message": "No encodings provided"}
        
        matches = []
        
        # Try professional system first
        if PROFESSIONAL_SYSTEM_AVAILABLE and professional_searcher:
            logger.info("Using Professional Face Search System for comparison")
            try:
                similarities = professional_searcher.calculate_similarities(reference_encoding, target_encodings)
                
                for i, similarity in enumerate(similarities):
                    distance = 1.0 - similarity
                    is_match = similarity >= professional_searcher.similarity_threshold
                    
                    matches.append({
                        "index": i,
                        "distance": float(distance),
                        "similarity": float(similarity),
                        "is_match": bool(is_match),
                        "threshold_used": professional_searcher.similarity_threshold,
                        "system": "professional"
                    })
                
                match_count = len([m for m in matches if m["is_match"]])
                
                return {
                    "matches": matches,
                    "message": f"Professional system found {match_count} matches out of {len(matches)} faces using {professional_searcher.model_name}",
                    "model_used": professional_searcher.model_name,
                    "threshold_used": professional_searcher.similarity_threshold
                }
            except Exception as prof_e:
                logger.warning(f"Professional comparison failed, using fallback: {prof_e}")
        
        # Fallback to legacy cosine similarity calculation
        logger.info("Using legacy cosine similarity calculation")
        for i, target_encoding in enumerate(target_encodings):
            if len(target_encoding) != len(reference_encoding):
                continue
                
            # Calculate cosine similarity
            ref_array = np.array(reference_encoding)
            target_array = np.array(target_encoding)
            
            dot_product = np.dot(ref_array, target_array)
            ref_norm = np.linalg.norm(ref_array)
            target_norm = np.linalg.norm(target_array)
            
            if ref_norm > 0 and target_norm > 0:
                cosine_similarity = dot_product / (ref_norm * target_norm)
                cosine_distance = 1 - cosine_similarity
                similarity = 1 - cosine_distance
            else:
                similarity = 0.0
                cosine_distance = 1.0
            
            # Use more permissive threshold for legacy system
            threshold = 0.6
            is_match = similarity >= threshold
            
            matches.append({
                "index": i,
                "distance": float(cosine_distance),
                "similarity": float(similarity),
                "is_match": bool(is_match),
                "threshold_used": threshold,
                "system": "legacy"
            })
        
        match_count = len([m for m in matches if m["is_match"]])
        
        return {
            "matches": matches,
            "message": f"Legacy system found {match_count} matches out of {len(matches)} faces using cosine similarity",
            "model_used": "Legacy Cosine",
            "threshold_used": 0.6
        }
        
    except Exception as e:
        logger.error(f"Error in compare_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8082))
    logger.info(f"Starting Face Recognition Service with DeepFace.verify() on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)