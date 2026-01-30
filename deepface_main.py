from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import tempfile
import os
from deepface import DeepFace
import json
import numpy as np
import aiohttp

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Service (DeepFace)", version="1.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Face Recognition Service is running with DeepFace"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "DeepFace service is running"}

@app.post("/encode-face")
async def encode_face(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file for encoding: {file.filename}")
        
        # Read the image file
        image_data = await file.read()
        
        # Create a temporary file for DeepFace processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name
        
        try:
            # Use DeepFace to get face embedding
            # Using Facenet model for face recognition
            embedding = DeepFace.represent(temp_file_path, model_name='Facenet', enforce_detection=False)
            
            # DeepFace.represent returns a list of dictionaries
            if embedding and len(embedding) > 0:
                face_encoding = embedding[0]['embedding']  # Get the first (and usually only) face
                
                return {
                    "face_encoding": face_encoding,
                    "has_face": True,
                    "message": f"Successfully extracted face encoding using DeepFace for {file.filename}"
                }
            else:
                return {
                    "face_encoding": [],
                    "has_face": False,
                    "message": f"No face detected in {file.filename}"
                }
                
        except Exception as e:
            logger.warning(f"DeepFace processing failed: {e}")
            return {
                "face_encoding": [],
                "has_face": False,
                "message": f"Face detection failed: {str(e)}"
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Error in encode_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/compare-faces")
async def compare_faces_json(request: Request):
    try:
        # Get the JSON body
        body = await request.json()
        reference_encoding = body.get("reference_encoding", [])
        target_encodings = body.get("target_encodings", [])
        
        logger.info(f"Received face comparison request: ref_encoding_len={len(reference_encoding)}, target_count={len(target_encodings)}")
        
        if not reference_encoding or not target_encodings:
            return {"matches": [], "message": "No encodings provided"}
        
        # Calculate cosine similarity between encodings (DeepFace uses cosine similarity)
        matches = []
        similarities = []
        
        logger.info(f"Starting DeepFace comparison: reference_len={len(reference_encoding)}, targets_count={len(target_encodings)}")
        
        for i, target_encoding in enumerate(target_encodings):
            if len(target_encoding) != len(reference_encoding):
                logger.warning(f"Target encoding {i} length mismatch: {len(target_encoding)} != {len(reference_encoding)}")
                continue
                
            # Calculate cosine similarity manually (same as DeepFace uses internally)
            ref_array = np.array(reference_encoding)
            target_array = np.array(target_encoding)
            
            # Cosine similarity calculation
            dot_product = np.dot(ref_array, target_array)
            ref_norm = np.linalg.norm(ref_array)
            target_norm = np.linalg.norm(target_array)
            
            if ref_norm > 0 and target_norm > 0:
                cosine_similarity = dot_product / (ref_norm * target_norm)
                # Convert cosine similarity to a distance (lower distance = more similar)
                cosine_distance = 1 - cosine_similarity
                # Convert to similarity percentage (higher = more similar)
                similarity = 1 - cosine_distance
            else:
                similarity = 0.0
                cosine_distance = 1.0
            
            similarities.append(similarity)
        
        # Use much more permissive threshold for better matching
        base_threshold = 0.6  # Much more permissive
        
        # Adaptive thresholding based on similarity distribution
        if similarities:
            similarities_array = np.array(similarities)
            max_sim = np.max(similarities_array)
            mean_sim = np.mean(similarities_array)
            std_sim = np.std(similarities_array)
            
            logger.info(f"Similarity analysis: max={max_sim:.3f}, mean={mean_sim:.3f}, std={std_sim:.3f}")
            
            # Much more permissive adaptive threshold 
            if max_sim > 0.7:  # Good similarity found
                adaptive_threshold = max(0.5, max_sim - (std_sim * 2.0))
            elif max_sim > 0.5:  # Moderate similarity
                adaptive_threshold = max(0.4, mean_sim + (std_sim * 0.2))
            else:  # Lower similarities overall
                adaptive_threshold = max(0.3, mean_sim)
            
            # Ensure threshold is reasonable - much more permissive range
            adaptive_threshold = max(0.3, min(0.8, adaptive_threshold))
            
            logger.info(f"Similarity stats: mean={mean_sim:.3f}, std={std_sim:.3f}, max={max_sim:.3f}, threshold={adaptive_threshold:.3f}")
        else:
            adaptive_threshold = base_threshold
        
        # Generate matches based on threshold
        for i, similarity in enumerate(similarities):
            distance = 1.0 - similarity
            is_match = similarity >= adaptive_threshold
            
            logger.info(f"Target {i}: cosine_distance={distance:.4f}, similarity={similarity:.4f}, is_match={is_match} (threshold={adaptive_threshold:.3f})")
            
            matches.append({
                "index": i,
                "distance": float(distance),
                "similarity": float(similarity),
                "is_match": bool(is_match)
            })
        
        match_count = len([m for m in matches if m["is_match"]])
        
        return {
            "matches": matches,
            "message": f"Found {match_count} matches out of {len(matches)} faces using DeepFace"
        }
        
    except Exception as e:
        logger.error(f"Error in compare_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    weddingId: str = Form(...),
    authorization: str = Header(None)
):
    try:
        logger.info(f"Received face search request for wedding: {weddingId}")
        
        # Step 1: Encode the uploaded face
        face_encoding_response = await encode_face(file)
        if not face_encoding_response.get("has_face"):
            return {"matches": [], "message": "No face detected in uploaded image"}
        
        uploaded_encoding = face_encoding_response["face_encoding"]
        
        # Step 2: Get all face encodings for this wedding from the backend
        import json
        
        try:
            # Prepare headers for backend API call
            headers = {}
            if authorization:
                headers['Authorization'] = authorization
                logger.info(f"Using authorization header for backend API call: {authorization[:20]}...")
            else:
                logger.warning("No authorization token provided")
            
            logger.info(f"Making request to: {BACKEND_URL}/api/photos/{weddingId}")
            logger.info(f"Request headers: {headers}")
            
            logger.info("Creating aiohttp ClientSession...")
            async with aiohttp.ClientSession() as session:
                logger.info("ClientSession created successfully")
                # Get photos for this wedding
                logger.info(f"About to make GET request to backend API...")
                async with session.get(f"{BACKEND_URL}/api/photos/{weddingId}", headers=headers) as resp:
                    logger.info(f"Backend API response status: {resp.status}")
                    if resp.status != 200:
                        logger.error(f"Failed to get photos from backend: {resp.status}")
                        return {"matches": [], "message": f"Failed to get photos from backend (status: {resp.status})"}
                    
                    photos_data = await resp.json()
                    photos = photos_data.get("photos", [])
                    
                    if not photos:
                        return {"matches": [], "message": "No photos found for this wedding"}
                    
                    logger.info(f"Found {len(photos)} photos for wedding {weddingId}")
                    
                    # Step 3: Collect face encodings for all photos
                    target_encodings = []
                    photo_metadata = []
                    
                    for photo in photos:
                        if photo.get("faceEncoding"):
                            target_encodings.append(photo["faceEncoding"])
                            photo_metadata.append({
                                "photo_id": photo.get("id"),
                                "photo_name": photo.get("filename", f"Photo {photo.get('id')}")
                            })
                            logger.info(f"Added face encoding for photo {photo.get('id')} - encoding length: {len(photo.get('faceEncoding', []))}")
                    
                    if not target_encodings:
                        logger.warning("No face encodings found for photos in this wedding")
                        return {"matches": [], "message": "No face encodings found for photos in this wedding"}
                    
                    logger.info(f"Comparing against {len(target_encodings)} face encodings")
                    logger.info(f"Reference encoding length: {len(uploaded_encoding)}")
                    
                    # Step 4: Compare faces using existing endpoint
                    comparison_request = {
                        "reference_encoding": uploaded_encoding,
                        "target_encodings": target_encodings
                    }
                    
                    # Use the existing compare_faces_json function
                    from fastapi import Request
                    import json
                    
                    # Create a mock request object for the comparison
                    class MockRequest:
                        def __init__(self, json_data):
                            self._json_data = json_data
                        
                        async def json(self):
                            return self._json_data
                    
                    mock_request = MockRequest(comparison_request)
                    comparison_result = await compare_faces_json(mock_request)
                    
                    # Step 5: Format results with photo information
                    matches = []
                    for i, match in enumerate(comparison_result["matches"]):
                        if match["is_match"] and i < len(photo_metadata):
                            matches.append({
                                "photo_id": photo_metadata[i]["photo_id"],
                                "photo_name": photo_metadata[i]["photo_name"],
                                "confidence": match["similarity"],
                                "distance": match["distance"]
                            })
                    
                    logger.info(f"Found {len(matches)} face matches for wedding {weddingId}")
                    
                    return {
                        "matches": matches,
                        "message": f"Found {len(matches)} matching photos using DeepFace"
                    }
                    
        except aiohttp.ClientError as e:
            logger.error(f"Backend connection error: {str(e)}")
            return {"matches": [], "message": f"Backend connection error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error during face search: {str(e)}")
            return {"matches": [], "message": f"Error during face search: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error in search_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching faces: {str(e)}")

@app.post("/verify-faces")
async def verify_faces(
    reference_image: UploadFile = File(...),
    target_images: str = Form(...),  # JSON string of image URLs/paths
    threshold: float = Form(0.68)
):
    """
    Simplified face verification using DeepFace.verify() - more accurate and simpler
    Similar to your working example with my_face and images_folder
    """
    try:
        logger.info(f"Received face verification request with threshold: {threshold}")
        
        # Save reference image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_temp:
            ref_content = await reference_image.read()
            ref_temp.write(ref_content)
            ref_path = ref_temp.name
        
        try:
            # Parse target images from JSON
            import json
            target_image_list = json.loads(target_images)
            
            result_matches = []
            
            for img_data in target_image_list:
                img_path = img_data.get("path")
                img_id = img_data.get("id")
                img_name = img_data.get("name", f"photo_{img_id}")
                
                if not img_path or not os.path.exists(img_path):
                    continue
                
                try:
                    # Use DeepFace.verify - simpler and more reliable
                    result = DeepFace.verify(
                        img1_path=ref_path, 
                        img2_path=img_path, 
                        model_name="Facenet",
                        distance_metric="cosine",
                        enforce_detection=False
                    )
                    
                    if result["verified"] and result["distance"] <= threshold:
                        result_matches.append({
                            "photo_id": img_id,
                            "photo_name": img_name,
                            "distance": float(result["distance"]),
                            "similarity": float(1 - result["distance"]),
                            "verified": True,
                            "threshold_used": float(result["threshold"])
                        })
                        logger.info(f"MATCH: {img_name} - distance: {result['distance']:.4f}")
                    else:
                        logger.info(f"NO MATCH: {img_name} - distance: {result['distance']:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Verification failed for {img_name}: {str(e)}")
                    continue
            
            return {
                "matches": result_matches,
                "message": f"DeepFace.verify found {len(result_matches)} matches using threshold {threshold}",
                "total_checked": len(target_image_list)
            }
            
        finally:
            # Clean up reference image
            if os.path.exists(ref_path):
                os.unlink(ref_path)
                
    except Exception as e:
        logger.error(f"Error in verify_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error verifying faces: {str(e)}")

@app.post("/compare-faces-form")
async def compare_faces(
    reference_image: UploadFile = File(...),
    threshold: float = Form(0.6),
    face_encodings: str = Form(...)
):
    try:
        logger.info(f"Received face comparison request with threshold: {threshold}")
        
        # Parse the face_encodings JSON
        try:
            stored_encodings = json.loads(face_encodings)
        except json.JSONDecodeError:
            stored_encodings = []
        
        # Create a temporary file for the reference image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_data = await reference_image.read()
            temp_file.write(image_data)
            temp_file_path = temp_file.name
        
        try:
            # Get reference face encoding using DeepFace
            ref_embedding = DeepFace.represent(temp_file_path, model_name='Facenet', enforce_detection=False)
            
            if not ref_embedding or len(ref_embedding) == 0:
                return {
                    "matches": [],
                    "message": "No face detected in reference image"
                }
            
            ref_encoding = ref_embedding[0]['embedding']
            
            # Compare with stored encodings
            matches = []
            for i, encoding_data in enumerate(stored_encodings):
                if 'encoding' in encoding_data:
                    stored_encoding = encoding_data['encoding']
                    
                    # Calculate cosine similarity
                    ref_array = np.array(ref_encoding)
                    stored_array = np.array(stored_encoding)
                    
                    dot_product = np.dot(ref_array, stored_array)
                    ref_norm = np.linalg.norm(ref_array)
                    stored_norm = np.linalg.norm(stored_array)
                    
                    if ref_norm > 0 and stored_norm > 0:
                        cosine_similarity = dot_product / (ref_norm * stored_norm)
                        cosine_distance = 1 - cosine_similarity
                        similarity = 1 - cosine_distance
                        
                        # Convert threshold from distance to similarity
                        similarity_threshold = 1 - threshold
                        is_match = similarity >= similarity_threshold
                        
                        matches.append({
                            "photo_id": encoding_data.get("photo_id", f"photo_{i}"),
                            "distance": float(cosine_distance),
                            "similarity": float(similarity),
                            "is_match": is_match
                        })
            
            successful_matches = [m for m in matches if m["is_match"]]
            
            return {
                "matches": successful_matches,
                "message": f"DeepFace found {len(successful_matches)} matches out of {len(matches)} photos"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in compare_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Face Recognition Service with DeepFace...")
    uvicorn.run(app, host="0.0.0.0", port=8081)