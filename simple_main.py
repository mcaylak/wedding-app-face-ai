from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Service (Debug Mode)", version="1.0.0")

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
    return {"message": "Face Recognition Service is running in debug mode"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running in debug mode - face_recognition not loaded"}

@app.post("/encode-face")
async def encode_face(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file for encoding: {file.filename}")
        
        # Read the image file
        image_data = await file.read()
        
        # Simulate face recognition with basic image analysis
        import hashlib
        import random
        from PIL import Image
        import io
        import numpy as np
        
        try:
            # Load and analyze the image
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to standard size for analysis
            img = img.resize((64, 64))
            
            # Extract robust facial features from the image
            img_array = np.array(img)
            
            # Convert to grayscale for luminance-based features
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
                
            # Normalize the image to reduce lighting variations
            gray = gray.astype(np.float32)
            gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)
            gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)
            
            features = []
            h, w = gray.shape
            
            # 1. Regional average intensities (lighting-normalized)
            for y in range(0, h, 8):  # Smaller regions for better detail
                for x in range(0, w, 8):
                    region = gray[y:y+8, x:x+8]
                    if region.size > 0:
                        features.append(np.mean(region))
            
            # 2. Gradient features (edge information, pose-invariant)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            for y in range(0, h, 16):
                for x in range(0, w, 16):
                    region = grad_magnitude[y:y+16, x:x+16]
                    if region.size > 0:
                        features.append(np.mean(region))
            
            # 3. Symmetry features (facial symmetry)
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            min_w = min(left_half.shape[1], right_half.shape[1])
            symmetry_diff = np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w]))
            features.append(symmetry_diff)
            
            # 4. Proportion features (facial proportions)
            # Upper vs lower half
            upper_half = gray[:h//2, :]
            lower_half = gray[h//2:, :]
            features.append(np.mean(upper_half))
            features.append(np.mean(lower_half))
            features.append(np.std(upper_half))
            features.append(np.std(lower_half))
            
            # 5. Central region features (face center is most stable)
            center_y, center_x = h//2, w//2
            center_region = gray[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
            if center_region.size > 0:
                features.append(np.mean(center_region))
                features.append(np.std(center_region))
                features.append(np.min(center_region))
                features.append(np.max(center_region))
            
            # 6. Face quality assessment
            # Sharp edges indicate good focus
            edge_strength = np.mean(grad_magnitude)
            # Contrast indicates good lighting
            contrast = np.max(gray) - np.min(gray)
            # Symmetry indicates frontal face
            face_quality = (edge_strength * 0.4) + (contrast * 0.3) + ((1.0 - symmetry_diff) * 0.3)
            features.append(face_quality)
            
            # Pad or trim to exactly 128 dimensions
            if len(features) < 128:
                features.extend([0.0] * (128 - len(features)))
            else:
                features = features[:128]
            
            # Create deterministic encoding based purely on image content
            # Use content-based hash for consistent encoding of same image
            content_hash = hashlib.md5(str(sorted(features)).encode()).hexdigest()
            random.seed(int(content_hash[:8], 16))
            
            # Create final encoding with deterministic variations based on image content
            encoding = []
            for i, feature in enumerate(features):
                # Add deterministic variations based on feature values and position
                position_factor = 0.1 * np.sin(i * 0.314159)  # Position-based consistency
                content_factor = 0.05 * np.tanh(feature - 0.5)  # Content-based variation (bounded)
                stability_factor = 0.02 * np.cos(feature * 10)  # Additional content-based stability
                
                # Minimal random variation for same content (will be identical for same image)
                random.seed(int(content_hash[i % 32], 16) + i)
                deterministic_noise = random.uniform(-0.01, 0.01)  # Very small noise
                
                final_feature = feature + position_factor + content_factor + stability_factor + deterministic_noise
                encoding.append(float(final_feature))
                
        except Exception as e:
            logger.warning(f"Could not analyze image, using fallback method: {e}")
            # Fallback to hash-based method if image processing fails
            file_hash = hashlib.md5(image_data).hexdigest()
            random.seed(file_hash)
            encoding = [random.uniform(-1.0, 1.0) for _ in range(128)]
        
        return {
            "face_encoding": encoding,
            "has_face": True,
            "message": f"Generated consistent encoding for {file.filename}"
        }
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
        
        import numpy as np
        
        # Calculate actual distances between encodings with adaptive thresholding
        matches = []
        distances = []
        similarities = []
        logger.info(f"Starting face comparison: reference_len={len(reference_encoding)}, targets_count={len(target_encodings)}")
        
        # First pass: calculate all distances and similarities
        for i, target_encoding in enumerate(target_encodings):
            if len(target_encoding) != len(reference_encoding):
                logger.warning(f"Target encoding {i} length mismatch: {len(target_encoding)} != {len(reference_encoding)}")
                continue
                
            # Calculate Euclidean distance
            ref_array = np.array(reference_encoding)
            target_array = np.array(target_encoding)
            distance = np.linalg.norm(ref_array - target_array)
            
            # Improved normalization based on encoding dimension
            max_possible_distance = np.sqrt(len(reference_encoding) * 4)  # Theoretical max for range [-2, 2]
            normalized_distance = min(distance / max_possible_distance, 1.0)
            similarity = 1.0 - normalized_distance
            
            distances.append(distance)
            similarities.append(similarity)
        
        # Adaptive threshold based on similarity distribution
        if similarities:
            similarities_array = np.array(similarities)
            mean_sim = np.mean(similarities_array)
            std_sim = np.std(similarities_array)
            max_sim = np.max(similarities_array)
            
            # More permissive dynamic threshold system for identical images
            if max_sim > 0.85:  # Excellent match found - be more permissive for near-identical
                adaptive_threshold = max_sim - (std_sim * 2.0)
            elif max_sim > 0.70:  # Good matches - moderate selectivity
                adaptive_threshold = mean_sim + (std_sim * 0.1)
            else:  # Lower similarity overall - be very permissive
                adaptive_threshold = mean_sim - (std_sim * 0.5)
            
            # Much more permissive range for identical image detection
            adaptive_threshold = max(0.45, min(0.85, adaptive_threshold))
            
            logger.info(f"Similarity stats: mean={mean_sim:.3f}, std={std_sim:.3f}, max={max_sim:.3f}, threshold={adaptive_threshold:.3f}")
        else:
            adaptive_threshold = 0.6
        
        # Second pass: apply adaptive threshold
        for i, (distance, similarity) in enumerate(zip(distances, similarities)):
            normalized_distance = 1.0 - similarity
            is_match = similarity >= adaptive_threshold
            
            logger.info(f"Target {i}: raw_distance={distance:.4f}, similarity={similarity:.4f}, is_match={is_match} (threshold={adaptive_threshold:.3f})")
            
            matches.append({
                "index": i,
                "distance": float(normalized_distance),
                "similarity": float(similarity),
                "is_match": bool(is_match)
            })
        
        match_count = len([m for m in matches if m["is_match"]])
        
        return {
            "matches": matches,
            "message": f"Found {match_count} matches out of {len(matches)} faces"
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
        import aiohttp
        import json
        
        try:
            # Prepare headers for backend API call
            headers = {}
            if authorization:
                headers['Authorization'] = authorization
                logger.info(f"Using authorization header for backend API call: {authorization[:20]}...")
            else:
                logger.warning("No authorization token provided")
            
            async with aiohttp.ClientSession() as session:
                # Get photos for this wedding
                async with session.get(f"http://localhost:8080/api/photos/wedding/{weddingId}", headers=headers) as resp:
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
                    
                    if not target_encodings:
                        return {"matches": [], "message": "No face encodings found for photos in this wedding"}
                    
                    logger.info(f"Comparing against {len(target_encodings)} face encodings")
                    
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
                        "message": f"Found {len(matches)} matching photos"
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

@app.post("/compare-faces-form")
async def compare_faces(
    reference_image: UploadFile = File(...),
    threshold: float = Form(0.6),
    face_encodings: str = Form(...)
):
    try:
        logger.info(f"Received face comparison request with threshold: {threshold}")
        
        # Parse the face_encodings JSON (this would contain stored encodings from database)
        import json
        try:
            stored_encodings = json.loads(face_encodings)
        except json.JSONDecodeError:
            stored_encodings = []
        
        # Return dummy results for testing
        dummy_results = []
        for i, encoding in enumerate(stored_encodings):
            # Simulate some matches and some non-matches
            distance = 0.4 if i % 2 == 0 else 0.8  # Alternating matches
            is_match = distance <= threshold
            
            dummy_results.append({
                "photo_id": encoding.get("photo_id", f"dummy_{i}"),
                "distance": distance,
                "is_match": is_match
            })
        
        return {
            "matches": [r for r in dummy_results if r["is_match"]],
            "message": f"Debug mode - found {len([r for r in dummy_results if r['is_match']])} matches out of {len(dummy_results)} photos"
        }
    except Exception as e:
        logger.error(f"Error in compare_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Face Recognition Service in debug mode...")
    uvicorn.run(app, host="0.0.0.0", port=8081)