from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    import face_recognition
except ImportError:
    print("face_recognition not available, running without it for testing")
    face_recognition = None
import numpy as np
from PIL import Image
import io
import base64
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for face encodings (in production, use a database)
face_encodings_db = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "face-recognition"}

@app.post("/encode-face")
async def encode_face(file: UploadFile = File(...)):
    """
    Upload a face image and get face encoding
    """
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="Face recognition library not available")
        
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_array)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        if len(face_locations) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces found. Please upload an image with only one face")
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        face_encoding = face_encodings[0]
        
        # Convert numpy array to list for JSON serialization
        encoding_list = face_encoding.tolist()
        
        return {
            "success": True,
            "encoding": encoding_list,
            "face_location": face_locations[0]
        }
        
    except Exception as e:
        logger.error(f"Error in encode_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/compare-faces")
async def compare_faces(data: dict):
    """
    Compare a reference face encoding with a list of target face encodings
    Returns similarity scores
    """
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="Face recognition library not available")
        
    try:
        reference_encoding = np.array(data.get("reference_encoding"))
        target_encodings = [np.array(encoding) for encoding in data.get("target_encodings", [])]
        
        if len(target_encodings) == 0:
            return {"matches": []}
        
        # Calculate face distances (lower = more similar)
        face_distances = face_recognition.face_distance(target_encodings, reference_encoding)
        
        # Convert distances to similarity scores (higher = more similar)
        similarity_scores = 1 - face_distances
        
        # Set threshold for face matching (can be adjusted)
        threshold = 0.6
        
        matches = []
        for i, (distance, similarity) in enumerate(zip(face_distances, similarity_scores)):
            matches.append({
                "index": i,
                "distance": float(distance),
                "similarity": float(similarity),
                "is_match": float(distance) < threshold
            })
        
        return {"matches": matches}
        
    except Exception as e:
        logger.error(f"Error in compare_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

@app.post("/store-face-encoding")
async def store_face_encoding(data: dict):
    """
    Store a face encoding with an identifier
    """
    try:
        user_id = data.get("user_id")
        encoding = data.get("encoding")
        
        if not user_id or not encoding:
            raise HTTPException(status_code=400, detail="user_id and encoding are required")
        
        face_encodings_db[user_id] = encoding
        
        return {"success": True, "message": f"Face encoding stored for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Error in store_face_encoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing face encoding: {str(e)}")

@app.get("/get-face-encoding/{user_id}")
async def get_face_encoding(user_id: str):
    """
    Retrieve stored face encoding for a user
    """
    try:
        if user_id not in face_encodings_db:
            raise HTTPException(status_code=404, detail="Face encoding not found for user")
        
        return {
            "user_id": user_id,
            "encoding": face_encodings_db[user_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_face_encoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving face encoding: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)