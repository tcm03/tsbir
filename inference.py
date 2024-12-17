import torch
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoTokenizer

import sys
sys.path.append("code")
from clip.model import CLIP

# Load Model and Utilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP.from_pretrained("tcm03/tsbir").to(device)
model.eval()

# Preprocessing Functions
from clip.clip import _transform, tokenize
transformer = _transform(model.visual.input_resolution, is_train=False)

def preprocess_image(image_base64):
    """Convert base64 encoded image to tensor."""
    image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    image = transformer(image).unsqueeze(0).to(device)
    return image

def preprocess_text(text):
    """Tokenize text query."""
    return tokenize([str(text)])[0].unsqueeze(0).to(device)

def get_fused_embedding(image_base64, text):
    """Fuse sketch and text features into a single embedding."""
    with torch.no_grad():
        # Preprocess Inputs
        image_tensor = preprocess_image(image_base64)
        text_tensor = preprocess_text(text)

        # Extract Features
        sketch_feature = model.encode_sketch(image_tensor)
        text_feature = model.encode_text(text_tensor)
        
        # Normalize Features
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        # Fuse Features
        fused_embedding = model.feature_fuse(sketch_feature, text_feature)
    return fused_embedding.cpu().numpy().tolist()

# Hugging Face Inference API Entry Point
def infer(inputs):
    """
    Inference API entry point. 
    Inputs:
      - 'image': Base64 encoded sketch image.
      - 'text': Text query.
    """
    image_base64 = inputs.get("image", "")
    text_query = inputs.get("text", "")
    if not image_base64 or not text_query:
        return {"error": "Both 'image' (base64) and 'text' are required inputs."}
    
    # Generate Fused Embedding
    fused_embedding = get_fused_embedding(image_base64, text_query)
    return {"fused_embedding": fused_embedding}
