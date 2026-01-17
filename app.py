


"""
AI Ethics Radar - Flask Application
Two-stage classification: Main Category → AI Ethics (if AI)
"""

from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
import json
import pickle
from transformers import AutoTokenizer
import os

app = Flask(__name__)

# ============================================
# LOAD MODELS AND RESOURCES
# ============================================

print("Loading models...")

# Stage 1: Main Category Classification (TF-IDF)
STAGE1_MODEL_PATH = 'onnx_models/stage1_tfidf/model.pkl'
STAGE1_VECTORIZER_PATH = 'onnx_models/stage1_tfidf/vectorizer.pkl'
STAGE1_LABELS_PATH = 'onnx_models/stage1_tfidf/labels.json'

# Load Stage 1 TF-IDF model
with open(STAGE1_MODEL_PATH, 'rb') as f:
    stage1_model = pickle.load(f)

with open(STAGE1_VECTORIZER_PATH, 'rb') as f:
    stage1_vectorizer = pickle.load(f)

# --- FIXED SECTION STARTS HERE ---
# Load Stage 1 Labels (Handling List format instead of Dictionary)
with open(STAGE1_LABELS_PATH, 'r') as f:
    stage1_raw_labels = json.load(f)

# Manually create the ID-to-Label dictionary from the list
stage1_id2label = {}
for idx, label in enumerate(stage1_raw_labels):
    # Convert "stage1_binary_artificial_intelligence" -> "Artificial Intelligence"
    clean_label = label.replace('stage1_binary_', '').replace('_', ' ').title()
    stage1_id2label[idx] = clean_label
# --- FIXED SECTION ENDS HERE ---

print("✓ Stage 1 (Main Category) model loaded")

# Stage 2: AI Ethics Multi-label Classification (ONNX)
STAGE2_MODEL_PATH = 'onnx_models/stage2_best_model_quantized.onnx'
STAGE2_TOKENIZER_PATH = 'stage2_best_model_tokenizer'

# Load Stage 2 ONNX model
stage2_session = ort.InferenceSession(STAGE2_MODEL_PATH)
stage2_tokenizer = AutoTokenizer.from_pretrained(STAGE2_TOKENIZER_PATH)

# Ethics labels
ETHICS_LABELS = [
    'Bias & Fairness',
    'Privacy & Data Misuse',
    'Job Displacement',
    'Misinformation & Deepfakes',
    'Accountability & Liability',
    'Environmental Impact',
    'Surveillance & Regulation',
    'Safety & Security',
    'Intellectual Property',
    'Lack of Transparency',
    'Algorithmic Manipulation',
    'No Ethical Issue Detected'
]

print("✓ Stage 2 (AI Ethics) model loaded")
print("✓ All models ready!")

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_main_category(text):
    """
    Stage 1: Predict main category using TF-IDF + LogReg
    Adapted for Multi-label: Picks the single category with the highest probability.
    Returns: (category_name, confidence_score)
    """
    # Vectorize text
    X = stage1_vectorizer.transform([text])
    
    # Get probabilities for all classes
    # resulting shape is usually (1, n_classes)
    probabilities = stage1_model.predict_proba(X)[0]
    
    # --- FIX FOR MULTI-LABEL ---
    # Instead of looking at the binary .predict() output (which might have 0 or multiple 1s),
    # we take the index of the HIGHEST probability score.
    prediction_idx = int(np.argmax(probabilities))
    # ---------------------------
    
    # Get category name using the index of the max score
    category = stage1_id2label[prediction_idx]
    
    # Get the confidence score for that winning category
    confidence = float(probabilities[prediction_idx])
    
    return category, confidence

def predict_ethics(text):
    """
    Stage 2: Predict AI ethics issues using ONNX model
    Returns: dict with label: probability pairs
    """
    # Tokenize
    inputs = stage2_tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='np'
    )
    
    # Run inference
    onnx_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    logits = stage2_session.run(None, onnx_inputs)[0]
    
    # Apply sigmoid to get probabilities
    probabilities = 1 / (1 + np.exp(-logits[0]))
    
    # Create result dict
    results = {}
    for label, prob in zip(ETHICS_LABELS, probabilities):
        results[label] = float(prob)
    
    return results

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main endpoint: Analyze article text
    Returns JSON with category and ethics predictions
    """
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) < 10:
            return jsonify({
                'error': 'Please provide valid article text (at least 10 characters)'
            }), 400
        
        # Stage 1: Predict main category
        category, confidence = predict_main_category(text)
        
        result = {
            'category': category,
            'confidence': confidence,
            'ethics': None
        }
        
        # Stage 2: If AI category, predict ethics
        if category == 'Artificial Intelligence':
            ethics_results = predict_ethics(text)
            result['ethics'] = ethics_results
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True
    })

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)