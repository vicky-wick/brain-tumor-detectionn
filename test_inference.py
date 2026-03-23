import torch
from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image
import os
import json
import traceback

def log(msg):
    with open("test_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

# Clear old log
if os.path.exists("test_log.txt"):
    os.remove("test_log.txt")

log("Starting test script...")

try:
    current_dir = os.path.abspath(os.getcwd())
    model_path = current_dir
    test_image_path = os.path.join(current_dir, "test_image.png")
    
    log(f"Model Path: {model_path}")
    log(f"Loading config...")
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
        id2label = config.get("id2label", {})
    
    log("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    
    log("Loading model...")
    model = SwinForImageClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    log("Processing image...")
    image = Image.open(test_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    log("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class_idx = logits.argmax(-1).item()
    prediction = id2label.get(str(predicted_class_idx), f"Unknown ({predicted_class_idx})")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][predicted_class_idx].item()
    
    log("\n" + "="*20)
    log(f"Result: {prediction}")
    log(f"Confidence: {confidence:.2%}")
    log("="*20 + "\n")
    
    log("Class Probabilities:")
    for i, prob in enumerate(probs[0]):
        label = id2label.get(str(i), f"Class {i}")
        log(f"- {label}: {prob.item():.2%}")
        
except Exception as e:
    log(f"ERROR: {str(e)}")
    log(traceback.format_exc())
