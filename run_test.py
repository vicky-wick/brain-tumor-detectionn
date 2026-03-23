import torch
from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image
import os
import json
import sys

def run_test(image_filename):
    current_dir = os.path.abspath(os.getcwd())
    model_path = current_dir
    test_image_path = os.path.join(current_dir, image_filename)
    
    if not os.path.exists(test_image_path):
        print(f"File not found: {test_image_path}")
        return

    # Load configuration
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
        id2label = config.get("id2label", {})
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = SwinForImageClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Load and process image
    image = Image.open(test_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class_idx = logits.argmax(-1).item()
    prediction = id2label.get(str(predicted_class_idx), f"Unknown ({predicted_class_idx})")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][predicted_class_idx].item()
    
    print("\n" + "="*30)
    print(f"Image: {image_filename}")
    print(f"Result: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print("="*30)
    
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs[0]):
        label = id2label.get(str(i), f"Class {i}")
        print(f"- {label}: {prob.item():.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_test(sys.argv[1])
    else:
        run_test("test_image.png")
