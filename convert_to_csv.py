import os
import json
import pandas as pd

# Input: Folder containing JSON files in nested structure
INPUT_DIR = r"C:\Users\shrad\TrendWeave\MERGE_OUTPUT_FINAL"  # <- Your output folder from YOLO pipeline
CSV_OUTPUT_DIR = r"C:\Users\shrad\TrendWeave\CSV_FINAL"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

def flatten_json_to_csv(json_path, csv_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    flattened_items = []
    for item in data.get('items', []):
        flat = {
            'image': data.get('image'),
            'yolo_label': item.get('yolo_label'),
            'bbox': item.get('bbox'),
            'confidence': item.get('confidence'),
            'silhouette_mask_path': item.get('silhouette_mask_path'),
        }
        # Flatten clip_attributes
        clip_attrs = item.get('clip_attributes', {})
        for k, v in clip_attrs.items():
            flat[f'{k}_label'] = v.get('label')
            flat[f'{k}_confidence'] = v.get('confidence')
        
        # Optional: add color or texture if needed
        flat['dominant_colors'] = item.get('dominant_colors')
        flat['texture_histogram'] = item.get('texture_histogram')

        flattened_items.append(flat)
    
    df = pd.DataFrame(flattened_items)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")

def process_all_jsons(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir): #walks through all nested subdirectories no matter how deep
        for file in files:
            if file.endswith("_analysis.json"):
                json_path = os.path.join(root, file)
                rel_path = os.path.relpath(json_path, input_dir)
                csv_path = os.path.join(output_dir, rel_path).replace("_analysis.json", ".csv")
                flatten_json_to_csv(json_path, csv_path)

# Run the conversion
process_all_jsons(INPUT_DIR, CSV_OUTPUT_DIR)
