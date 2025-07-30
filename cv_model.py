import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import mediapipe as mp
os.environ["CURL_CA_BUNDLE"] = ""


# --- CONFIGURATION ---


# Set device for computation (uses CUDA if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Directory containing (possibly nested) images to process
IMAGE_DIR = r"path\to\pdf"


# Directory to save output JSON files
OUTPUT_DIR = "cv_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist


# --- LOAD MODELS ---


print("Loading YOLOv8 segmentation model (auto-downloads weights if not present)...")
yolo = YOLO("yolov8n-seg.pt")  # NOTE: use YOLOv8 SEGMENTATION model!


print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)  # CLIP for vision-language embedding
print("CLIP model loaded.")
print("Loading CLIP processor...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
print("CLIP processor loaded.")


print("Loading MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)  # For pose estimation


# --- FASHION ATTRIBUTE PROMPTS ---


FEATURE_PROMPTS_FEMALE = {
    "palette": ["pastel", "neon", "monochrome", "earth tones", "bold colors"],
    "dress_silhouette": ["bodycon dress", "fit and flare", "wrap dress", "maxi dress", "shirt dress"],
    "neckline": ["v-neck", "square neckline", "halter neck", "off-shoulder", "sweetheart"],
    "sleeve_type": ["sleeveless", "cap sleeves", "puffed sleeves", "bell sleeves", "long sleeves"],
    "top_silhouette": ["crop top", "peplum", "wrap top", "fitted blouse", "camisole"],
    "bottom_style": ["mini skirt", "culottes", "flared pants", "wide-leg pants", "denim shorts"],
    "outerwear": ["denim jacket", "blazer", "cardigan", "long coat", "shrug"],
    "shoe_type": ["heels", "ballet flats", "ankle boots", "strappy sandals", "platform sneakers"],
    "bag_type": ["tote bag", "clutch", "mini backpack", "crossbody bag", "shoulder bag"],
    "accessory": ["hoop earrings", "sunglasses", "hairband", "watch", "scarf"],
    "hairstyle": ["high bun", "loose curls", "braids", "straight hair", "ponytail"],
    "setting": ["beach casual", "party night", "brunch look", "street style", "office wear"],
    "fabric_type": ["satin", "chiffon", "linen", "cotton", "organza"],
    "season": ["summer", "winter", "spring", "fall"],
    "style_theme": ["boho", "y2k", "minimalist", "vintage", "elegant chic"]
}
FEATURE_PROMPTS_MALE = {
    "palette": ["monochrome", "earth tones", "muted tones", "navy and grey", "bold colors"],
    "dress_silhouette": [],
    "neckline": ["crew neck", "v-neck", "polo", "buttoned collar", "mandarin collar"],
    "sleeve_type": ["short sleeves", "rolled sleeves", "long sleeves", "sleeveless", "half sleeves"],
    "top_silhouette": ["oversized tee", "fitted t-shirt", "button-up shirt", "hoodie", "tank top"],
    "bottom_style": ["chinos", "joggers", "denim jeans", "cargo shorts", "tailored trousers"],
    "outerwear": ["leather jacket", "bomber jacket", "blazer", "hoodie", "denim jacket"],
    "shoe_type": ["sneakers", "loafers", "derby shoes", "slip-ons", "boots"],
    "bag_type": ["messenger bag", "backpack", "duffel bag", "crossbody bag"],
    "accessory": ["watch", "cap", "chain", "sunglasses", "bracelet"],
    "hairstyle": ["undercut", "slick back", "buzz cut", "curly top", "messy hair"],
    "setting": ["street casual", "gym wear", "date night", "beach look", "smart casual"],
    "fabric_type": ["denim", "cotton", "linen", "polyester", "wool"],
    "season": ["summer", "winter", "spring", "monsoon"],
    "style_theme": ["streetwear", "minimalist", "grunge", "classic", "sporty"]
}
FEATURE_PROMPTS_UNISEX = {
    "palette": ["black & white", "pastel", "bold primary colors", "earth tones", "neon"],
    "shoe_type": ["sneakers", "sandals", "slip-ons", "boots", "canvas shoes"],
    "bag_type": ["crossbody bag", "backpack", "tote bag", "belt bag", "messenger bag"],
    "accessory": ["watch", "sunglasses", "bracelet", "ring", "hat"],
    "setting": ["streetwear", "travel outfit", "festival wear", "college casual", "lounge"],
    "fabric_type": ["denim", "cotton", "linen", "nylon", "jersey"],
    "season": ["summer", "winter", "rainy", "fall"],
    "style_theme": ["minimalist", "y2k", "boho", "athleisure", "vintage"]
}


# --- CORE FUNCTIONS ---


def detect_with_yolov8(image_path):
    """
    Detects objects in the image using YOLOv8 SEGMENTATION and returns bounding boxes, labels, and masks.
    """
    results = yolo(image_path)
    detections = []
    names = results[0].names
    boxes = results[0].boxes
    masks = results[0].masks  # Segmentation masks
    for i, box in enumerate(boxes):
        label = names[int(box.cls)]
        conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()
        # Get corresponding segmentation mask (as list)
        mask = None
        if masks is not None:
            mask = masks.data[i].cpu().numpy().tolist()  # binary mask per object
        detections.append({"label": label, "bbox": bbox, "confidence": conf, "mask": mask})
    return detections


def crop_image(image_path, bbox):
    """
    Crops the image using the given bounding box coordinates.
    """
    image = Image.open(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))


def detect_gender(pil_img):
    """
    Predicts gender as 'man', 'woman', or 'unisex' using CLIP; returns label and confidence.
    """
    prompts = ["a man wearing modern clothes", "a woman in stylish clothes", "unisex clothing"]
    inputs = clip_processor(text=prompts, images=pil_img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy().flatten()
    return prompts[np.argmax(probs)], float(np.max(probs))


def clip_attributes(pil_img, feature_prompts):
    """
    For each fashion feature category, returns the best-matching prompt and confidence using CLIP.
    """
    outputs = {}
    image_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    for category, labels in feature_prompts.items():
        if not labels:
            continue
        text_inputs = clip_processor(text=labels, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        best_idx = int(np.argmax(similarity))
        best_label = labels[best_idx]
        outputs[category] = {"label": best_label, "confidence": float(similarity[best_idx])}
    return outputs


def analyze_color_texture(pil_img):
    """
    Extracts dominant colors (using KMeans in LAB space) and a simple texture histogram from the image.
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray], [0], None, [256], [0,256])
    return dominant_colors.tolist(), lbp.flatten().tolist()


def detect_pose(image_path):
    """
    Detects human pose keypoints using MediaPipe Pose.
    Returns a list of keypoints or None if no pose is detected.
    """
    image = cv2.imread(image_path)
    results = pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
    return keypoints


def save_mask(mask_data, save_path):
    """
    Saves a binary (0/1) mask as a PNG image file.
    """
    mask_np = np.array(mask_data, dtype=np.uint8) * 255
    mask_img = Image.fromarray(mask_np)
    mask_img.save(save_path)


def process_image(image_path):
    """
    Runs the full pipeline on a single image:
    - Detects objects with segmentation
    - Crops and analyzes each detected object for attributes, color, texture
    - Saves a silhouette mask PNG for each object
    - Optionally detects pose
    - Saves results as a JSON file in the output directory
    """
    # Use relative path for unique output filenames (handles nested folders)
    rel_path = os.path.relpath(image_path, IMAGE_DIR)
    safe_name = rel_path.replace(os.sep, "__").replace("/", "__")
    json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(safe_name)[0]}_analysis.json")


    result = {"image": image_path, "items": []}


    # Detect objects in the image with segmentation
    yolo_detections = detect_with_yolov8(image_path)


    # For each detected item, extract visual and semantic attributes
    for idx, det in enumerate(yolo_detections):
        bbox = det["bbox"]
        cropped = crop_image(image_path, bbox)
        # Gender-aware prompt selection and multi-attribute CLIP analysis
        gender_label, _ = detect_gender(cropped)
        if "woman" in gender_label:
            prompt_set = FEATURE_PROMPTS_FEMALE
        elif "man" in gender_label:
            prompt_set = FEATURE_PROMPTS_MALE
        else:
            prompt_set = FEATURE_PROMPTS_UNISEX
        clip_attrs = clip_attributes(cropped, prompt_set)
        dominant_colors, texture_hist = analyze_color_texture(cropped)
        # Save silhouette mask image if detection has mask
        mask = det.get("mask")
        silhouette_mask_path = None
        if mask is not None:
            silhouette_mask_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(safe_name)[0]}_mask_{idx}.png")
            save_mask(mask, silhouette_mask_path)
        item = {
            "yolo_label": det["label"],
            "bbox": bbox,
            "confidence": det["confidence"],
            "clip_attributes": clip_attrs,
            "dominant_colors": dominant_colors,
            "texture_histogram": texture_hist,
            "silhouette_mask_path": silhouette_mask_path
        }
        result["items"].append(item)


    # Optionally detect pose for the whole image
    pose_keypoints = detect_pose(image_path)
    result["pose_keypoints"] = pose_keypoints


    # Save the results as a JSON file
    print(f"Saving output for {image_path}")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---- BATCH PIPELINE ----


def batch_process_images(image_dir):
    """
    Recursively finds and processes all images in the given directory and its subdirectories.
    """
    image_files = []
    # Walk through all subdirectories to find images
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} images.")
    # Process each image and handle errors gracefully
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            process_image(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


# ---- DISPLAY SUMMARY FUNCTION ----

def display_summary(json_data):
    """
    Visualizes a processed image with bounding boxes and feature labels.
    """
    # Load image
    image_path = json_data["image"]
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Draw bounding boxes for each detected item
    for item in json_data["items"]:
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 6, f"{item['yolo_label']}", color='lime', fontsize=11, weight='bold')
    
    # Prepare summary text for each detected item
    summary_lines = []
    for idx, item in enumerate(json_data["items"]):
        summary_lines.append(f"Item {idx+1}: {item['yolo_label']} (Confidence {item['confidence']:.2f})")
        for k, v in item["clip_attributes"].items():
            summary_lines.append(f"     {k.replace('_',' ').title()}: {v['label']} ({v['confidence']:.2f})")
        summary_lines.append("")  # Blank line between items
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gcf().text(0.75, 0.04, "\n".join(summary_lines), fontsize=10, verticalalignment='bottom', bbox=props)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the batch processing pipeline
    batch_process_images(IMAGE_DIR)
    print("All images processed. Results saved to", OUTPUT_DIR)

    # Optional: Interactive visualization after processing
    import glob
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*_analysis.json"))
    if json_files:
        print("\nVisualization mode: Select a JSON output file to visualize.")
        for idx, fname in enumerate(json_files):
            print(f"{idx}: {os.path.basename(fname)}")
        sel = input("Enter the index of the image to display (or blank to exit): ").strip()
        if sel:
            idx = int(sel)
            with open(json_files[idx], "r") as f:
                data = json.load(f)
            display_summary(data)
