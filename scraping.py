import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from tqdm import tqdm
import cv2
import exifread
import json

# For video scene detection:
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Folders: update these paths as needed
pdf_folder = r"C:\Users\yasha\Downloads\wgsn_pdfs\wgsn_dataset"
image_folder = r""  # Leave blank or set to "" if not available yet
video_folder = r""  # Leave blank or set to "" if not available yet
output_folder = "extracted_data"
os.makedirs(output_folder, exist_ok=True)

# --- PDF Extraction (images in single PDF folder + single OCR text file + all image metadata in one JSON) ---
if os.path.isdir(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="PDFs processed"):
        try:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_output_dir = os.path.join(output_folder, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)

            # 1. Extract images from PDF, saving all images in the PDF's output folder
            doc = fitz.open(pdf_path)
            image_metadata_list = []
            for page_num in range(len(doc)):
                images = doc.load_page(page_num).get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    # Save all images in the PDF's output folder, use page and image index in filename
                    image_filename = os.path.join(pdf_output_dir, f"page{page_num+1}_img{img_index+1}.{image_ext}")
                    # Only extract if not already present
                    if not os.path.exists(image_filename):
                        with open(image_filename, "wb") as img_file:
                            img_file.write(image_bytes)
                    # Extract EXIF metadata if possible
                    tags_dict = {}
                    try:
                        with open(image_filename, 'rb') as imgf:
                            tags = exifread.process_file(imgf)
                            tags_dict = {str(tag): str(tags[tag]) for tag in tags.keys()}
                    except Exception:
                        pass
                    image_metadata_list.append({
                        "page": page_num + 1,
                        "image_index": img_index + 1,
                        "image_filename": os.path.relpath(image_filename, pdf_output_dir),
                        "exif_metadata": tags_dict
                    })
            doc.close()
            # Save all image metadata for this PDF in one JSON file
            all_image_meta_path = os.path.join(pdf_output_dir, "all_images_metadata.json")
            if not os.path.exists(all_image_meta_path):
                with open(all_image_meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(image_metadata_list, meta_file, indent=2)

            # 2. OCR text extraction: accumulate all text and save once
            text_file = os.path.join(pdf_output_dir, f"{pdf_name}_ocr.txt")
            if not os.path.exists(text_file):
                page_images = convert_from_path(pdf_path)
                all_text = ""
                for page_num, image in enumerate(page_images):
                    # Directly process OCR in memory, do not save the image
                    text = pytesseract.image_to_string(image)
                    text = text.replace("-\n", "")
                    all_text += f"\n\n--- Page {page_num+1} ---\n\n{text}"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(all_text)
            else:
                print(f"OCR text for {pdf_file} already exists, skipping.")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
else:
    print("No PDF folder found or provided, skipping PDF processing.")

# --- Image Processing (OpenCV + Pillow + EXIF as JSON) ---
if image_folder and os.path.isdir(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in tqdm(image_files, desc="Images processed"):
        try:
            img_path = os.path.join(image_folder, image_file)
            resized_path = os.path.join(output_folder, f"resized_{image_file}")
            gray_path = os.path.join(output_folder, f"gray_{image_file}")
            webp_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.webp")
            meta_path = os.path.join(output_folder, f"{image_file}_metadata.json")

            if not os.path.exists(resized_path):
                img = cv2.imread(img_path)
                resized = cv2.resize(img, (256, 256))
                cv2.imwrite(resized_path, resized)
            if not os.path.exists(gray_path):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(gray_path, gray)
            if not os.path.exists(webp_path):
                pil_img = Image.open(img_path)
                pil_img.save(webp_path, "WEBP")
            if not os.path.exists(meta_path):
                with open(img_path, 'rb') as f:
                    tags = exifread.process_file(f)
                    tags_dict = {str(tag): str(tags[tag]) for tag in tags.keys()}
                    with open(meta_path, "w") as meta_file:
                        json.dump(tags_dict, meta_file, indent=2)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
else:
    print("No image folder found or provided, skipping image processing.")

# --- Video Processing (Scene-based key frame extraction + ffmpeg + Whisper, metadata as JSON) ---
def extract_scene_representative_frames(video_path, output_folder, base_name):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    for idx, (start, end) in enumerate(scene_list):
        mid_frame = (start.get_frames() + end.get_frames()) // 2
        frame_filename = os.path.join(output_folder, f"{base_name}_scene_{idx+1}.jpg")
        if not os.path.exists(frame_filename):
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_filename, frame)
    cap.release()
    video_manager.release()

if video_folder and os.path.isdir(video_folder):
    whisper_model = whisper.load_model("base")  # or "small", "medium", "large"
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for video_file in tqdm(video_files, desc="Videos processed"):
        try:
            video_path = os.path.join(video_folder, video_file)
            base_name = os.path.splitext(video_file)[0]

            # 1. Scene-based key frame extraction
            extract_scene_representative_frames(video_path, output_folder, base_name)

            # 2. Extract audio with ffmpeg
            audio_filename = os.path.join(output_folder, f"{base_name}.wav")
            if not os.path.exists(audio_filename):
                (
                    ffmpeg
                    .input(video_path)
                    .output(audio_filename, acodec='pcm_s16le', ac=1, ar='16000')
                    .run(overwrite_output=True)
                )

            # 3. Transcribe audio with Whisper
            transcript_path = os.path.join(output_folder, f"{base_name}_transcript.txt")
            segments_path = os.path.join(output_folder, f"{base_name}_segments.json")
            if not os.path.exists(transcript_path) or not os.path.exists(segments_path):
                result = whisper_model.transcribe(audio_filename)
                transcript_text = result["text"]
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                with open(segments_path, "w", encoding="utf-8") as f:
                    json.dump(result["segments"], f, indent=2)

            # 4. Save video metadata as JSON
            meta_path = os.path.join(output_folder, f"{video_file}_metadata.json")
            if not os.path.exists(meta_path):
                probe = ffmpeg.probe(video_path)
                with open(meta_path, "w") as meta_file:
                    json.dump(probe, meta_file, indent=2)
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
else:
    print("No video folder found or provided, skipping video processing.")

print("All extraction and processing complete.")
