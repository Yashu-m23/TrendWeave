import os
import pandas as pd

# === CONFIGURATION ===

# Root directory containing nested folders with CSV files
input_dir = r"C:\Users\shrad\TrendWeave\CSV_FINAL" # Change to your actual root folder name

# List of columns to DROP (specify the ones you don't need)
columns_to_drop = ['bbox', 'confidence','palette_confidence','neckline_confidence','sleeve_type_confidence','top_silhouette_confidence','bottom_style_confidence','outerwear_confidence','shoe_type_confidence','shoe_type_confidence','accessory_confidence','hairstyle_confidence','setting_confidence','fabric_type_confidence','dress_silhouette_confidence','bag_type_confidence','season_confidence','style_theme_label','style_theme_confidence','dominant_colors','texture_histogram']  # ← Change as needed

# Output file path for the combined CSV
combined_csv_output = 'Combined_CSV_files.csv'

# === PROCESSING ===

# List to store all loaded DataFrames
all_dataframes = []

# Walk through nested folders and find CSV files
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.csv'):
            csv_path = os.path.join(root, file)
            try:
                df = pd.read_csv(csv_path)

                # Drop unwanted columns (if they exist in the file)
                df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

                # Optionally add file source info (for traceability)
                df['source_file'] = os.path.relpath(csv_path, input_dir)

                all_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

# Combine all dataframes into one
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to output CSV
    combined_df.to_csv(combined_csv_output, index=False)
    print(f"✅ Combined CSV saved as: {combined_csv_output}")
else:
    print("⚠️ No CSV files found!")
