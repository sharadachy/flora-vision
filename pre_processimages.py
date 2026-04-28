import cv2
import os

input_dir = "D:/data"   
output_dir = "G:\Floravision8\fullcleaned"

os.makedirs(output_dir, exist_ok=True)

resize_shape = (224, 224)


def preprocess_image(image):
    # Only resize. No color or pixel changes.
    return cv2.resize(image, resize_shape, interpolation=cv2.INTER_AREA)


for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(root, filename)

        img = cv2.imread(file_path)

        if img is None:
            print(f"⚠ Skipping corrupted: {file_path}")
            continue

        # ONLY resize — no enhancement
        processed = preprocess_image(img)

        # Maintain folder structure
        relative = os.path.relpath(root, input_dir)
        save_folder = os.path.join(output_dir, relative)
        os.makedirs(save_folder, exist_ok=True)

        # Save as PNG = lossless, preserves full quality
        base = os.path.splitext(filename)[0]
        save_path = os.path.join(save_folder, f"{base}.png")

        cv2.imwrite(save_path, processed, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 
        # 0 = absolutely no compression loss


print("preprocessing completed .")
