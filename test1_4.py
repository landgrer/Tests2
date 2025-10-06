import cv2
import numpy as np
import os

def process_image(image_path, save_path):
    # Load grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return  # skip if not an image
    
    # Step 1: Enhance contrast
    img_eq = cv2.equalizeHist(img)

    # Step 2: Adaptive threshold
    binary = cv2.adaptiveThreshold(img_eq, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   35, 5)

    # Step 3: Keep largest connected component (antler)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.uint8(labels == largest_label) * 255

    # Step 4: Fill small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=3)

    # Step 5: Smooth edges
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # Save result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, binary)


def process_folder(root_input, root_output, valid_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    for subdir, _, files in os.walk(root_input):
        for file in files:
            if file.lower().endswith(valid_exts):
                input_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(input_path, root_input)
                output_path = os.path.join(root_output, rel_path)
                process_image(input_path, output_path)
                print(f"Processed: {input_path} -> {output_path}")


# Example usage:
input_root = "Antlers"
output_root = "BAntlers"
process_folder(input_root, output_root)
