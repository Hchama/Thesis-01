import os
import cv2
import numpy as np

#Here we get the min/max of the image pixel value and test whether it falls under range of (0,1) to considered it as normalized or not
main_folder = 'Dataset/Prepped_img/'

# Function to iterate on the subfolders under prepped_img
def process_images(folder_path):

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            process_image(item_path)

        elif os.path.isdir(item_path):
            process_images(item_path)

def process_image(image_path):

    if image_path.endswith('.jpg') or image_path.endswith('.png'):

        normalized_image = cv2.imread(image_path)

        normalized_image = normalized_image.astype(np.float32) / 255.0

        min_val = np.min(normalized_image)
        max_val = np.max(normalized_image)

        print(f"Image: {image_path}")
        print(f"Min Pixel Value: {min_val}")
        print(f"Max Pixel Value: {max_val}")

        if min_val >= 0 and max_val <= 1:
            print("Image is normalized.")
        else:
            print("Image is not normalized.")

        print("---------------------------------------------")

process_images(main_folder)
