import os
import cv2
import numpy as np

folder_path = 'Dataset/digitalSamples/'

#resizing image size
new_width = 224
new_height = 224
save_folder = 'Dataset/Prepped_img/Digital/'


if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#normalization and resizing
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        resized_image = cv2.resize(image, (new_width, new_height))

        normalized_image = resized_image.astype(np.float32) / 255.0

        save_path = os.path.join(save_folder, filename)
        cv2.imwrite(save_path, normalized_image * 255.0)

   
print(f'Images converted Succesfully')  

