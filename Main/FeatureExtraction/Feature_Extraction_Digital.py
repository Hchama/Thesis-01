import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'Dataset/Prepped_img/Digital'

# Color histogram
def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
    
    return hist_hue

#-----------------------------------------------------------------

# Gabor Filters
def extract_gabor_features(image):
    # Define parameters for Gabor filter
    ksize = 31  # Kernel size (odd number)
    sigma = 4   # Standard deviation of the Gaussian envelope
    theta = np.pi / 4  # Orientation of the normal to the parallel stripes of the Gabor function
    lambda_ = 10  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio
    psi = 0  # Phase offset

    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    
    return filtered_image

#-----------------------------------------------------------------



for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            
            #Call the functions
            color_histogram = extract_color_histogram(image)
            gabor_features = extract_gabor_features(image)
            
            # Combine gabor filter and color histogram values
            feature_vector = np.concatenate((color_histogram, gabor_features.flatten()), axis=None)
            combined_image = feature_vector.reshape((1, len(feature_vector)))

            print("Image:", filename)
            print("Feature Vector:", feature_vector)
            print("---------------------------------------------")
            
            # Plot color histogram
            plt.plot(color_histogram)
            plt.title('Color Histogram')
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            plt.show()

            # Plot color histogram
            plt.imshow(gabor_features, cmap='gray')
            plt.title('Gabor Filter Kernel')
            plt.axis('off')
            plt.show()

            # Visualize the combined features
            plt.imshow(combined_image, cmap='viridis', aspect='auto')
            plt.title('Combined Features')
            plt.xlabel('Feature Index')
            plt.ylabel('Image')
            plt.colorbar()
            plt.show()

