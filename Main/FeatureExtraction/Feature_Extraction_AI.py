import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


folder_path = 'Dataset/Prepped_img/AI'

#---------------------------------Feature Extraction------------------------------------------------------

# Color histogram
def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
    
    return hist_hue

# Gabor Filters
def extract_gabor_features(image):
    ksize = 31  # Kernel size (odd number)
    sigma = 4   # Standard deviation of the Gaussian envelope
    theta = np.pi / 4  # Orientation of the normal to the parallel stripes of the Gabor function
    lambda_ = 10  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio
    psi = 0  # Phase offset

    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    
    return filtered_image

# Combine color histogram and Gabor features
def combine_features(color_histogram, gabor_features):
    return np.concatenate((color_histogram, gabor_features.flatten()), axis=None)

# Apply K-means clustering for spatial representation
def apply_kmeans(features, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

all_feature_vectors = []

# Load images, extract features, and accumulate feature vectors
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            
            # Extract features
            color_histogram = extract_color_histogram(image)
            gabor_features = extract_gabor_features(image)
            
            # Combine features
            feature_vector = combine_features(color_histogram, gabor_features)
            
            # Append the combined feature vector to the list
            all_feature_vectors.append(feature_vector)

# Convert the list of feature vectors to a NumPy array
all_feature_vectors = np.array(all_feature_vectors)

#---------------------------------spatial representation------------------------------------------------------

# Apply K-means clustering for spatial representation
cluster_labels = apply_kmeans(all_feature_vectors, num_clusters=5)

#---------------------------------Spike train conversion------------------------------------------------------

spike_rate = 100  # Desired spike rate in Hz
time_resolution = 0.001  # Time resolution in seconds (1 ms)
num_neurons = 100  # Number of neurons per feature dimension

# Map Features to Spike Trains
def encode_feature(feature_value):
    # Example: Rate coding (convert feature value to spike rate)
    spike_count = int(feature_value * spike_rate)
    spike_train = np.zeros(num_neurons)
    spike_train[:spike_count] = 1  # Set spikes in the spike train
    return spike_train

# Convert Features to Spike Trains
def convert_to_spike_trains(feature_vectors):
    spike_trains = []
    for feature_vector in feature_vectors:
        spike_trains_per_feature = []
        for feature_value in feature_vector:
            spike_train = encode_feature(feature_value)
            spike_trains_per_feature.append(spike_train)
        spike_trains.append(spike_trains_per_feature)
    return np.array(spike_trains)

# Convert Features to Spike Trains
spike_trains = convert_to_spike_trains(all_feature_vectors)

