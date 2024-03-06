import numpy as np
from Feature_Extraction_AI import *
 
# Compute mean and standard deviation of combined features
mean_features = np.mean(combined_features)
std_features = np.std(combined_features)

# Set threshold as a multiple of standard deviation above the mean
threshold = mean_features + k * std_features  # Choose appropriate value for k

# Classify image based on threshold
if combined_features > threshold:
    label = "AI"
else:
    label = "non-AI"
