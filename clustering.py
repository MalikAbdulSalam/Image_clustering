
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def read_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def extract_image_statistics(images):
    statistics = []
    for img in images:
        # Compute image statistics
        mean, std_dev = cv2.meanStdDev(img)
        stats = np.concatenate([mean, std_dev]).flatten()
        statistics.append(stats)
    return np.array(statistics)

def cluster_images(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    return kmeans.labels_, kmeans.cluster_centers_

# Define your folder path containing images
folder_path = '/home/malik/Projects/image_quantification/clustering_technique/input'

# Step 1: Read images from the folder
images, filenames = read_images_from_folder(folder_path)

# Step 2: Extract image statistics
features = extract_image_statistics(images)

# Step 3: Cluster images
num_clusters = 2  # Define the number of clusters
labels, centroids = cluster_images(features, num_clusters)

# Step 3.5: Print image names along with their corresponding clusters
for filename, label in zip(filenames, labels):
    print(f"Image: {filename}, Cluster: {label}")

# Step 4: Perform dimensionality reduction using PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Step 5: Visualize clusters in a scatter plot
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_points = reduced_features[labels == i]
    for point, filename in zip(cluster_points, np.array(filenames)[labels == i]):
        plt.scatter(point[0], point[1], label=f'Cluster {i}')
        plt.annotate(f'{filename} (Cluster {i})', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')


plt.title('Cluster Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.legend()
plt.grid(True)
plt.show()



