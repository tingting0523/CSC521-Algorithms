import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
 
# Set a random seed to ensure the reproducibility of the results.
np.random.seed(0)
 
# Generate three different clusters of data points using a normal distribution, where loc is the mean, scale is the standard deviation, and size is the number of points to generate
cluster_1 = np.random.normal(loc=5, scale=1, size=(15, 2))  
cluster_2 = np.random.normal(loc=15, scale=1, size=(15, 2))  
cluster_3 = np.random.normal(loc=25, scale=1, size=(15, 2)) 
 
# create some noise data
noise = np.array([[2, 3], [3, 18], [17, 4], [22, 15]])
 
# Combine all points into a single dataset.
data = np.vstack([cluster_1, cluster_2, cluster_3, noise])
 
print("Original data:",data)

# Visualize the original data points using a scatter plot.
plt.scatter(data[:, 0], data[:, 1], c='black', label='People')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('People in the Park')
plt.legend()
plt.show()   
 
# Set the parameters for the DBSCAN algorithm
epsilon = 3  # Radius ε: The neighborhood range of the points.
min_samples = 3  # MinPts: The minimum number of points that must be present in the neighborhood to form a cluster.
 
# Apply the DBSCAN algorithm: Create a DBSCAN object and apply it to the dataset.
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan.fit(data)
  
# Get the clustering results.
labels = dbscan.labels_   # -1 means noise data
 
# Visualize the clustering results: plot the noise points, and then plot points for each cluster in different colors.
plt.scatter(data[labels == -1][:, 0], data[labels == -1][:, 1], c='black', label='Noise')
for i in range(max(labels) + 1):
    plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f'Cluster {i + 1}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()
 
# Return the clustering labels
labels
