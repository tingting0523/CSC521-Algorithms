import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

data_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2023.xlsx'
df = pd.read_excel(data_path)

# Selecting longitude and latitude coordinates in earthquake data as clustering features
coordinates = df[['Latitude', 'Longitude']].dropna().values

# Set the k value (usually the value of min_samples): k is 2*dimension-1. (where dimension is the number of features)
k = 3  

# Use NearestNeighbors to calculate the k distance of each point
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(coordinates)
distances, indices = neigh.kneighbors(coordinates)

# Take the distance from each point to its kth neighbor and sort in descending order
k_distances = np.sort(distances[:, k-1])[::-1]

# Draw the k-distance diagram
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances, marker='o', linestyle='-')
plt.xlabel("Points sorted by distance")
plt.ylabel(f"Distance to {k}th nearest neighbor")
plt.title(f"{k}-Distance Graph for Determining Optimal eps")
plt.grid()
plt.show()
