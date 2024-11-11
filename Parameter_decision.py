import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2023.xlsx'
df = pd.read_excel(data_path)

# Data preprocessing: Select features 'Latitude', 'Longitude', 'Magnitude', 'Depth' and standardize them
Clus_dataSet = df[['Latitude', 'Longitude', 'DEPTH', 'Magnitude']].dropna()  # Remove missing values
Clus_dataSet = np.nan_to_num(Clus_dataSet)  # Replace all NaN values with 0
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)  

coordinates = df[['Latitude', 'Longitude']].values

# Initialization
eps_range = np.arange(0.01, 0.6, 0.05)
min_samples_range = np.arange(3, 12, 1)

best_silhouette_score = -1
best_eps = None
best_min_samples = None
realClusterNum = 0

# Create a list to store the silhouette coefficients for each eps and min_samples
silhouette_results = []

# Iterate over all combinations of eps and min_samples
for eps in eps_range:
    for min_samples in min_samples_range:
        # Clustering using DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(coordinates)
        
        # Calculate the silhouette coefficient
        if len(set(labels)) > 1:  # At least one cluster
            silhouette_avg = silhouette_score(coordinates, labels)
            
            # Get the number of clusters (ignore noise points)
            realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Print the silhouette coefficient and number of clusters corresponding to each eps and min_samples
            print(f"eps: {eps}, min_samples: {min_samples}, silhouette_score: {silhouette_avg}, "
                  f"Total number of clusters (excluding noise): {realClusterNum}, "
                  f"Total number of clusters (including noise): {len(set(labels))}")
            
            # Save the results to a list
            silhouette_results.append({'eps': eps, 
                                       'min_samples': min_samples, 
                                       'silhouette_score': silhouette_avg, 
                                       'real_cluster_num': realClusterNum,
                                       'total_clusters': len(set(labels))})

            # Update the best silhouette coefficient
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples
                realClusterNum = realClusterNum

# Create a DataFrame to save the results
silhouette_df = pd.DataFrame(silhouette_results)

output_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/silhouette_results.xlsx'
silhouette_df.to_excel(output_path, index=False)


print(f"\nBest Silhouette Score: {best_silhouette_score}")
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
print(f"Best cluster_num: {realClusterNum}")