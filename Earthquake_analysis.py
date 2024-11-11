import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import tkinter as tk
from pylab import rcParams
from sklearn.metrics import silhouette_score
from matplotlib import cm
from scipy.interpolate import griddata

data_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2023.xlsx'
df = pd.read_excel(data_path)

Clus_dataSet = df[['Latitude', 'Longitude', 'DEPTH', 'Magnitude']].dropna()  
Clus_dataSet = np.nan_to_num(Clus_dataSet) 
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)  

coordinates = df[['Latitude', 'Longitude']].values

# Clustering using DBSCAN
db = DBSCAN(eps=0.56, min_samples=9).fit(coordinates)
labels = db.labels_

# Add clustering results to the data frame
df['Clus_Db'] = labels

# Get the number of clusters (ignore noise points)
realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Total number of clusters (excluding noise): {realClusterNum}")
print(f"Total number of clusters (including noise): {len(set(labels))}")

core_points = len(db.core_sample_indices_)
noise_points = np.sum(labels == -1)
border_points = len(labels) - core_points - noise_points

print(f"Core points: {core_points}")
print(f"Border points: {border_points}")
print(f"Noise points: {noise_points}")

# Calculate the silhouette coefficient
if len(set(labels)) > 1: 
    silhouette_avg = silhouette_score(coordinates, labels)
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("Silhouette Score: Cannot be computed due to lack of clusters")

# Calculate the maximum and minimum latitude and longitude ranges displayed on the map
max_lat = df['Latitude'].max() + 1
min_lat = df['Latitude'].min() - 1
max_lon = df['Longitude'].max() + 1
min_lon = df['Longitude'].min() - 1

# Filter the data to ensure that the coordinates are within the display range
df = df[(df['Longitude'] > min_lon) & (df['Longitude'] < max_lon) & (df['Latitude'] > min_lat) & (df['Latitude'] < max_lat)]

# Get the screen size and calculate the appropriate canvas size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Resize the canvas to 80% of the screen size
fig_width = screen_width * 0.8 / 100  # Convert screen width to inches
fig_height = screen_height * 0.8 / 100  # Convert screen height to inches

# Set drawing parameters
rcParams['figure.figsize'] = (fig_width, fig_height)

# Create a map (2 rows and 2 columns layout)
fig, ax = plt.subplots(2, 2) 

# Plot 1: the location of all points
my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=min_lon, llcrnrlat=min_lat,
                 urcrnrlon=max_lon, urcrnrlat=max_lat, ax=ax[0, 0])

# Draw coastlines and country borders
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

for i in range(len(df)):
    lat = df.iloc[i]['Latitude']
    lon = df.iloc[i]['Longitude']
    x, y = my_map(lon, lat)
    my_map.scatter(x, y, color='blue', marker='o', s=10, alpha=0.85)

ax[0, 0].set_title("All Points in Map")

# Plot 2: core, boundary, noise points
my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=min_lon, llcrnrlat=min_lat,
                 urcrnrlon=max_lon, urcrnrlat=max_lat, ax=ax[0, 1])

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

# Assign colors to each cluster
colors_core = 'blue'   # Core points
colors_border = 'green'  # Border points
colors_noise = 'gray'   # Noise points

# Plotting different types of points
for i in range(len(df)):
    lat = df.iloc[i]['Latitude']
    lon = df.iloc[i]['Longitude']
    label = labels[i]

    # Get map coordinates
    x, y = my_map(lon, lat)

    if label == -1:  # Noise points
        my_map.scatter(x, y, color=colors_noise, marker='o', s=10, alpha=0.85)
    elif i in db.core_sample_indices_:  # Core points
        my_map.scatter(x, y, color=colors_core, marker='o', s=10, alpha=0.85)
    else:  # Border points
        my_map.scatter(x, y, color=colors_border, marker='o', s=10, alpha=0.85)

# Add legend
core_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_core, markersize=10, label='Core Points')
border_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_border, markersize=10, label='Border Points')
noise_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_noise, markersize=10, label='Noise Points')

ax[0, 1].legend(handles=[core_patch, border_patch, noise_patch], loc='lower left')

ax[0, 1].set_title("Core, Border, Noise Points")

# Plot 3: clusters
my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=min_lon, llcrnrlat=min_lat,
                 urcrnrlon=max_lon, urcrnrlat=max_lat, ax=ax[1, 0])

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

# Color Mapping: Assign a different color to each cluster
unique_labels = set(labels)
cluster_colors = plt.cm.get_cmap('tab20', len(unique_labels)) 

# Plot the points for each cluster
for i in range(len(df)):
    lat = df.iloc[i]['Latitude']
    lon = df.iloc[i]['Longitude']
    label = labels[i]

    x, y = my_map(lon, lat)
    
    if label != -1:  # Eliminate noise points
        color = cluster_colors(label)
        my_map.scatter(x, y, color=color, marker='o', s=10, alpha=0.85)

# Display the number of clusters
ax[1, 0].text(0.5, 0.95, f'Number of Clusters (excluding noise): {realClusterNum}', 
           horizontalalignment='center', verticalalignment='center', 
           transform=ax[1, 0].transAxes, fontsize=12, color='black', weight='bold')

ax[1, 0].set_title("Clusters by Color")


#Plot 4:Bouguer gravity anomaly map
# Creating a Grid
grid_lon, grid_lat = np.meshgrid(
    np.linspace(min_lon, max_lon, 500),
    np.linspace(min_lat, max_lat, 500)
)

# calculate Bouguer Anomaly value
df['Bouguer_Anomaly'] = df['Magnitude'] * (df['DEPTH'] / 10)

# Interpolate outliers onto a grid
grid_anomaly = griddata(
    (df['Longitude'], df['Latitude']),
    df['Bouguer_Anomaly'],
    (grid_lon, grid_lat),
    method='cubic'
)

fig, ax = plt.subplots(figsize=(12, 8))
my_map = Basemap(projection='merc', resolution='l', area_thresh=1000.0,
                 llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon, urcrnrlat=max_lat, ax=ax)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

# Plot the Bouguer gravity anomaly on the map, with colors ranging from green to red representing low to high outliers
colormap = cm.get_cmap('RdYlGn_r')  # Gradient color map from green to yellow to red
x, y = my_map(grid_lon, grid_lat)
anomaly_plot = my_map.pcolormesh(x, y, grid_anomaly, shading='auto', cmap=colormap)

# Add color bar
cbar = my_map.colorbar(anomaly_plot, location='right', pad="5%")
cbar.set_label('Bouguer Gravity Anomaly')


plt.tight_layout()
plt.show()
