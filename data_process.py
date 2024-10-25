import pandas as pd
import os
from scipy.spatial.distance import cdist

directory = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/rawdata'

dataframes = []

# 读取所有 Excel 文件并合并
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory, filename)
        
        df = pd.read_excel(file_path, header=1)  # header=1 is title, not read
        df.columns = ['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']  # rename column names

        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)


# Calculate the nearest Euclidean distance for each point
coordinates = combined_df[['Latitude', 'Longitude']].values

distances = cdist(coordinates, coordinates)

# Find the closest distance to each point (exclude yourself)
for i in range(len(distances)):
    distances[i, i] = float('inf')  # Exclude your own distance

min_distances = distances.min(axis=1)  # Find the minimum value for each row

# Leave the minimum distance to two decimal places and add to the new column
combined_df['DIST'] = min_distances.round(5)

print(combined_df)

output_file_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2024.xlsx'
combined_df.to_excel(output_file_path, index=False)

