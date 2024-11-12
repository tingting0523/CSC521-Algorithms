import pandas as pd
from scipy.spatial.distance import cdist

df = pd.read_csv('/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/rawdata/eq20.csv', encoding="utf_16", sep="\t", on_bad_lines="skip")

df.columns = df.columns.str.strip()

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df_2023 = df[df['DATE'].dt.year.isin([2023])]

print(df_2023.head(5))

# Save as an Excel file
output_file_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2023.xlsx'
df_2023.to_excel(output_file_path, index=False)

print(f"File with 2022 and 2023 data saved to {output_file_path}")

# Make sure the 'Latitude' and 'Longitude' columns are floating point numbers
df_2023['Latitude'] = pd.to_numeric(df_2023['Latitude'], errors='coerce')
df_2023['Longitude'] = pd.to_numeric(df_2023['Longitude'], errors='coerce')


# Calculate Euclidean distance between points
coordinates = df_2023[['Latitude', 'Longitude']].values
distances = cdist(coordinates, coordinates)

# Exclude 0 values ​​on the diagonal (i.e. the distance from a point to itself)
distances[distances == 0] = float('inf')

# Calculate the minimum, maximum and median
min_distance = distances[distances != float('inf')].min()
max_distance = distances[distances != float('inf')].max()
median_distance = pd.DataFrame(distances[distances != float('inf')]).median().median()

print(f"Minimum distance: {min_distance:.5f}")
print(f"Maximum distance: {max_distance:.5f}")
print(f"Median distance: {median_distance:.5f}")
