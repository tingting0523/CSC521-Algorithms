import pandas as pd
import os

directory = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/rawdata'

dataframes = []

for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory, filename)
        
        df = pd.read_excel(file_path, header=1)  # header=1 is title, not read

        df.columns = ['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']  # rename column names

        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df)

# export earthquake data
output_file_path = '/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Earthquake_spread_location/earthquake_data_2024.xlsx'
combined_df.to_excel(output_file_path, index=False)
