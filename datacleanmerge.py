import pandas as pd

# Load the dataset
path_to_file = "[RM] Water Quality-2023-12-26 151745.csv"
large_dataset = pd.read_csv(path_to_file)

# Select the important columns
important_columns = large_dataset[["StationNumber", "Station", "LatitudeDecimalDegrees",
                                   "LongitudeDecimalDegrees", "SampleDateTime",
                                   "VariableName", "MeasurementValue", "UnitCode"]]

# Define the labels you want to filter by in the VariableName column
labels = ["MICROCYSTIN, TOTAL",
          "PHOSPHORUS TOTAL (P)",
          "PHOSPHORUS TOTAL DISSOLVED",
          "TEMPERATURE AIR",
          'TEMPERATURE WATER',
          "TOTAL WATER DEPTH",
          'OXYGEN DISSOLVED (FIELD METER)',
          'OXYGEN DISSOLVED (WINKLER)',
          'OXYGEN DISSOLVED (FIELD QC)',
          'OXYGEN DISSOLVED % SATURATN.']

# Filter the dataset to include only rows where VariableName is in the list of labels
filtered_data = important_columns[important_columns["VariableName"].isin(
    labels)]

# Pivot the table so that each VariableName becomes its own column
pivoted_data = filtered_data.pivot_table(index=["StationNumber", "Station", "LatitudeDecimalDegrees",
                                                "LongitudeDecimalDegrees", "SampleDateTime"],
                                         columns="VariableName",
                                         values="MeasurementValue")

# Reset the index to flatten the DataFrame
pivoted_data = pivoted_data.reset_index()

pivoted_data["SampleDateTime"] = pd.to_datetime(
    pivoted_data["SampleDateTime"], format='%m/%d/%Y %H:%M:%S')


pivoted_data['SampleDate'] = pivoted_data['SampleDateTime'].dt.date
pivoted_data['SampleTime'] = pivoted_data['SampleDateTime'].dt.time
pivoted_data.drop(columns=['SampleDateTime'], inplace=True)

# Export the result to a new CSV file
output_path = "pivoted_water_quality_data.csv"
pivoted_data.to_csv(output_path, index=False)


# # Load the pivoted data CSV file
pivoted_data_path = "pivoted_water_quality_data.csv"
pivoted_data = pd.read_csv(pivoted_data_path)

# # Load the Excel file with cyanobacterial data
bloom_indicators_path = "all-bloom-indicators.xlsx"
bloom_data = pd.read_excel(bloom_indicators_path)

newcolums = {"Sample number": "StationNumber",
             "Waterbody name": "Station",
             "Latitude": "LatitudeDecimalDegrees",
             "Longitude": "LongitudeDecimalDegrees",
             'Collection date': 'SampleDate',
             'Collection time': 'SampleTime'}


bloom_data.rename(columns=newcolums, inplace=True)

bloom_data["Station"] = bloom_data["Station"].astype(str) + '(old)'
bloom_data["Station"] = bloom_data["Station"].str.upper()
# SampleDateTime


Col = ["StationNumber",
       "Station",
       "LatitudeDecimalDegrees",
       "LongitudeDecimalDegrees",
       'SampleDate',
       'SampleTime',
       "Total cyanobacterial cell count (cells/mL)"]

bloom_data = bloom_data[Col]

merged_data = pd.concat([pivoted_data, bloom_data], axis=0, ignore_index=True)

# # Export the merged dataset to a new CSV file
output_path = "merged_water_quality_data.csv"
merged_data.to_csv(output_path, index=False)

# # Display the merged data (optional)
# print(merged_data.head())
