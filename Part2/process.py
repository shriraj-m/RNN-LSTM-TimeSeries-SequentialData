import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

csvs = "nrel_data"
os.makedirs("averages", exist_ok=True)
'''
for year in ["2012", "2013", "2014", "2015"]:
    year_files = [os.path.join(csvs, f) for f in os.listdir(csvs) if f.endswith(f"{year}.csv")]
    dfs = [pd.read_csv(f, skiprows=2) for f in year_files]
    print(dfs[0].head())
    combined_df = pd.concat(dfs)
    final_df = combined_df.groupby(["Year", "Month", "Day", "Hour", "Minute"], as_index=False).mean()
    final_df.to_csv(f"averages/{year}_averages.csv", index=False)

xls = "Loads"

for file in os.listdir(xls):
    if file.endswith(".xls"):
        # Read the Excel file
        df = pd.read_excel(os.path.join(xls, file))
        
        # Select relevant columns
        df = df[['Hour_End', "WEST"]]
        
        # Convert 'Hour_End' to datetime
        df['Hour_End'] = pd.to_datetime(df['Hour_End'])
        
        # Round 'Hour_End' to the nearest hour
        df['Hour_End'] = df['Hour_End'].dt.round('h')
        
        # Extract Year, Month, Day, and Hour
        df['Year'] = df['Hour_End'].dt.year
        df['Month'] = df['Hour_End'].dt.month
        df['Day'] = df['Hour_End'].dt.day
        df['Hour'] = df['Hour_End'].dt.hour
        df = df.drop('Hour_End', axis=1)
        
        # Save the DataFrame as a CSV
        df.to_csv(os.path.join("averages", file.replace(".xls", ".csv")), index=False)


for file in os.listdir("averages"):
    if file.endswith("averages.csv"):
        df = pd.read_csv(os.path.join("averages", file))
        
        # Create a datetime column from Year, Month, Day, Hour, and Minute
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

        # Add 30 minutes to the datetime column
        df['datetime'] = df['datetime'] + pd.Timedelta(minutes=30)

        # Extract updated Year, Month, Day, Hour, Minute from the updated datetime column
        df['Year'] = df['datetime'].dt.year
        df['Month'] = df['datetime'].dt.month
        df['Day'] = df['datetime'].dt.day
        df['Hour'] = df['datetime'].dt.hour
        df['Minute'] = df['datetime'].dt.minute

        # Remove the datetime column as it's no longer needed
        df = df.drop('datetime', axis=1)

        # Save the updated DataFrame back to CSV
        df.to_csv(os.path.join("averages", f"adjusted_{file}"), index=False)

data_dir = "averages"
merged_dir = "merged_data"
os.makedirs(merged_dir, exist_ok=True)

for year in ["2012", "2013", "2014", "2015"]:
    weather_path = os.path.join(data_dir, f"adjusted_{year}_averages.csv")
    loads_path = os.path.join(data_dir, f"{year}_ercot_hourly_load_data.csv")
    weather_df = pd.read_csv(weather_path)
    loads_df = pd.read_csv(loads_path)

    # Merge the two DataFrames on Year, Month, Day, Hour, and Minute
    merged_df = pd.merge(weather_df, loads_df, on=["Year", "Month", "Day", "Hour"], how="inner")
    print(merged_df.head())

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(os.path.join(merged_dir, f"merged_{year}.csv"), index=False)
'''

merged_dir = "merged_data"

def handle_outliers(df, column):
    """Handle outliers in a DataFrame column using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[column].median()
    df[column] = df[column].mask((df[column] < lower_bound) | (df[column] > upper_bound), median)
    return df

df_list = []
for file in os.listdir(merged_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(merged_dir, file))
        df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

merged_df = merged_df.sort_values(by=["Year", "Month", "Day", "Hour", "Minute"])

# Handle duplicates
merged_df = merged_df.drop_duplicates(subset=["Year", "Month", "Day", "Hour", "Minute"])
# Handle NaN values using forward fill
merged_df = merged_df.ffill()
# Handle any remaining NaN values by dropping them
merged_df = merged_df.dropna()
# Handle outliers in west column
merged_df = handle_outliers(merged_df, "WEST")
# Handle outliers in temperature column
merged_df = handle_outliers(merged_df, "Temperature")
# Handle outliers in humidity column
merged_df = handle_outliers(merged_df, "Relative Humidity")

merged_df.to_csv(f"{merged_dir}/merged_all_years.csv", index=False)

normalized_df = merged_df.copy()

scaler = MinMaxScaler()
normalized_df[['Temperature', 'Relative Humidity', 'WEST']] = scaler.fit_transform(
    merged_df[['Temperature', 'Relative Humidity', 'WEST']]
)

normalized_df.to_csv(f"{merged_dir}/normalized_merged_all_years.csv", index=False)


