import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_csv("merged_data/merged_all_years.csv")

# Combine the Year, Month, and Day into a single datetime column
df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Get a list of unique days (in datetime format) from the data
unique_days = df['DateTime'].dt.date.unique()

# Randomly select a day
random_day = random.choice(unique_days)
print(f"Randomly selected day: {random_day}")

# Filter data for the randomly selected day
df_day = df[df['DateTime'].dt.date == random_day]  # Only include data for that day

# Sort the data by Time (for continuous plotting)
df_day_sorted = df_day.sort_values(by='DateTime')

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot Temperature over the course of the day
axs[0].plot(df_day_sorted['DateTime'], df_day_sorted['Temperature'], marker='o')
axs[0].set_title(f"Temperature for {random_day}")
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Temperature (°C)')
axs[0].tick_params(axis='x', rotation=45)

# Plot Humidity over the course of the day
axs[1].plot(df_day_sorted['DateTime'], df_day_sorted['Relative Humidity'], marker='o', color='orange')
axs[1].set_title(f"Relative Humidity for {random_day}")
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Relative Humidity (%)')
axs[1].tick_params(axis='x', rotation=45)

# Plot Load (WEST) over the course of the day
axs[2].plot(df_day_sorted['DateTime'], df_day_sorted['WEST'], marker='o', color='green')
axs[2].set_title(f"Load (WEST) for {random_day}")
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Load (WEST)')
axs[2].tick_params(axis='x', rotation=45)

# Format the x-axis to show time only (ignore the date)
axs[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as a PNG file
output_file = f"weather_plots_{random_day}.png"
plt.savefig(output_file)