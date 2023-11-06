import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('city_day.csv')

# Explore the first few rows of the dataset
print(df.head())

# Basic info about the dataset
print(df.info())

# Statistical summary of the dataset
print(df.describe())

# Handling missing values - There are multiple strategies for this, such as:
# - Dropping rows with missing values.
# - Filling missing values with mean/median/mode.
# - Using an algorithm to predict missing values.

# Here, we'll fill missing values with the mean for simplicity.
df.fillna(df.mean(), inplace=True)

# Check for any remaining null values
print(df.isnull().sum())

# If there are categorical variables, we would encode them using either Label Encoding or One-Hot Encoding
# However, since we're focusing on pre-processing for now, we'll assume all features are numerical.

# Let's assume we are only interested in a subset of all the available features.
# We'll choose the features based on the data understanding step we did earlier.
selected_features = ['PM2.5', 'NO2', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'AQI', 'AQI_Bucket']

# Filter the dataset for selected features
df_selected = df[selected_features]

# Normalization or Standardization - As an example, we'll standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

# Check the standardized data
print(df_standardized.head())

# Now, the data is preprocessed and ready to be used for building a model.
