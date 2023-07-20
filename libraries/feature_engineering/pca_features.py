import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def generate_pca_columns(dataframe:pd.DataFrame, columns:list, threshold:float):
    if columns is None:
        columns = dataframe.columns
    # Select the specified columns from the DataFrame
    selected_data = dataframe[columns]

    # Apply PCA to the selected data
    pca = PCA()
    pca.fit(selected_data)

    # Calculate the explained variance ratio for each PCA dimension
    explained_variance_ratio = pca.explained_variance_ratio_

    # Determine the number of significant PCA dimensions based on the threshold
    num_significant_dimensions = sum(explained_variance_ratio >= threshold)

    # Generate the additional columns based on the significant PCA dimensions
    pca_features = pca.transform(selected_data)[:, :num_significant_dimensions]
    pca_columns = [f'PCA_{i+1}' for i in range(num_significant_dimensions)]
    pca_dataframe = pd.DataFrame(pca_features, columns=pca_columns)

    # Concatenate the original DataFrame with the PCA DataFrame
    result_dataframe = pd.concat([dataframe, pca_dataframe], axis=1)

    return result_dataframe

def generate_pca_columns(dataframe, columns, threshold):
    # Select the specified columns from the DataFrame
    selected_data = dataframe[columns]

    # Apply PCA to the selected data
    pca = PCA()
    pca.fit(selected_data)

    # Calculate the explained variance ratio for each PCA dimension
    explained_variance_ratio = pca.explained_variance_ratio_

    # Determine the number of significant PCA dimensions based on the threshold
    num_significant_dimensions = sum(explained_variance_ratio >= threshold)

    # Generate the additional columns based on the significant PCA dimensions
    pca_features = pca.transform(selected_data)[:, :num_significant_dimensions]
    pca_columns = [f'PCA_{i+1}' for i in range(num_significant_dimensions)]
    pca_dataframe = pd.DataFrame(pca_features, columns=pca_columns)

    # Concatenate the original DataFrame with the PCA DataFrame
    result_dataframe = pd.concat([dataframe, pca_dataframe], axis=1)

    return result_dataframe

# Example usage:
# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
}
df = pd.DataFrame(data)

# Define the columns to consider for PCA
columns_to_pca = ['A', 'B', 'C']

# Set the threshold for significant PCA dimensions
pca_threshold = 0.9

# Generate additional columns based on PCA values
result_df = generate_pca_columns(df, columns_to_pca, pca_threshold)

# Print the resulting DataFrame
print(result_df)