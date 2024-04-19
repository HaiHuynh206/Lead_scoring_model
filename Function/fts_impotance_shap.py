import numpy as np
import pandas as pd

def summarize_shap_values(shap_values, percentile=90, class_index=1):
    """
    Summarizes the SHAP values by calculating the mean absolute SHAP value for each feature for a specified class,
    filtering features based on a percentile threshold, and sorting them in descending order of contribution.

    Parameters:
    - shap_values: The SHAP values object, which must have 'values' with shape (samples, features, classes).
    - percentile: The percentile to use as a threshold for filtering (default is 90).
    - class_index: The index of the class to summarize SHAP values for (default is 1 for binary classification).

    Returns:
    - A DataFrame containing the filtered and sorted mean absolute SHAP values for each feature for the specified class.
    """
    # Select SHAP values for the specified class and calculate mean absolute values
    if len(shap_values.values.shape) == 3:
        mean_abs_shap_values = np.abs(shap_values.values[:, :, class_index]).mean(axis=0)
    else:
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)

    # Calculate the percentile threshold
    threshold = np.percentile(mean_abs_shap_values, percentile)

    # Create DataFrame from mean absolute SHAP values
    df_shap = pd.DataFrame(mean_abs_shap_values, index=shap_values.feature_names, columns=["SHAP Value"])

    # Filter features based on the threshold
    df_filtered = df_shap[df_shap["SHAP Value"] >= threshold]

    # Sort features by their contribution in descending order
    df_filtered_sorted = df_filtered.sort_values(by="SHAP Value", ascending=False)

    return df_filtered_sorted