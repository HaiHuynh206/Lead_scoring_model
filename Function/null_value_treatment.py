import pandas as pd


def Cat_info(df, categorical_columns):
    """
    Analyzes categorical columns in a DataFrame for unique values, number of unique values,
    null values, and percentage of null values.

    Parameters:
    - df: pandas DataFrame to analyze.
    - categorical_columns: List of column names in df that are categorical.

    Returns:
    - A pandas DataFrame with columns ['columns', 'values', 'unique_values', 'null_values', 'null_percent'],
      sorted by 'null_values' in descending order.
    """
    data = []
    for column in categorical_columns:
        unique_values = df[column].unique()
        n_unique_values = df[column].nunique()
        null_values = df[column].isna().sum()
        null_percent = (null_values / len(df) * 100).round(1)
        data.append((column, unique_values, n_unique_values, null_values, null_percent))
    df_result = pd.DataFrame(data, columns=["columns", "values", "unique_values", "null_values", "null_percent"])
    df_result.sort_values("null_values", ascending=False, inplace=True)
    df_result.set_index("columns", inplace=True)
    return df_result

def Num_info(df, numeric_columns):
    """
    Analyzes numeric columns in a DataFrame for null values and percentage of null values.

    Parameters:
    - df: pandas DataFrame to analyze.
    - numeric_columns: List of column names in df that are numeric.

    Returns:
    - A pandas DataFrame with columns ['columns', 'null_values', 'null_percent'],
      sorted by 'null_values' in descending order.
    """
    data = []
    for column in numeric_columns:
        null_values = df[column].isna().sum()
        null_percent = (null_values / len(df) * 100).round(1)
        data.append({
            "columns": column,
            "null_values": null_values,
            "null_percent": null_percent
        })
    df_result = pd.DataFrame(data)
    df_result.sort_values("null_values", ascending=False, inplace=True)
    df_result.set_index("columns", inplace=True)
    return df_result