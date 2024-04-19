import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def univariate_categorical(lead, feature, label_rotation=False, horizontal_layout=True):
    """
    Plots a univariate analysis of categorical data with respect to a binary target variable ('CONVERTED').
    Generates two subplots:
    1. A count plot showing the distribution of categories in the feature column, differentiated by 'CONVERTED' status.
    2. A bar plot showing the percentage of 'CONVERTED' leads within each category of the feature.

    Parameters:
    - lead: DataFrame containing the dataset.
    - feature: String name of the categorical column to be analyzed.
    - label_rotation: Boolean indicating whether to rotate the x-axis labels (default is False).
    - horizontal_layout: Boolean indicating the layout of the subplots. True for horizontal, False for vertical (default is True).

    Returns:
    - None: Displays the plots.
    """
    temp_count = lead[feature].value_counts()
    temp_perc = lead[feature].value_counts(normalize=True)
    df1 = pd.DataFrame({feature: temp_count.index, 'Total Leads': temp_count.values, '% Values': temp_perc.values * 100})
    print(df1)

    # Calculate the percentage of Converted=1 per category value
    cat_perc = lead[[feature, 'CONVERTED']].groupby([feature], as_index=False).mean()
    cat_perc["CONVERTED"] = cat_perc["CONVERTED"] * 100
    cat_perc.sort_values(by='CONVERTED', ascending=False, inplace=True)

    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 24))

    # Subplot 1: Count plot of categorical column
    sns.set_palette("Set2")
    sns.countplot(ax=ax1, x=feature, data=lead, hue="CONVERTED", order=cat_perc[feature], palette='Set1')
    ax1.set_title(feature, fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Black'})
    ax1.legend(["Not Converted", "Converted"])

    if(label_rotation):
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # Subplot 2: Percentage of converted leads within the categorical column
    sns.barplot(ax=ax2, x=feature, y="CONVERTED", order=cat_perc[feature], data=cat_perc, palette='Dark2')
    if(label_rotation):
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of Converted leads [%]', fontsize=15)
    plt.xlabel(feature, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " (Converted %)", fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Black'})

    plt.show()
