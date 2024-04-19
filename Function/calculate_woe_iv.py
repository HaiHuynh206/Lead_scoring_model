import numpy as np
import pandas as pd

def calculate_woe_iv(dataset, feature, target):
    """
    Calculates the Weight of Evidence (WoE) and Information Value (IV) for a given feature in a dataset.

    Parameters:
    - dataset: A pandas DataFrame containing the data.
    - feature: The name of the feature for which WoE and IV should be calculated.
    - target: The name of the target variable, indicating the binary outcome (0 for non-event, 1 for event).

    Returns:
    - A tuple containing:
        - A pandas DataFrame with the following columns:
            - 'Value': The unique values of the feature.
            - 'All': The count of observations for each value.
            - 'Good': The count of non-event observations for each value.
            - 'Bad': The count of event observations for each value.
            - 'Distr_Good': The distribution of non-events among the values.
            - 'Distr_Bad': The distribution of events among the values.
            - 'WoE': The Weight of Evidence for each value.
            - 'IV': The Information Value contribution of each value.
        - A float representing the total Information Value for the feature.
    """
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv