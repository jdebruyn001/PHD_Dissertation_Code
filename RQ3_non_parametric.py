import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.stats import f_oneway
from itertools import combinations

# File path to the location of the Excel file
file_path = 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/RQ3_Dataset.xlsx'

# Load the dataset into a pandas DataFrame
data = pd.read_excel(file_path)

# Rank transformation of dependent variables
data['F1_Rank'] = data.groupby('Round')['F1'].rank()
data['BLEU_Rank'] = data.groupby('Round')['BLEU'].rank()

# Perform ANOVA on ranks to simulate MANOVA
f1_anova_results = f_oneway(
    data[data['Model Description'] == 'RNN']['F1_Rank'],
    data[data['Model Description'] == 'LSTM']['F1_Rank'],
    data[data['Model Description'] == 'TNN']['F1_Rank']
)

bleu_anova_results = f_oneway(
    data[data['Model Description'] == 'RNN']['BLEU_Rank'],
    data[data['Model Description'] == 'LSTM']['BLEU_Rank'],
    data[data['Model Description'] == 'TNN']['BLEU_Rank']
)

# Function to perform a permutation test
def permutation_test(data, group_column, numeric_columns, num_permutations=10000):
    results = {}
    original_data_grouped = data.groupby(group_column)
    for col in numeric_columns:
        original_stat = original_data_grouped[col].mean().var()
        permuted_stats = np.zeros(num_permutations)
        for i in range(num_permutations):
            permuted_labels = np.random.permutation(data[group_column])
            permuted_stat = data[col].groupby(permuted_labels).mean().var()
            permuted_stats[i] = permuted_stat
        p_value = np.mean(permuted_stats >= original_stat)
        results[col] = p_value
    return results

# Perform permutation tests
permutation_results_f1 = permutation_test(data, 'Model Description', ['F1'])
permutation_results_bleu = permutation_test(data, 'Model Description', ['BLEU'])

# Print the ANOVA results
print(f'ANOVA Results for F1 Rank: statistic={f1_anova_results.statistic:.5f}, pvalue={f1_anova_results.pvalue:.5f}')
print(f'ANOVA Results for BLEU Rank: statistic={bleu_anova_results.statistic:.5f}, pvalue={bleu_anova_results.pvalue:.5f}')

# Print the permutation test results
print(f'Permutation Test Results for F1: F1={permutation_results_f1["F1"]:.5f}')
print(f'Permutation Test Results for BLEU: BLEU={permutation_results_bleu["BLEU"]:.5f}')

# Function to perform pairwise permutation tests for all pairs of groups
def pairwise_permutation_test(data, group_column, value_column, num_permutations=10000):
    unique_groups = data[group_column].unique()
    results = pd.DataFrame(index=unique_groups, columns=unique_groups, data=np.nan)

    for group1, group2 in combinations(unique_groups, 2):
        original_diff = data[data[group_column] == group1][value_column].mean() - \
                        data[data[group_column] == group2][value_column].mean()
        count_extreme_diffs = 0
        
        for _ in range(num_permutations):
            permuted_labels = np.random.permutation(data[group_column])
            permuted_diff = data[value_column].groupby(permuted_labels).mean().loc[group1] - \
                            data[value_column].groupby(permuted_labels).mean().loc[group2]
            if np.abs(permuted_diff) >= np.abs(original_diff):
                count_extreme_diffs += 1
        p_value = (count_extreme_diffs + 1) / (num_permutations + 1)
        results.loc[group1, group2] = p_value
        results.loc[group2, group1] = p_value  
    return results

# Perform pairwise permutation tests for F1 and BLEU scores
pairwise_results_f1 = pairwise_permutation_test(data, 'Model', 'F1')
pairwise_results_bleu = pairwise_permutation_test(data, 'Model', 'BLEU')

# Print the pairwise permutation test results
print("Pairwise Permutation Test Results for F1:")
print(pairwise_results_f1)
print("\nPairwise Permutation Test Results for BLEU:")
print(pairwise_results_bleu)
