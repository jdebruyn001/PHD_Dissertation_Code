import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot
from pingouin import multivariate_normality

# Loading the data
file_path = 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/RQ3_Dataset.xlsx'
df = pd.read_excel(file_path)

# Testing two variables
variables = df[['F1', 'BLEU']]

# Perform the Multivariate Normality Test
result = multivariate_normality(variables, alpha=0.05)
print('Multivariate Normality Test Results:', result)

# Visualizing the data with a scatter plot
sns.jointplot(x='F1', y='BLEU', data=df, kind='scatter')
plt.suptitle('Scatter Plot of F1 vs BLEU')
plt.show()

# Histograms with KDE for each variable
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['F1'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Histogram of F1 with KDE')
sns.histplot(df['BLEU'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Histogram of BLEU with KDE')
plt.show()

# Q-Q Plots for each variable
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
probplot(df['F1'], dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot of F1')
probplot(df['BLEU'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of BLEU')
plt.show()