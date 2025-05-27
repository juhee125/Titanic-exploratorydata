#1)Import Libraries and load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.head()


#2)Summary statistics
# Basic info and summary
print(df.info())
print(df.describe(include='all'))  # Include non-numeric as well

#3)Histograms and Boxplots
# Histograms
df.hist(figsize=(12,10), bins=20)
plt.tight_layout()
plt.show()
# Boxplots for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

#4)Correlation and pairlot
# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# Pairplot (might be slow with large datasets)
sns.pairplot(df.select_dtypes(include=['float64', 'int64']), diag_kind='kde')
plt.show()

#5)Missing Values and Patterns
# Check missing values
missing = df.isnull().sum()
print(missing[missing > 0])
# Plot missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#6)Feature-level Inferences
# Survived by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.show()
# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()
