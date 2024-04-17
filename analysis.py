import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from scipy import

drive.mount('/drive')

data = pd.read_csv("http://theta.edu.pl/wp-content/uploads/2022/10/dane_projekt.csv", sep=";", decimal=",")

part = data.sample(n=150, random_state=1)
part.to_csv("/drive/My Drive/Colab Notebooks/sample_file_part.csv", index=False)

sam = data.sample(frac=0.01, random_state=1)

part['Groups'] = part['Groups'].astype("category")
part['Sex'] = part['Sex'].astype("category")
part['Anaemia'] = part['Anaemia'].astype("category")
part['Age'] = part['Age'].astype("float64")

sam = sam.fillna(0)
sam['Age'] = sam['Age'].round(0)

print('Shape:', sam.shape)
print('Null values:\n', sam.isnull().sum())
print('Duplicates:', sam.duplicated().sum())
print('Descriptive stats:\n', sam.describe(include="all"))
print('Skewness:\n', sam.skew())
print('Kurtosis:\n', sam.kurtosis().abs().sort_values())

sns.set_style("whitegrid")
fig, axes = plt.subplots(5, 3, figsize=(15, 15))
variables = ['Groups', 'Sex', 'Age', 'BMI', 'LVEF', 'GFR', 'Hb', 'Anaemia', 'Ferritin', 'Tsat', 'sTfR', 'hsCRP', 'NT.pBNP', 'UA', 'CHOL', 'SysBP']
for i, var in enumerate(variables):
    sns.histplot(sam[var], ax=axes[i//3, i%3], color='skyblue')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
for i, var in enumerate(variables[2:]):
    sns.regplot(x='Age', y=var, data=sam, ax=axes[i//3, i%3])
plt.tight_layout()
plt.show()

X = sm.add_constant(part[['Ferritin', 'hsCRP', 'NT.pBNP']])
y = part['Age']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

sam['Sex'] = sam['Sex'].replace("M", 0).replace("F", 1)
y = sam['Anaemia']
X = sm.add_constant(sam[['Groups', 'Sex', 'Age', 'BMI', 'LVEF', 'GFR', 'Ferritin', 'Tsat', 'sTfR', 'hsCRP', 'NT.pBNP', 'UA', 'CHOL', 'SysBP']])
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

test = pd.DataFrame({'numbers': [31, 45, 121, 125, 141, 150]})
test['SysBP'] = pd.cut(test['numbers'], bins=[-np.inf, 120, 140, np.inf], labels=["A", "B", "C"])
print(test)

sam['SysBP'] = pd.cut(sam['SysBP'], bins=[-np.inf, 120, 140, np.inf], labels=["A", "B", "C"])
sns.boxplot(x='SysBP', y='Age', data=sam)
plt.show()

model = ols('Age ~ SysBP', sam).fit()
print(model.summary())
