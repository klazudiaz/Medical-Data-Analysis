# Medical-Data-Analysis
This repository contains a Python script for analyzing medical data related to cardiovascular health. The script performs various data preprocessing steps, descriptive statistics, visualization, and statistical modeling techniques to gain insights into the dataset.

# Features:
- Data Import and Sampling: The script imports medical data from a CSV file hosted online and samples a subset for analysis.
- Data Preprocessing: It preprocesses the data by handling missing values, converting data types, and rounding numerical values.
- Descriptive Statistics: The script computes descriptive statistics such as mean, median, skewness, kurtosis, and checks for null values and duplicates.
- Visualization: It creates histograms and regression plots to visualize the distribution and relationships between variables.
- Linear Regression: The script performs linear regression analysis to explore the relationship between selected predictor variables (e.g., Ferritin, hsCRP, NT.pBNP) and the age of patients.
- Logistic Regression: It conducts logistic regression to predict the likelihood of patients having anaemia based on various clinical parameters.
- Categorical Variable Creation: The script creates a categorical variable based on the systolic blood pressure (SysBP) levels and visualizes its relationship with age using boxplots.
- ANOVA: An analysis of variance (ANOVA) is performed to assess the effect of SysBP categories on the age of patients.

# Dependencies:
- Python 3
- Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, scipy
