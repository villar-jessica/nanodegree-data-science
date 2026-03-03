# Gender equality in leadership: A data-driven analysis in Latin America

## Project overview
This project investigates the macroeconomic, social, and educational factors that influence female representation in leadership (management) positions in 17 Latin American countries, covering the period from 2015 to 2024. The analysis seeks to answer essential business questions about the impact of education and job quality on women's career advancement.

This project was developed as part of the requirements for completing the **Udacity Data Scientist Nanodegree**.

The article about the project can be find in [this link](https://medium.com/@jessicavillar679/glass-ceiling-or-leaky-pipeline-a-data-driven-analysis-of-female-leadership-in-latin-america-97351edac1b4).

## Libraries used
The project was developed in Python and requires the following libraries for code execution, data manipulation, visualization, and predictive modeling:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `openpyxl` (necessary to read the data file in Excel format)

## Project and data files
The repository contains the following main files:

1. **`Project 1.ipynb`:** This is the main Jupyter Notebook. It contains the exploratory data analysis (EDA), visualizations, formulation and answers to 4 fundamental business questions, and a comparison of the performance of Machine Learning models.

2. **`gender_equality_analyzer.py`:** Python module containing the `GenderEqualityAnalyzer` class. This script was created to modularize and organize the code. It contains the entire pipeline for extraction, preprocessing (cleaning, null data imputation, wide/long format transformation), and predictive model training (Linear Regression, Random Forest, Gradient Boosting).

3. **Data files:** The source file `p_data_extract_from_gender_statistics.xlsx` contains gender statistics data. The purpose of this data is to provide the basic indicators (such as school enrollment, unemployment, labor force participation, and GDP) that serve as independent variables (*features*) to predict and understand female leadership rates (*target*).

## Summary of results
The analysis revealed important and counterintuitive insights:
* Educational factors (such as secondary and higher education enrollment) have a negative relationship with the proportion of women in leadership. This indicates a "leaky funnel" phenomenon, where education does not translate into promotion.

* Purely economic factors, such as GDP per capita and the Gini Index, do not have a significant impact on gender equality in corporate management positions.

* Formal salaried employment is the best way to guarantee access to leadership positions, contrasting with mere general participation in the workforce, which often masks informality.

* The Gradient Boosting model achieved almost 89% predictive capacity on leadership rates using these indicators.

## Data sources and attributions
All data used in this analysis are in the public domain and were extracted from the **Gender Statistics** database of the World Bank (https://databank.worldbank.org/source/gender-statistics).
