![image](https://www.linkpicture.com/q/istockphoto-1390716712-170667a.jpg)

# PySpark and Machine Learning Project ✨

The PySpark and Machine Learning project focuses on analyzing a dataset using PySpark, a fast and distributed data processing framework. 

- It combines the power of PySpark with various popular libraries to perform data analysis, preprocessing, visualization, and train a Gradient Boosted Tree model for prediction.

- This project aims to leverage the power of PySpark, along with various data manipulation and visualization libraries such as NumPy, Pandas, Seaborn, and Matplotlib, to perform data analysis and build a machine learning model using the Gradient Boosted Tree (GBT) algorithm.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Analysis](#data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)


## Introduction

The PySpark and  Machine Learning project is designed to analyze a dataset containing information about 100 million employees, with a primary focus on their salaries. 

 - The project utilizes the PySpark framework for data analysis and employs SQL queries to extract valuable insights. Furthermore, PySpark's MLlib is utilized to conduct Feature Engineering, a process that enhances the input data, and trains a Gradient Boosted Tree (GBT) algorithm. 

 - The GBT model is then utilized to predict the salaries of the employees based on the provided features. Overall, the project combines the power of PySpark and machine learning techniques to gain insights into employee data and predict their salaries accurately.

## Installation
To run this project, ensure you have the following dependencies installed:
- PySpark
- NumPy
- Pandas
- Seaborn
- Matplotlib

You can install these dependencies using `pip`:
```bash
pip install pyspark numpy pandas seaborn matplotlib
```

## Data Analysis
The project starts with exploratory data analysis using PySpark and other libraries. It involves loading the dataset, understanding its structure, and performing basic data profiling. You can find this analysis in the notebook or script file.

- The industry with highest income is OIL, second is FINANCE and third is WEB.

- For every jobtype, workers with higher degree have a higher income.

- The highest paid job is CEO, while the least paid is      JANITOR.For every jobtype, Engineering, Businees and Math are the majors which lead to higher income.

- The average income of workers with no experience differs a lot among the jobtype:
   CEO has a base average income around 120k$, CFO around   110k\$ while Janitor only 47k$.

- Among all jobs, mean, median and mode salary are respectively 116k$, 114k\$ and 108k$. They do not coincide due to the right skeweness of the salary distribution.

## Feature Engineering
In the feature engineering phase, various data preprocessing techniques are applied to enhance the quality of the dataset. This may involve handling missing values, encoding categorical variables, scaling numerical features, and creating new derived features.

## Model Training and Evaluation
The machine learning model is trained using the Gradient Boosted Tree algorithm from PySpark's MLlib. The dataset is split into training and testing sets. The GBT model is trained on the training set and evaluated on the testing set using appropriate evaluation metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

## Results
The project concludes by presenting the results of the trained model. This may include visualizations of feature importance, performance metrics, and any other relevant insights obtained from the analysis.

 - We can see that the algorithm predicts better salaries lower than 130k$, and start to understimate salaries over 175k\$. This could be due to the right skeweness of the salary distribution.

 - Overall, the results are satifying in both terms of RMSE and R2. Further improvements could be achieved by a proper hyperparameter tuning and feature engineering on the data.


