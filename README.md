# Cardiovascular Disease Prediction 🫀
### Overview
The project aims to create a machine learning model for predicting cardiovascular disease (CVD) risk. By accurately predicting the risk of CVD, the goal is to enable early detection, reduce mortality, minimize healthcare costs, and contribute to personalized medicine. This is essential since cardiovascular diseases are the leading cause of death globally, with significant social and economic impacts.

### Motivation
The primary motivation is to address the global challenge of cardiovascular diseases, which account for approximately 31% of deaths worldwide. Early detection through a predictive model can save lives, reduce healthcare costs, and improve quality of life, especially for individuals with existing risk factors like hypertension, diabetes, or hyperlipidemia.

### Dataset
The dataset used in this project contains 11 features, which are crucial indicators of cardiovascular disease. These features enable the model to predict the likelihood of heart disease effectively.

### Technical Aspects
The project consists of several key components:

1. **Domain Analysis**: Analysis of cardiovascular diseases to understand their global impact and identify relevant features for prediction.
2. **Data Preprocessing**: Cleaning the dataset to remove any irrelevant data, handle missing values, and standardize the input.
3. **Feature Selection**: Identifying the most significant features that contribute to accurate CVD prediction.
4. **Model Training**: Using machine learning algorithms to train the predictive model.
5. **Model Evaluation**: Evaluating the model's performance using metrics such as accuracy, precision, recall, and ROC-AUC score.
6. **Visualization**: Employing visualization techniques to understand data distributions and model outcomes.

## Installation
The project is developed in Python 3.7+. To install the required packages and libraries, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Run the following command:

```bash
pip install -r requirements.txt
```

### Run
To run the project locally:

1. **Data Preprocessing**: Execute the preprocessing steps provided in the notebook to clean and prepare the data.
2. **Model Training**: Train the model using the dataset and save the trained model.
3. **Model Evaluation**: Use evaluation metrics to test the accuracy and efficiency of the trained model.

### Directory Tree
```
├── data
│   └── heart_disease.csv
├── notebooks
│   └── Heart_Disease_Analysis.ipynb
├── models
│   └── trained_model.pkl
├── requirements.txt
└── README.md
```

### Model Evaluation
The model's performance is evaluated based on various metrics:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Evaluates the quality of positive predictions.
- **Recall**: Assesses the model's ability to detect actual positives.
- **ROC-AUC Score**: Provides a summary of the model's performance.

### Performance Analysis
The effectiveness of the model is measured both qualitatively and quantitatively:

- **Visualization**: Data distributions, correlations, and prediction accuracy are visualized using libraries like `matplotlib` and `seaborn`.
- **Evaluation Metrics**: Accuracy, precision, recall, and ROC-AUC scores provide a comprehensive view of model performance.
- **Feature Analysis**: Insights into how different features impact prediction accuracy.

### Technologies Used
- **Python**: Programming Language
- **Pandas**: Data Manipulation
- **NumPy**: Numerical Computations
- **Matplotlib & Seaborn**: Data Visualization
- **Scikit-learn**: Machine Learning Library
- **Jupyter Notebook**: Interactive Computing Environment

### Credits
- **Data Source**: The dataset used for predicting cardiovascular disease.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Matplotlib & Seaborn**: For visualizing data and results.
- **Scikit-learn**: For implementing machine learning models.
