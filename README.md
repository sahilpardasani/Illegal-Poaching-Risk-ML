# Illegal-Poaching-Risk-ML
This project aims to predict regions at high risk of illegal poaching using machine learning. The dataset used is the CITES Trade Database, which contains records of international trade in endangered species. The goal is to identify patterns and factors that contribute to high-risk poaching activities.

Data Source

CITES Trade Database: A comprehensive database of international trade in endangered species, containing information such as species, quantity, origin, and purpose of trade.
Methods and Libraries Used

1. Data Preprocessing

Pandas: Used for data manipulation and cleaning.
Loaded the CITES Trade Database, which consists of multiple CSV files.
Combined the CSV files into a single DataFrame.
Handled missing values and encoded categorical variables.
SQLAlchemy: Used to interact with a PostgreSQL database.

2. Feature Engineering

OneHotEncoder: Used to encode categorical variables into numeric format.
Applied one-hot encoding to categorical columns such as Taxon, Purpose, and Source.
ColumnTransformer: Used to apply different preprocessing steps to numeric and categorical columns.

3. Model Training
Scikit-learn: Used for machine learning.
Split the data into training and testing sets using train_test_split.
Trained a RandomForestClassifier to predict high-risk poaching activities.
Evaluated the model using metrics such as accuracy, precision, recall, and F1-score.
Optuna: Used for hyperparameter tuning.
Optimized hyperparameters such as n_estimators, max_depth, min_samples_split, and min_samples_leaf.
Used cross-validation to ensure the model generalizes well to unseen data.

4. Model Evaluation
Accuracy: Measured the proportion of correctly classified instances.
Classification Report: Provided detailed metrics such as precision, recall, and F1-score for each class.

Model Deployment
Joblib: Used to save the trained model for future use.
Saved the optimized RandomForestClassifier to a file (optimized_poaching_risk_model.pkl).
The other model is overfitted to the data.

Install Dependencies: pip install pandas scikit-learn optuna joblib sqlalchemy
Load the Data: You can find the dataset on the internet.
Run the Code: Execute the script to preprocess the data, train the model, and evaluate its performance.
Save the Model: The trained model will be saved to a file

Future Work
Feature Selection: Explore feature importance and select the most relevant features for the model.
Model Improvement: Experiment with other machine learning algorithms such as Gradient Boosting or Neural Networks.
Real-Time Prediction: Deploy the model as a web service using Flask or FastAPI for real-time predictions.
