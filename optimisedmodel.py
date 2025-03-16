import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Specify data types for problematic columns
dtype_dict = {
    'Id': 'int64',
    'Year': 'int64',
    'Quantity': 'float64',
    'Import.permit.RandomID': 'str',
    'Export.permit.RandomID': 'str',
    'Origin.permit.RandomID': 'str',
    # Add other columns as needed
}

# Load the CSV file in chunks
chunks = []
for chunk in pd.read_csv('/Users/sahil.pardasani/Desktop/Illegal Poaching Project/Trade_database_download_v2024.1/combined_data.csv', chunksize=100000, dtype=dtype_dict, low_memory=False):
    chunks.append(chunk)

# Combine chunks into a single DataFrame
combined_df = pd.concat(chunks, ignore_index=True)

# Define the target column
combined_df['HighRisk'] = (combined_df['Quantity'] > 1000).astype(int)  # Example rule

# Preprocessing
combined_df = combined_df.dropna()  # Handle missing values

# Features and target
X = combined_df.drop(columns=['Id', 'Year', 'HighRisk'])
y = combined_df['HighRisk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numeric columns
categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for categorical and numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_columns),  # Pass numeric columns through
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-hot encode categorical columns
    ])

# Apply preprocessing to the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Define the hyperparameters obtained from Optuna
best_params = {
    'n_estimators': 137,  # From Trial 1
    'max_depth': 22,      # From Trial 1
    'min_samples_split': 7,  # From Trial 1
    'min_samples_leaf': 1,   # From Trial 1
    'class_weight': 'balanced'
}

# Initialize the model with the best hyperparameters
best_model = RandomForestClassifier(**best_params, random_state=42)

# Train the model on the full training set
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(best_model, 'optimized_poaching_risk_model.pkl')