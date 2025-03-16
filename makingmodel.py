import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'poaching_risk_model.pkl')