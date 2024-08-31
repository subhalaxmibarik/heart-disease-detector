import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Function to fetch heart disease dataset from UCI repository
def fetch_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    return pd.read_csv(url, names=column_names)

# Function to preprocess data and build model
def preprocess_and_train_model():
    # Fetch heart disease dataset
    heart_disease_df = fetch_heart_disease_data()
    
    # Replace '?' with NaN
    heart_disease_df.replace('?', pd.NA, inplace=True)
    
    # Convert object columns to numeric
    heart_disease_df = heart_disease_df.apply(pd.to_numeric, errors='ignore')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    heart_disease_df_imputed = pd.DataFrame(imputer.fit_transform(heart_disease_df), columns=heart_disease_df.columns)
    
    # Split data into features and target
    X = heart_disease_df_imputed.drop('target', axis=1)
    y = heart_disease_df_imputed['target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

# Function to make predictions
def predict(model, input_data):
    return model.predict(input_data)
