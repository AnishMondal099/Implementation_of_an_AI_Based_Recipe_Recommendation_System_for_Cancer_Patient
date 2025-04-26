import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
# 1. preprocess_data(df)
# Purpose:
# Prepares the dataset by handling missing values, encoding categorical features, and normalizing numerical features.

# Steps:

# Handle missing values → Replaces missing values with "None".
# One-hot encoding categorical features → Converts categorical variables into numerical representations.
# Uses OneHotEncoder(sparse_output=False, handle_unknown='ignore') to encode categorical columns.
# Normalize numerical features → Scales numerical values for consistency.
# Uses StandardScaler() to scale numeric columns.
# Combine encoded and scaled features → Creates a single feature matrix.
# Returns:

# Processed DataFrame (df)
# Feature matrix (feature_matrix)
# Encoder (encoder)
# Scaler (scaler)

def preprocess_data(df):
    # Fill missing values
    df.fillna("None", inplace=True)

    # One-hot encode categorical features
    categorical_cols = ['Cancer_Type', 'Cancer_Stage', 'Comorbidities', 'Dietary_Restrictions', 'Cuisine', 'Preparation_Method']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])

    # Normalize numerical features
    numerical_cols = ['Caloric_Intake_Requirement (kcal/day)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 'Cooking_Time (minutes)']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])

    # Combine all features into a single matrix
    feature_matrix = np.hstack((encoded_features, scaled_features))
    return df, feature_matrix, encoder, scaler
# 2. build_neural_network(input_dim)
# Purpose:
# Creates a neural network model to predict recipe suitability.

# Steps:

# Defines an MLPRegressor (Multilayer Perceptron Regressor) model with:
# Three hidden layers → 128, 64, and 32 neurons.
# Activation function → 'relu' (Rectified Linear Unit).
# Optimizer → 'adam' (Adaptive Moment Estimation).
# Max iterations → 500 (controls training duration).
# Returns:
# A neural network model (MLPRegressor).

def build_neural_network(input_dim):
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=500)
    return model
# 3. recommend_recipes(patient_profile, df, feature_matrix, encoder, scaler, model, top_n=5)
# Purpose:
# Generates personalized recipe recommendations based on the patient’s dietary needs.

# Steps:

# Convert patient profile to DataFrame.
# Encode patient’s categorical features using the previously fitted encoder.
# Normalize patient’s numerical features using the previously fitted scaler.
# Create a patient feature vector by combining encoded and normalized features.
# Predict recipe suitability scores using the trained model.
# Select top N recipes with the highest predicted scores.
# Returns:
# A DataFrame containing the top N recommended recipes.

def recommend_recipes(patient_profile, df, feature_matrix, encoder, scaler, model, top_n=5):
    # Convert patient profile to DataFrame
    patient_df = pd.DataFrame([patient_profile])

    # Encode categorical features
    encoded_patient = encoder.transform(patient_df[['Cancer_Type', 'Cancer_Stage', 'Comorbidities', 'Dietary_Restrictions', 'Cuisine', 'Preparation_Method']])

    # Normalize numerical features
    scaled_patient = scaler.transform(patient_df[['Caloric_Intake_Requirement (kcal/day)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 'Cooking_Time (minutes)']])

    # Combine features into a vector
    patient_vector = np.hstack((encoded_patient, scaled_patient))

    # Predict suitability scores for all recipes
    scores = model.predict(feature_matrix)

    # Get top N recommended recipes
    top_indices = scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Cuisine', 'Preparation_Method', 'Caloric_Intake_Requirement (kcal/day)']]
# Load dataset
df = pd.read_excel('cancer_recipe_recommendation_dataset.xlsx', sheet_name='Sheet1')
df, feature_matrix, encoder, scaler = preprocess_data(df)
# Build and train the neural network model
model = build_neural_network(input_dim=feature_matrix.shape[1])
model.fit(feature_matrix, np.ones(feature_matrix.shape[0]))
# Example patient profile
# patient_profile = {
#     'Cancer_Type': 'Lung',
#     'Cancer_Stage': 'Stage II',
#     'Comorbidities': 'Diabetes',
#     'Dietary_Restrictions': 'Low-fat',
#     'Cuisine': 'Indian',
#     'Preparation_Method': 'Baked',
#     'Caloric_Intake_Requirement (kcal/day)': 1800,
#     'Protein (g)': 75,
#     'Carbohydrates (g)': 200,
#     'Fat (g)': 40,
#     'Fiber (g)': 5,
#     'Cooking_Time (minutes)': 30
# }
# 4. Training and Testing
# Load dataset from an Excel file.
# Preprocess dataset using preprocess_data(df).
# Build the neural network with build_neural_network(input_dim).
# Train the model using a placeholder target (for now, an array of ones).
# Define an example patient profile.
# Generate recipe recommendations using recommend_recipes().
# This approach personalizes recipe recommendations based on cancer type, dietary restrictions, and nutrient needs.

# Would you like any modifications or enhancements?

# Get recommendations
# recommendations = recommend_recipes(patient_profile, df, feature_matrix, encoder, scaler, model, top_n=5)
# print(recommendations)