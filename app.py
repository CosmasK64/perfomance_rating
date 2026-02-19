import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler_viz.pkl')  # Ensure this scaler matches training features

# Get feature names used during training
trained_columns = scaler.feature_names_in_

# -----------------------------
# APP TITLE
# -----------------------------
st.title('Employee Performance Prediction')

# -----------------------------
# MAPPINGS
# -----------------------------
education_mapping = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
satisfaction_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
work_life_balance_mapping = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
stock_option_mapping = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'}
performance_rating_mapping = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

# NEW: Department Mapping
department_mapping = {
    1: 'Sales',
    2: 'Research & Development',
    3: 'Human Resources',
    4: 'Finance',
    5: 'Data Science'
}

# -----------------------------
# USER INPUT
# -----------------------------
input_data = {}

# Numeric fields
numeric_fields = {
    'Age': (18, 60, 30),
    'DistanceFromHome': (0, 100, 5),
    'NumCompaniesWorked': (0, 50, 1),
    'PercentSalaryHike': (0, 100, 10),
    'TotalWorkingYears': (0, 50, 5),
    'TrainingTimesLastYear': (0, 20, 2),
    'ExperienceYearsAtThisCompany': (0, 50, 5),
    'YearsAtCompany': (0, 50, 3),
    'YearsInCurrentRole': (0, 50, 2),
    'YearsSinceLastPromotion': (0, 50, 1),
    'YearsWithCurrManager': (0, 50, 2)
}

for field, (min_val, max_val, default) in numeric_fields.items():
    input_data[field] = st.number_input(field, min_val, max_val, value=default)

# Categorical fields
input_data['Education'] = st.selectbox(
    'Education Level',
    options=list(education_mapping.keys()),
    format_func=lambda x: education_mapping[x]
)

input_data['EnvironmentSatisfaction'] = st.selectbox(
    'Environment Satisfaction',
    options=list(satisfaction_mapping.keys()),
    format_func=lambda x: satisfaction_mapping[x]
)

input_data['JobInvolvement'] = st.selectbox(
    'Job Involvement',
    options=list(satisfaction_mapping.keys()),
    format_func=lambda x: satisfaction_mapping[x]
)

input_data['JobSatisfaction'] = st.selectbox(
    'Job Satisfaction',
    options=list(satisfaction_mapping.keys()),
    format_func=lambda x: satisfaction_mapping[x]
)

input_data['RelationshipSatisfaction'] = st.selectbox(
    'Relationship Satisfaction',
    options=list(satisfaction_mapping.keys()),
    format_func=lambda x: satisfaction_mapping[x]
)

input_data['WorkLifeBalance'] = st.selectbox(
    'Work Life Balance',
    options=list(work_life_balance_mapping.keys()),
    format_func=lambda x: work_life_balance_mapping[x]
)

input_data['StockOptionLevel'] = st.selectbox(
    'Stock Option Level',
    options=list(stock_option_mapping.keys()),
    format_func=lambda x: stock_option_mapping[x]
)

# NEW: Department Input
input_data['Department'] = st.selectbox(
    'Department',
    options=list(department_mapping.keys()),
    format_func=lambda x: department_mapping[x]
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button('Predict'):

    # Initialize input DataFrame with trained columns
    input_df = pd.DataFrame(columns=trained_columns)
    input_df.loc[0] = 0  # Default zeros

    # Fill matching columns
    for col in input_data:
        if col in trained_columns:
            input_df.at[0, col] = input_data[col]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    predicted_label = performance_rating_mapping.get(prediction, prediction)

    st.success(f'Predicted Performance Rating: {predicted_label}')
