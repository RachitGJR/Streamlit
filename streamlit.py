import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# App Title
st.title("Exam Score Predictor")

# Sidebar for manual data entry (Mock Training Data)
st.sidebar.header("Model Training Data")
st.sidebar.info("This app uses a simple Linear Regression model based on Study Hours.")

# Hardcoded training data: [Hours Studied] -> [Score]
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_train = np.array([45, 50, 55, 60, 68, 72, 80, 85, 92, 98])

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# User Input for Prediction
st.subheader("Predict Your Score")
hours = st.number_input("Enter number of hours studied:", min_value=0.0, max_value=24.0, value=5.0)

if st.button("Predict Score"):
    # Reshape input for sklearn prediction
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)
    
    # Display result
    predicted_score = min(100, round(prediction[0], 2)) # Cap score at 100
    st.success(f"Estimated Exam Score: **{predicted_score}%**")
