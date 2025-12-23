# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Set page configuration
# st.set_page_config(page_title="Student Result Predictor", layout="centered")

# # Load the saved model
# try:
#     model = joblib.load("logistic_model.joblib")
# except FileNotFoundError:
#     st.error("Model file not found. Please ensure 'logistic_model.joblib' is in the same directory.")

# # App Header
# st.title("🎓 Student Pass/Fail Predictor")
# st.write("""
# This app uses a **Logistic Regression** model to predict if a student will pass or fail 
# based on their study hours and attendance.
# """)

# st.divider()

# # Input Section
# st.subheader("Enter Student Data")
# col1, col2 = st.columns(2)

# with col1:
#     hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

# with col2:
#     attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

# # Prediction Logic
# if st.button("Predict Result", type="primary"):
#     # Prepare data for prediction
#     features = np.array([[hours, attendance]])
    
#     # Make prediction
#     prediction = model.predict(features)[0]
#     probability = model.predict_proba(features)[0][1] # Probability of passing (class 1)

#     st.divider()

#     # Display Results
#     if prediction == 1:
#         st.success(f"### Result: PASS ✅")
#         st.write(f"The model is **{probability*100:.2f}%** confident the student will pass.")
#     else:
#         st.error(f"### Result: FAIL ❌")
#         st.write(f"The model is **{(1-probability)*100:.2f}%** confident the student will fail.")

# # Sidebar Information (Optional)
# st.sidebar.header("About the Model")
# st.sidebar.info("""
# The model was trained on a dataset of 5,000 students using:
# - **Features:** Hours Studied, Attendance
# - **Algorithm:** Logistic Regression
# """)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Result Predictor", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure 'logistic_model.pkl' is in the same directory as this script
    return joblib.load("linear_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the app directory.")
    st.stop()

# --- UI DESIGN ---
st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter the student's data below to predict if they will Pass (1) or Fail (0).")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=7)

with col2:
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)

# Prediction Logic
if st.button("Predict Result"):
    # Create input dataframe matching the model's expected features
    input_data = pd.DataFrame([[hours_studied, attendance]], 
                              columns=['Hours_Studied', 'Attendance'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    
    if prediction == 1:
        st.success(f"### Result: PASS ✅")
    else:
        st.error(f"### Result: FAIL ❌")
        
    st.write(f"Confidence Level: **{probability:.2%}**")

# --- VISUALIZATION SECTION ---
st.sidebar.header("About the Data")
st.sidebar.info(
    "This model uses Logistic Regression to classify students based on "
    "their study hours and classroom attendance."
)

# Optional: Display a sample chart if a dataset is available
if st.sidebar.checkbox("Show Distribution Insight"):
    st.subheader("General Trends (Conceptual)")
    # Recreating the scatter plot logic from your notebook
    # Note: In a real app, you'd load your actual CSV here
    st.write("The model suggests that higher attendance and more study hours "
             "significantly increase the likelihood of passing.")
    
    # Simple visual representation of a decision boundary
    fig, ax = plt.subplots()
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Attendance")
    ax.set_title("Success Zone Illustration")
    ax.axvspan(6, 12, alpha=0.2, color='green', label='Higher Pass Probability')
    ax.axhspan(70, 100, alpha=0.2, color='blue')
    st.pyplot(fig)