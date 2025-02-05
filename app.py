import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title and Description
st.title("🌊 Titanic Survival Prediction App")
st.write("""
This app predicts whether a passenger on the Titanic would have survived or not based on their details.  
**Instructions:**  
1. Use the sidebar to input the passenger details.  
2. The model will predict if the passenger would survive or not along with the probabilities.
""")

# Sidebar for user input
st.sidebar.header("Enter Passenger Details Below:")

# Collect user input
def user_input():
    Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.sidebar.selectbox("Sex", ["male", "female"])
    Age = st.sidebar.slider("Age", 1, 100, 25)
    SibSp = st.sidebar.slider("Number of Siblings/Spouses aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Number of Parents/Children aboard", 0, 6, 0)
    Fare = st.sidebar.slider("Fare Paid ($)", 0, 500, 50)
    Embarked = st.sidebar.selectbox(
        "Port of Embarkation",
        ["Cherbourg (France)", "Queenstown (Ireland)", "Southampton (UK)"]
    )

    # Map categorical variables to numeric values for the model
    Sex = 0 if Sex == "male" else 1
    Embarked = {"Cherbourg (France)": 0, "Queenstown (Ireland)": 1, "Southampton (UK)": 2}[Embarked]

    # Create input DataFrame for the model
    data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }
    return pd.DataFrame(data, index=[0])


# Get user input
input_data = user_input()

# Display user input
st.subheader("Passenger Information")
st.write(input_data)

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction result with professional formatting
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("🚶 **The passenger would survive.**")
else:
    st.error("☠️ **The passenger would not survive.**")

# Display prediction probabilities
st.subheader("Prediction Probabilities")
st.write(f"🔵 **Survival Probability:** {prediction_proba[0][1] * 100:.2f}%")
st.write(f"🔴 **Death Probability:** {prediction_proba[0][0] * 100:.2f}%")

# Footer
st.write("---")
st.caption("This app uses a logistic regression model trained on the Titanic dataset to make survival predictions.")