import streamlit as st
import os
import pickle
import pandas as pd
import tensorflow as tf

st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout='centered',
    page_icon=':1234:',
)

global churn_cutoff
churn_cutoff = 0.3

#  Function to load machine learning components
def load_components_function(fp):
    #To load the machine learning components saved to re-use in the app
    return tf.keras.models.load_model(fp)

def load_components(fp):
    #To load the machine learning components saved to re-use in the app
    with open(fp,"rb") as f:
        object = pickle.load(f)
    return object

# Loading the machine learning components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH,"Assets","telco_model.h5")
ml_components_dict = load_components_function(fp=ml_core_fp)
scaler_fp = os.path.join(DIRPATH,"Assets","scalerKer.pkl")
scaler_dict = load_components(fp=scaler_fp)


st.title(":1234: Telecom Churn App")
st.write("""Welcome to ChurnShield Telecom Churn Prediction app!
         This app allows you to predict the probability of Churn for a specific 
         customer based on our trained deep learning models.""")

with st.form(key="information",clear_on_submit=True):
    st.write("Enter the information of your Customer")
    gender = st.selectbox("Gender ?", ['M', 'F'])
    seniorcitizen = st.selectbox("Senior Citizen ?", ['Yes', 'No'])
    tenure = st.number_input("Enter tenure in months: ")
    phoneservice = st.selectbox("Has the Customer opted for Phone Service? ", ['Yes', 'No'])
    onlinesecurity = st.selectbox("Has the Customer opted for Online Security Service? ", ['Yes', 'No', 'No internet Service'])
    onlinebackup = st.selectbox("Has the Customer opted for Online Backup Service? ", ['Yes', 'No', 'No internet Service'])
    deviceprotection = st.selectbox("Has the Customer opted for Device Protection Service? ", ['Yes', 'No', 'No internet Service'])
    techsupport = st.selectbox("Has the Customer opted for Tech Support Service? ", ['Yes', 'No', 'No internet Service'])
    streamingmovies = st.selectbox("Has the Customer opted for Streaming Movie Service? ", ['Yes', 'No', 'No internet Service'])
    monthlycharges = st.number_input("Enter the Monthly charges (Rs): ")
    contract = st.selectbox("What's the Contract type of Customer?  ", ['Month-to-month', 'One year', 'Two year'])
    billing = st.selectbox("Is it Paperless Billing Service? ", ['Yes', 'No'])
    paymentmethod = st.selectbox("What's the Payment Method of Service? ", ['Electronic Check', 'Mailed Check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    totalcharges = st.number_input("Enter the Total charges (Rs): ")


    # Prediction
    if st.form_submit_button("Predict"):
        # Dataframe Creation
        data = pd.DataFrame({
            "SeniorCitizen": [seniorcitizen],
            "tenure": [tenure],
            "PhoneService": [phoneservice],
            "OnlineSecurity_Yes": [onlinesecurity],
            "OnlineBackup_Yes": [onlinebackup],
            "DeviceProtection_Yes": [deviceprotection],
            "TechSupport_Yes": [techsupport],
            "StreamingMovies_Yes": [streamingmovies],
            "MonthlyCharges": [monthlycharges],
            "Contract_One year": [contract],
            "PaperlessBilling": [billing],
            "PaymentMethod_Credit card (automatic)": [paymentmethod],
            "TotalCharges": [totalcharges]
        })       
        
        # Feature Engineering
        categorical_columns = ['SeniorCitizen', 'PhoneService', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingMovies_Yes', 'Contract_One year', 'PaperlessBilling', 'PaymentMethod_Credit card (automatic)']
        for col in categorical_columns:
            data[col] = 1 if data[col].iloc[0] == 'Yes' else 0
        
        # Scale the numerical columns
        columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        data[columns_to_scale] = scaler_dict.transform(data[columns_to_scale])
        data = data.drop('TotalCharges', axis=1)
        # st.write(data)

        # Make prediction using the model
        data = data[['SeniorCitizen', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges',
             'Contract_One year', 'PaymentMethod_Credit card (automatic)', 'OnlineSecurity_Yes',
             'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingMovies_Yes']]

        predictions = ml_components_dict.predict(data)
        # st.write(predictions)
        ans = predictions[0]
        # Display the predictions
        st.balloons()

        if ans>churn_cutoff:
            st.success(f"The Customer is likely to Churn.")
        else:
            st.success(f"The Customer will not Churn.")
        
        # Display the predictions with custom styling
        # st.success(f"Predicted Churn Rate: {predictions[0]:,.2f}",icon="✅")