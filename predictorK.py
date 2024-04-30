import streamlit as st
import os
import pickle
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout='centered',
    page_icon=':1234:',
)

global churn_cutoff
churn_cutoff = 0.3

# Loading the machine learning components
def app():
    ml_components_dict = tf.keras.models.load_model("./Assets/telco_model.h5")
    scaler_dict = joblib.load("./Assets/scalerKer.pkl","rb")

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
            

            if ans>churn_cutoff:
                st.success(f"The Customer is likely to Churn. 😢")
            else:
                st.success(f"The Customer will not Churn.")
                st.balloons()
            
        # Display the predictions with custom styling
        # st.success(f"Predicted Churn Rate: {predictions[0]:,.2f}",icon="✅")

app()