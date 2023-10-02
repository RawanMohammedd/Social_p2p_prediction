import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title
st.set_page_config(page_title="Online P2P lending markets App")
st.title("Welcome to our Online P2P lending markets App")

# create a checkbox to toggle between loan and ROI predictions
predict_roi_checkbox = st.checkbox("ROI Predictions", value=False)

# loading the saved models
loaded_model = pickle.load(open(r'C:\Users\MBR\Downloads\Logistic_model_saved', 'rb'))
loaded_model_ROI = pickle.load(open(r'C:\Users\MBR\Downloads\Linear_model_saved', 'rb'))

# creating a function for loan status prediction
def Loanstatus_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        return 'Loan Rejected'
    else:
        return 'Loan Accepted'


# Define the ROIprediction function
def ROIprediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions using the loaded model
    roi_emi_prediction = loaded_model_ROI.predict(input_data_reshaped)

    # Check that the roi_emi_prediction array has two elements
    if roi_emi_prediction.shape == (1, 2):
        roi_emi_prediction = roi_emi_prediction.reshape(2,)

    # Separate the ROI and EMI predictions
    roi_prediction, emi_prediction = roi_emi_prediction[0], roi_emi_prediction[1]

    # Return the predicted ROI and EMI values
    return roi_prediction, emi_prediction

if predict_roi_checkbox:
    # display the user inputs for ROI predictions
    st.header('ROI and EMI Predictions')

    # getting the input data from the user
    BorrowerAPR2 = st.text_input('BorrowerAPR')
    EstimatedReturn = st.text_input('EstimatedReturn')
    EstimatedEffectiveYield = st.text_input('EstimatedEffectiveYield')
    MonthlyLoanPayment = st.text_input('MonthlyLoanPayment')
    LP_CustomerPayments = st.text_input('LP_CustomerPayments')
    LP_CustomerPrincipalPayments = st.text_input('LP_CustomerPrincipalPayments')
    LoanOriginalAmount = st.text_input('LoanOriginalAmount')
    LoanMonthsSinceOrigination = st.text_input('LoanMonthsSinceOrigination')
    LenderYield2 = st.text_input('LenderYield')
    BorrowerRate2 = st.text_input('BorrowerRate')
    EstimatedLoss = st.text_input('EstimatedLoss')
    LP_ServiceFees = st.text_input('LP_ServiceFees')



    # creating a button for ROI prediction
    if st.button('Predict ROI'):
        roi_pred, emi_pred = ROIprediction(
            [BorrowerAPR2, EstimatedReturn, EstimatedEffectiveYield, MonthlyLoanPayment, LP_CustomerPayments,
             LP_CustomerPrincipalPayments, LoanOriginalAmount, LoanMonthsSinceOrigination, LenderYield2, BorrowerRate2,EstimatedLoss,LP_ServiceFees])

        # display the predicted ROI and EMI values to the user
        st.write(f"Predicted ROI: {roi_pred}")
        st.write(f"Predicted EMI: {emi_pred}")
else:
    # display the user inputs for loan predictions
    st.header('Loan Status Predictions')

    # getting the input data from the user
    LoanCurrentDaysDelinquent = st.text_input('LoanCurrentDaysDelinquent')
    LoanNumber = st.text_input('LoanNumber')
    ListingNumber = st.text_input('ListingNumber')
    LP_GrossPrincipalLoss = st.text_input('LP_GrossPrincipalLoss')
    LP_NetPrincipalLoss = st.text_input('LP_NetPrincipalLoss')
    LoanStatus_Chargedoff = st.selectbox('LoanStatus_Chargedoff', [0, 1])
    LoanStatus_Current = st.selectbox('LoanStatus_Current', [1, 0])
    BorrowerAPR = st.text_input('BorrowerAPR')
    BorrowerRate = st.text_input('BorrowerRate')
    LenderYield = st.text_input('LenderYield')

    # creating a button for loan status prediction
    if st.button('Predict Loan Status'):
        pred = Loanstatus_prediction(
            [LoanCurrentDaysDelinquent,LoanNumber, ListingNumber, LP_GrossPrincipalLoss, LP_NetPrincipalLoss, LoanStatus_Chargedoff,
             LoanStatus_Current, BorrowerAPR, BorrowerRate, LenderYield])

        # display the prediction to the user using a radio button
        if pred:
            loan_status = st.radio('Loan Status:', ['Loan Rejected', 'Loan Accepted'],
                                   index=1 if pred == 'Loan Accepted' else 0)
