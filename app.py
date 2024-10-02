import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler using pickle
with open('logistic_regression_fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Centering the transaction form using CSS
st.markdown("""
    <style>
    .center-form {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .center-content {
        width: 50%;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Credit Card Fraud Detection System")

# Transaction form in the center
st.markdown("<div class='center-form'><div class='center-content'>", unsafe_allow_html=True)
st.write("Enter the transaction details below:")


def user_input_features():
    amount = st.number_input("Amount", min_value=0.0, step=0.01, format="%.2f")

    v_features = []
    for i in range(1, 29):
        v = st.number_input(f"V{i}", value=0.0, step=0.0001)
        v_features.append(v)

    return [amount] + v_features


features = user_input_features()
st.markdown("</div></div>", unsafe_allow_html=True)

if st.button("Predict"):
    # Preprocess the input
    scaled_amount = scaler.transform(np.array(features[0]).reshape(-1, 1))[0][0]
    processed_features = [scaled_amount] + features[1:]
    input_data = np.array(processed_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display the result
    if prediction == 1:
        st.error(f"**Fraudulent** transaction detected with a probability of {probability:.2f}.")
    else:
        st.success(f"**Legitimate** transaction detected with a probability of {1 - probability:.2f}.")
