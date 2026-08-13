import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("FraudDetectionPipeline_RF.pkl")

st.title("Web App - Dự đoán Gian Lận trong Giao Dịch")

st.markdown("Hãy nhập thông tin giao dịch và ấn nút dự đoán")

st.divider()

transaction_type = st.selectbox("Loại giao dịch", ["PAYMENT","TRANSFER","CASH_IN","CASH_OUT","DEBIT"])
amount = st.number_input("Số tiền", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Số dư hiện tại (Người gửi)", min_value=0.0, value=10000.0)
newbalanceOrig	= st.number_input("Số dư sau (Người gửi)", min_value=0.0, value=90000.0)
oldbalanceDest = st.number_input("Số dư hiện tại (Người nhận)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Số dư sau (Người nhận)", min_value=0.0, value=0.0)

if st.button("Dự đoán"):
    input_data = pd.DataFrame([
        {"type": transaction_type,
         "amount": amount,
         "oldbalanceOrg": oldbalanceOrg,
         "newbalanceOrig": newbalanceOrig,
         "oldbalanceDest": oldbalanceDest,
         "newbalanceDest": newbalanceDest
        }
    ]) 
    prediction = model.predict(input_data)[0]

    st.subheader(f"Dự đoán: {int(prediction)}")

    if prediction == 1:
        st.error("Giao dịch này có thể là GIAN LẬN")
    else:
        st.success("Giao dịch này không như GIAN LẬN")


