import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import streamlit as st
import pickle


model = tf.keras.models.load_model('ann.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('one_encoder_geography.pkl','rb') as file:
    one_encoder =pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler =pickle.load(file)

# User input
geography =st.selectbox('Geography', one_encoder.categories_[0])
gender =st.selectbox('Gender', label_encoder.classes_)
age =st.slider('Age', 18, 92)
balance =st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure =st.slider('Tenure', 0, 10)
num_of_products =st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox( ' Is Active Member', [0, 1])

input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_encoder.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})
geo_encoder =one_encoder.transform([[geography]]).toarray()
geo_encoder_df =pd.DataFrame(geo_encoder,columns=one_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)
input_data_scaled =scaler.transform(input_data)
predection = model.predict(input_data_scaled)
predection_prob=predection[0][0]
st.write("the prediction probability is :",predection_prob)
if(predection_prob>0.5):
    st.write("this is likely to churn")

else:
    st.write("this is not likely to churn")
st.markdown("---")

