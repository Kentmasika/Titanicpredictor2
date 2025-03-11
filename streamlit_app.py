import streamlit as st
import joblib
import pandas as pd
model = joblib.load('titanic_model.pkl')
st.title('Titanic Survival Predictor')
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.number_input('Siblings/Spouses', 0, 10, 0)
parch = st.number_input('Parents/Children', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 500.0, 32.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
input_data = pd.DataFrame({
    'Pclass': [pclass], 'Age': [age], 'SibSp': [sibsp],
    'Parch': [parch], 'Fare': [fare], 'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0], 'Embarked_S': [1 if embarked == 'S' else 0]
})
if st.button('Predict Survival'):
    prediction = model.predict(input_data)[0]
    st.success(f'Prediction: {"Survived" if prediction == 1 else "Did Not Survive"}')