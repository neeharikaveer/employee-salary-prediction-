import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Define label encodings used during training
# Map for categorical variables (ensure these match your training)
workclass_map = {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4, 'Local-gov': 5, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 8, 'Others': 3}
marital_status_map = {'Married': 2, 'Single': 4, 'Divorced': 3, 'Widowed': 5, 'Separated': 6}
occupation_map = {
    'Tech-support': 4, 'Craft-repair': 6, 'Other-service': 8, 'Sales': 0, 'Exec-managerial': 11,
    'Prof-specialty': 3, 'Handlers-cleaners': 2, 'Machine-op-inspct': 13, 'Adm-clerical': 6,
    'Farming-fishing': 8, 'Transport-moving': 0, 'Priv-house-serv': 0, 'Protective-serv': 0,
    'Armed-Forces': 0
}
relationship_map = {'Wife': 1, 'Husband': 2, 'Not-in-family': 3, 'Own-child': 4, 'Unmarried': 5, 'Other-relative': 0}
race_map = {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 5}
gender_map = {'Male': 1, 'Female': 0}
country_map = {'United-States': 39, 'Others': 39}  # Adjust accordingly

# Feature order used during training
expected_columns = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country']

# Streamlit UI
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("FNLWGT", min_value=10000, value=200000)
educational_num = st.sidebar.slider("Educational Number", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", list(country_map.keys()))

# Input DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass_map.get(workclass, 3),
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status_map.get(marital_status, 4),
    'occupation': occupation_map.get(occupation, 0),
    'relationship': relationship_map.get(relationship, 3),
    'race': race_map.get(race, 4),
    'gender': gender_map.get(gender, 1),
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': country_map.get(native_country, 39)
}])

st.write("### 🔎 Input Data (Encoded)")
st.write(input_df)

# Predict single record
if st.button("Predict Salary Class"):
    processed_input = input_df[expected_columns]
    prediction = model.predict(processed_input)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with same format as training dataset", type="csv")

def preprocess_batch(df):
    df = df.copy()

    df['workclass'] = df['workclass'].map(workclass_map).fillna(3)
    df['marital-status'] = df['marital-status'].map(marital_status_map).fillna(4)
    df['occupation'] = df['occupation'].map(occupation_map).fillna(0)
    df['relationship'] = df['relationship'].map(relationship_map).fillna(3)
    df['race'] = df['race'].map(race_map).fillna(4)
    df['gender'] = df['gender'].map(gender_map).fillna(1)
    df['native-country'] = df['native-country'].map(country_map).fillna(39)

    return df[expected_columns]

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    try:
        processed_batch = preprocess_batch(batch_data)
        batch_preds = model.predict(processed_batch)
        batch_data['PredictedClass'] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
