
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
@st.cache
def load_data():
    data = pd.read_csv('Cervical Cancer Risk Classification.csv')
    # Handling missing values, replace with median or mode as appropriate
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())
    return data

data = load_data()

# Title and Header
st.title('Cervical Cancer Risk Analysis')
st.header('Dataset Overview')

# Display data
if st.checkbox('Show raw data'):
    st.write(data.head())

# Display statistics
st.header('Statistical Summary')
st.write(data.describe())

# Data Visualization
st.header('Data Visualization')
selected_visualization = st.selectbox('Select the visualization:', ['Correlation Heatmap', 'Risk Factor Distributions', 'Biopsy Outcome Count'])
if selected_visualization == 'Correlation Heatmap':
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot(plt)
elif selected_visualization == 'Risk Factor Distributions':
    factor = st.selectbox('Select a risk factor to visualize:', data.columns)
    plt.figure(figsize=(10, 4))
    sns.histplot(data[factor], kde=True, color='blue')
    plt.title(f'Distribution of {factor}')
    st.pyplot(plt)
elif selected_visualization == 'Biopsy Outcome Count':
    plt.figure(figsize=(10, 4))
    sns.countplot(x='Biopsy', data=data)
    plt.title('Biopsy Outcomes')
    st.pyplot(plt)

# Predictive Modeling
st.header('Predictive Modeling')

# Feature Selection and Preprocessing
feature_cols = data.columns.drop('Biopsy')
X = data[feature_cols]
y = data['Biopsy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Model Evaluation
st.subheader('Model Evaluation')
st.text('Classification Report:')
st.text(classification_report(y_test, y_pred))

# User Inputs for Prediction
st.header('Interactive Risk Prediction')
user_data = {col: st.number_input(f"Enter {col}", min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].median())) for col in feature_cols}
user_data_array = np.array([list(user_data.values())]).astype(float)
user_data_scaled = scaler.transform(user_data_array)

if st.button('Predict Risk'):
    prediction = model.predict(user_data_scaled)
    result = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    st.subheader(f'Your predicted risk of needing a biopsy: {result}')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .streamlit-expanderHeader { font-size: 16px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)
