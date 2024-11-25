import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('Cervical Cancer Risk Classification.csv')
    data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
    data.dropna(inplace=True)  # Drop rows with NaN values
    data = data.astype(float)  # Convert all data to float

    # Assuming 'Biopsy' is the target variable
    X = data.drop('Biopsy', axis=1)
    y = data['Biopsy'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, data

model, scaler, X_train, X_test, y_train, y_test, data = load_and_preprocess_data()

# Streamlit dashboard layout with tabs
st.title('Cervical Cancer Risk Classification Dashboard')
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Data Overview", "Visualizations", "Model Performance", "Predict"])

with tab1:
    st.header('Welcome to the Cervical Cancer Risk Classification Dashboard')
    st.write('Navigate through the tabs to explore different sections.')

with tab2:
    st.header('Data Overview')
    st.write('Basic statistics:')
    st.write(data.describe())

with tab3:
    st.header('Data Visualizations')
    
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    ax.hist(data['Age'], bins=20, color='blue')
    st.pyplot(fig)

    st.subheader('Number of Cases by Diagnosis Result')
    biopsy_counts = data['Biopsy'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(biopsy_counts.index, biopsy_counts.values, color=['green', 'red'])
    ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
    st.pyplot(fig)

    st.subheader('Correlation Heatmap')
    corr = data.drop('Biopsy', axis=1).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab4:
    st.header('Algorithm Performance')
    
    # Models for performance comparison
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier()
    }

    selected_models = {model: st.checkbox(model, key=model) for model in models.keys()}
    for model_name, selected in selected_models.items():
        if selected:
            st.subheader(f'{model_name} Results')
            current_model = models[model_name]
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)
            st.write('Classification Report:')
            st.text(classification_report(y_test, y_pred))
            
            st.write('Confusion Matrix:')
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred, labels=current_model.classes_)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
            st.pyplot(fig)

with tab5:
    st.header('Model Predictions')
    st.write('Input data to predict the risk of cervical cancer.')
    # Input form for prediction
    input_data = {feature: st.number_input(f"Enter {feature}", format="%.2f") for feature in data.columns[:-1]}  # Excludes target
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)  # Scale the input

    if st.button('Predict'):
        prediction = model.predict(input_scaled)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        st.success(f'The predicted biopsy result is: {result}')
