import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Post-COVID Health Effects Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Post-COVID-19 Health Effects & Risk Predictor")
st.markdown("""
Welcome to the interactive dashboard for visualizing post-COVID health effects and predicting long-term risks. 
This application supports the dissertation research by providing data profiling, predictive modeling, and model explainability (XAI).
""")

# ==========================================
# 2. DATA LOADING & PREPROCESSING (Cached)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('post_covid_health_effects_expanded.csv')
    df.columns = df.columns.str.strip()
    
    # Keep original dataframe for EDA
    df_raw = df.copy()
    
    # Preprocessing
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        
    X = df_encoded.drop(columns=['Long_COVID_Risk'])
    y = df_encoded['Long_COVID_Risk']
    
    # For this dashboard, we will train a baseline Random Forest 
    # (In your actual thesis, you can load your pre-trained saved model using joblib or pickle)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    return df_raw, df_encoded, X, y, rf_model, label_encoders

df_raw, df_encoded, X, y, model, encoders = load_and_preprocess_data()

# ==========================================
# 3. SIDEBAR NAVIGATION & TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🏥 Predict Patient Risk", "🧠 Model Explainability (SHAP)"])

# ------------------------------------------
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
with tab1:
    st.header("Dataset Overview")
    
    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patient Records", df_raw.shape[0])
    col2.metric("Total Features", df_raw.shape[1] - 1)
    col3.metric("Target Variable", "Long_COVID_Risk")
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Target Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_raw, x='Long_COVID_Risk', palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col_chart2:
        st.subheader("Age vs. Days to Recovery")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df_raw, x='Age', y='Days_to_Recovery', hue='Long_COVID_Risk', palette='viridis', alpha=0.7, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(10))

# ------------------------------------------
# TAB 2: PREDICT PATIENT RISK
# ------------------------------------------
with tab2:
    st.header("Patient Risk Assessment Tool")
    st.markdown("Enter patient metrics below to predict their Long COVID risk level using the Random Forest model.")
    
    # Create input fields matching your dataset columns
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", df_raw['Gender'].unique())
        covid_severity = st.selectbox("COVID Severity", df_raw['COVID_Severity'].unique())
        hospitalized = st.selectbox("Hospitalized", df_raw['Hospitalized'].unique())
        
    with col_input2:
        days_recovery = st.number_input("Days to Recovery", min_value=0, value=14)
        fatigue = st.slider("Fatigue Level (0-10)", min_value=0, max_value=10, value=5)
        breathing = st.selectbox("Breathing Issue", df_raw['Breathing_Issue'].unique())
        brain_fog = st.selectbox("Brain Fog", df_raw['Brain_Fog'].unique())
        
    with col_input3:
        taste_smell = st.selectbox("Loss of Taste/Smell", df_raw['Loss_of_Taste_Smell'].unique())
        physical_activity = st.selectbox("Physical Activity Level", df_raw['Physical_Activity_Level'].unique())
        mental_health = st.slider("Mental Health Impact (0-10)", min_value=0, max_value=10, value=5)

    if st.button("Predict Long COVID Risk", type="primary"):
        # 1. Format input into a dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'COVID_Severity': [covid_severity],
            'Hospitalized': [hospitalized],
            'Days_to_Recovery': [days_recovery],
            'Fatigue_Level': [fatigue],
            'Breathing_Issue': [breathing],
            'Brain_Fog': [brain_fog],
            'Loss_of_Taste_Smell': [taste_smell],
            'Physical_Activity_Level': [physical_activity],
            'Mental_Health_Impact': [mental_health]
        })
        
        # 2. Encode categorical inputs using the fitted label encoders
        input_encoded = input_data.copy()
        for col in input_encoded.columns:
            if col in encoders:
                # Handle unseen labels just in case
                known_classes = list(encoders[col].classes_)
                if input_encoded[col].iloc[0] in known_classes:
                    input_encoded[col] = encoders[col].transform(input_encoded[col])
                else:
                    input_encoded[col] = 0 # Fallback
                    
        # 3. Make Prediction
        prediction_encoded = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]
        
        # 4. Decode Prediction
        predicted_class = encoders['Long_COVID_Risk'].inverse_transform([prediction_encoded])[0]
        
        # 5. Display Result
        st.success(f"### Predicted Risk Level: **{predicted_class}**")
        
        st.write("**Prediction Probabilities:**")
        prob_df = pd.DataFrame({
            'Risk Level': encoders['Long_COVID_Risk'].classes_,
            'Probability': prediction_proba
        }).sort_values(by='Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Risk Level'))

        # Store input for SHAP tab
        st.session_state['last_input'] = input_encoded

# ------------------------------------------
# TAB 3: MODEL EXPLAINABILITY (SHAP)
# ------------------------------------------
with tab3:
    st.header("Global Feature Importance & Local Explanations")
    st.markdown("This section fulfills RO3 of the dissertation by providing transparency into how the AI model makes decisions.")
    
    st.subheader("Global Feature Importance")
    st.markdown("Which features drive the predictions across the *entire* patient dataset?")
    
    # SHAP Global Summary (calculated once for a sample to save time)
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(model)
        # Using a sample of 100 to make the dashboard load faster
        X_sample = X.sample(min(100, X.shape[0]), random_state=42) 
        shap_values = explainer.shap_values(X_sample)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        # Handle SHAP multi-class format differences
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, show=False) # Showing impact on class 1
        else:
            shap.summary_plot(shap_values[:, :, 1], X_sample, show=False)
            
        st.pyplot(fig)
        
    st.subheader("Local Explanation (Last Predicted Patient)")
    if 'last_input' in st.session_state:
        st.markdown("How did the model arrive at the specific prediction for the patient you just tested in the **Predict** tab?")
        
        single_shap = explainer.shap_values(st.session_state['last_input'])
        
        fig, ax = plt.subplots(figsize=(10, 3))
        
        if isinstance(single_shap, list):
            shap.waterfall_plot(shap.Explanation(values=single_shap[1][0], 
                                                 base_values=explainer.expected_value[1], 
                                                 data=st.session_state['last_input'].iloc[0], 
                                                 feature_names=X.columns), show=False)
        else:
            shap.waterfall_plot(shap.Explanation(values=single_shap[0, :, 1], 
                                                 base_values=explainer.expected_value[1], 
                                                 data=st.session_state['last_input'].iloc[0], 
                                                 feature_names=X.columns), show=False)
        st.pyplot(fig)
    else:
        st.info("Go to the 'Predict Patient Risk' tab and generate a prediction to see the patient-level SHAP explanation here.")
