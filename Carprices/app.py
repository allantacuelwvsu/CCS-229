import streamlit as st
import pandas as pd
import joblib

# Set page title
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Assessment", "Dataset Checker", "Prediction"])

# Assessment Page
if page == "Assessment":
    st.title("Model Assessment")
    st.write("This page will display performance metrics and comparisons for our trained models.")

    # Placeholder for metrics (to be filled in later)
    st.subheader("Performance Metrics")
    st.write("Model evaluation results will go here.")

    # Placeholder for graphs (to be added later)
    st.subheader("Visualizations")
    st.write("Performance graphs will be displayed here.")
    
    # Placeholder for model comparison (to be added later)
    st.subheader("Model Comparison")
    st.write("Model evaluation results will go here.")

# Dataset Checker Page
elif page == "Dataset Checker":
    st.title("Dataset Checker")
    st.write("View the raw dataset or the anomaly-free, cleaned version.")
    st.write("To save time, only a portion of the dataset is loaded.")
    st.write("Actual dataset has 50000+ rows.")

    # Load a small portion of the dataset first
    @st.cache_data
    def load_raw_data():
        return pd.read_csv("carprices.csv", nrows=1000)  # Load first 100 rows only

    @st.cache_data
    def load_cleaned_data():
        df = pd.read_csv("carprices.csv", nrows=1000)
        df = df[df['price'] <= 100000]
        df = df[df['price'] >= 1000]
        columns = ['year', 'manufacturer', 'odometer', 'fuel', 'transmission', 'state', 'price']
        df = df[columns]
        df = df[df['price'] > 0].dropna(subset=['price'])
        df['odometer'] = df['odometer'].fillna(0)
        df[['manufacturer', 'fuel', 'transmission', 'state']] = df[['manufacturer', 'fuel', 'transmission', 'state']].fillna('Unknown')
        df = df.dropna()
        return df

    # Selection box to choose dataset version
    dataset_option = st.radio("Choose Dataset Version:", ["Raw", "Cleaned  (Some indices are missing because of cleaning)"])
    
    # Input box for number of rows to display
    num_rows = st.slider("Number of rows to display:", min_value=5, max_value=1000, value=100, step=5)

    # Toggle between head() and tail()
    view_type = st.radio("View Type:", ["Head", "Tail"])

    # Load and display dataset based on selection
    if dataset_option == "Raw":
        data = load_raw_data()
    else:
        data = load_cleaned_data()

    # Display head() or tail() based on selection
    if view_type == "Head":
        st.dataframe(data.head(num_rows))
    elif view_type == "Tail":
        st.dataframe(data.tail(num_rows).iloc[::-1])
        
# Predictor Page
elif page == "Prediction":
    st.title("Car Price Predictor")
    st.write("Use this page to predict car prices based on certain features.")

    # Load trained model
    model_option = st.radio("Choose Model:", ["Logistic Linear Regression", "Random Forest"])
    # Load and display model based on selection
    if model_option == "Logistic Linear Regression":
        model = joblib.load("models/logreg.pkl")
    elif model_option == "Random Forest":
        model = joblib.load("models/randomforest.pkl")
    
    # Manual mapping of dictionaries for tokenized values (trained model uses these mappings)
    manufacturer_map = {'jeep': 0, 'bmw': 1, 'dodge': 2, 'chevrolet': 3, 'ford': 4, 'honda': 5, 'toyota': 6, 'nissan': 7, 'subaru': 8, 'gmc': 9, 'volkswagen': 10, 'kia': 11, 'acura': 12, 'ram': 13, 'chrysler': 14, 'hyundai': 15, 'cadillac': 16, 'volvo': 17, 'mini': 18, 'mercedes-benz': 19, 'Unknown': 20, 'audi': 21, 'mazda': 22, 'pontiac': 23, 'buick': 24, 'infiniti': 25, 'mitsubishi': 26, 'rover': 27, 'lincoln': 28, 'lexus': 29, 'fiat': 30, 'jaguar': 31, 'mercury': 32, 'saturn': 33, 'datsun': 34, 'porche': 35, 'tesla': 36, 'harley-davidson': 37, 'ferrari': 38, 'land rover': 39, 'alfa-romeo': 40, 'morgan': 41, 'aston-martin': 42}
    fuel_map = {'gas': 0, 'diesel': 1, 'Unknown': 2, 'hybrid': 3, 'other': 4, 'electric': 5}
    transmission_map = {'automatic': 0, 'manual': 1, 'other': 2, 'Unknown': 3}
    state_map = {state: idx for idx, state in enumerate(['az', 'or', 'sc', 'me', 'fl', 'mt', 'wi', 'ia', 'al', 'sd', 'tx', 'va', 'nc', 'ca', 'ny', 'md', 'tn', 'ut', 'ma', 'vt', 'ga', 'ak', 'oh', 'wa', 'mi', 'ok', 'pa', 'id', 'wy', 'mn', 'ar', 'wv', 'ms', 'mo', 'nj', 'ks', 'hi', 'il', 'ri', 'ne', 'nv', 'nd', 'la', 'ct', 'nm', 'co', 'ky', 'de', 'in', 'nh', 'dc'])}

    # User input fields
    year = st.number_input("Year of Manufacture", min_value=1900, max_value=2025, value=2015)
    manufacturer = st.selectbox("Manufacturer", list(manufacturer_map.keys()))
    odometer = st.number_input("Odometer (miles)", min_value=0, value=50000)
    fuel = st.selectbox("Fuel Type", list(fuel_map.keys())  )
    transmission = st.selectbox("Transmission", list(transmission_map.keys()))
    state = st.selectbox("State", list(state_map.keys()))

    # Convert user input to encoded values
    input_data = {
        'year': year,
        'manufacturer': manufacturer_map[manufacturer],
        'odometer': odometer,
        'fuel': fuel_map[fuel],
        'transmission': transmission_map[transmission],
        'state': state_map[state]
    }

    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    predicted_price = model.predict(input_df)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")