import streamlit as st
import pandas as pd
import joblib # For loading the trained model and scaler
import numpy as np # For numerical operations like np.expm1
import warnings

# Suppress warnings for cleaner output in Streamlit
warnings.simplefilter('ignore')

# --- 1. Load Pre-trained Resources ---
def load_resources():
    """
    Loads the pre-trained machine learning model, scaler, and a list of
    feature names from the training data.

    Returns:
        tuple: A tuple containing the loaded model, scaler, and features,
               or (None, None, None) if loading fails.
    """
    try:
        # Load the trained Random Forest model
        loaded_rf_model = joblib.load('rf_model.pkl')
        # Load the StandardScaler object
        loaded_scaler = joblib.load('scaler.pkl')
        # Load the list of feature names used during training
        loaded_features = joblib.load('model_features.pkl')
        return loaded_rf_model, loaded_scaler, loaded_features
    except FileNotFoundError:
        st.error("Saved model, scaler, or feature names not found. Please ensure 'rf_model.pkl', 'scaler.pkl', and 'model_features.pkl' are in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None, None

# Load the resources at the start of the script
loaded_rf_model, loaded_scaler, loaded_features = load_resources()

# --- 2. Data Preprocessing Function ---

# Define the categorical and numerical features based on the training data
CATEGORICAL_FEATURES = [
    'Area', 'Zone', 'Furnishing Status', 'Recommended For', 'Water Supply'
]
NUMERICAL_COLS_PRESENT = [
    'Size_In_Sqft', 'Carpet_Area_Sqft', 'Bedrooms', 'Bathrooms', 'Balcony',
    'Number_Of_Amenities', 'Floor_No', 'Total_floors_In_Building', 'Road_Connectivity',
    'gym', 'gated_community', 'intercom', 'lift', 'pet_allowed',
    'pool', 'security', 'wifi', 'gas_pipeline', 'sports_facility', 'kids_area',
    'power_backup', 'Garden', 'Fire_Support', 'Parking', 'ATM_Near_me',
    'Airport_Near_me', 'Bus_Stop__Near_me', 'Hospital_Near_me', 'Mall_Near_me',
    'Market_Near_me', 'Metro_Station_Near_me', 'Park_Near_me', 'School_Near_me',
    'Property_Age'
]

def preprocess_new_data(input_data: dict, original_df_columns: list, scaler, categorical_features: list) -> pd.DataFrame:
    """
    Preprocesses new input data to match the format expected by the trained model.
    This replicates the feature engineering steps from the training script, specifically for
    one-hot encoding and column alignment.

    Args:
        input_data (dict): A dictionary containing the raw user inputs.
        original_df_columns (list): A list of feature names from the training data,
                                    used to align columns.
        scaler: The fitted StandardScaler object used during training.
        categorical_features (list): A list of column names that are categorical.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for prediction.
    """
    # Create a DataFrame from the new input data
    new_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to categorical features
    for feature in categorical_features:
        if feature in new_df.columns:
            # Using get_dummies to create one-hot encoded columns
            new_df = pd.get_dummies(new_df, columns=[feature], drop_first=True)

    # Align columns with the training data (CRITICAL for consistent input)
    # Add any missing columns with a value of 0. This handles categories that weren't selected.
    missing_cols = set(original_df_columns) - set(new_df.columns)
    for c in missing_cols:
        new_df[c] = 0

    # Drop any extra columns that might have been created but weren't in the training set
    extra_cols = set(new_df.columns) - set(original_df_columns)
    new_df = new_df.drop(columns=list(extra_cols))
    
    # Ensure the order of columns is exactly the same as during training
    new_df = new_df[original_df_columns]

    # Scale numerical features using the pre-fitted scaler
    # This step must happen AFTER one-hot encoding and column alignment
    numerical_columns = [col for col in NUMERICAL_COLS_PRESENT if col in new_df.columns]
    if numerical_columns:
        new_df[numerical_columns] = scaler.transform(new_df[numerical_columns])

    return new_df

# --- 3. Streamlit Application Layout & Widgets ---

# Set the page configuration for a better look and feel
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="auto"
)

st.title("🏠 House Rent Prediction")
st.markdown("Enter the details of a house to get an estimated monthly rent.")

# Define dropdown options
area_options = [
    'Hingna', 'Trimurti Nagar', 'Ashirwad Nagar', 'Beltarodi', 'Besa',
    'Bharatwada', 'Boriyapura', 'Chandrakiran Nagar', 'Dabha', 'Dhantoli',
    'Dharampeth', 'Dighori', 'Duttawadi', 'Gandhibagh', 'Ganeshpeth',
    'Godhni', 'Gotal Panjri', 'Hudkeswar', 'Itwari', 'Jaitala', 'Jaripatka',
    'Kalamna', 'Kalmeshwar', 'Khamla', 'Kharbi', 'Koradi Colony',
    'Kotewada', 'Mahal', 'Manewada', 'Manish Nagar', 'Mankapur',
    'Medical Square', 'MIHAN', 'Nandanwan', 'Narendra Nagar Extension',
    'Nari Village', 'Narsala', 'Omkar Nagar', 'Parvati Nagar',
    'Pratap Nagar', 'Ram Nagar', 'Rameshwari', 'Reshim Bagh', 'Sadar',
    'Sanmarga Nagar', 'Seminary Hills', 'Shatabdi Square', 'Sitabuldi',
    'Somalwada', 'Sonegaon', 'Teka Naka', 'Vayusena Nagar',
    'Wanadongri', 'Wardsman Nagar', 'Wathoda', 'Zingabai Takli'
]
zone_options = ['East Zone', 'North Zone', 'South Zone', 'West Zone', 'Rural']
furnishing_status_options = ['Fully Furnished', 'Semi Furnished', 'Unfurnished']
recommended_for_options = ['Anyone', 'Bachelors', 'Family', 'Family and Bachelors', 'Family and Company']
water_supply_options = ['Borewell', 'Both', 'Municipal']

st.subheader("Property Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Basic Property Information")
    size_sqft = st.number_input("Size (in Sqft)", min_value=100, max_value=50000, value=1000, step=50)
    carpet_area_sqft = st.number_input("Carpet Area (in Sqft)", min_value=50, max_value=40000, value=800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    balcony = st.number_input("Balcony", min_value=0, max_value=5, value=1, step=1)
    property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5, step=1)

with col2:
    st.markdown("##### Location & Structure")
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=1, step=1)
    total_floors_building = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=5, step=1)
    road_connectivity = st.number_input("Road Connectivity (0-10)", min_value=0, max_value=10, value=7, step=1)
    number_of_amenities = st.number_input("Number of Amenities", min_value=0, max_value=30, value=5, step=1)

    area = st.selectbox("Area", options=area_options)
    zone = st.selectbox("Zone", options=zone_options)
    furnishing_status = st.selectbox("Furnishing Status", options=furnishing_status_options)
    recommended_for = st.selectbox("Recommended For", options=recommended_for_options)
    water_supply = st.selectbox("Water Supply", options=water_supply_options)

with col3:
    st.markdown("##### Amenities")
    # Using checkboxes for boolean features
    gym = st.checkbox("Gym")
    gated_community = st.checkbox("Gated Community")
    intercom = st.checkbox("Intercom")
    lift = st.checkbox("Lift")
    pet_allowed = st.checkbox("Pet Allowed")
    pool = st.checkbox("Pool")
    security = st.checkbox("Security")
    wifi = st.checkbox("Wifi")
    gas_pipeline = st.checkbox("Gas Pipeline")
    sports_facility = st.checkbox("Sports Facility")
    kids_area = st.checkbox("Kids Area")
    power_backup = st.checkbox("Power Backup")
    garden = st.checkbox("Garden")
    fire_support = st.checkbox("Fire Support")
    parking = st.checkbox("Parking")
    atm_near_me = st.checkbox("ATM Near Me")
    airport_near_me = st.checkbox("Airport Near Me")
    bus_stop_near_me = st.checkbox("Bus Stop Near Me")
    hospital_near_me = st.checkbox("Hospital Near Me")
    mall_near_me = st.checkbox("Mall Near Me")
    market_near_me = st.checkbox("Market Near Me")
    metro_station_near_me = st.checkbox("Metro Station Near Me")
    park_near_me = st.checkbox("Park Near Me")
    school_near_me = st.checkbox("School Near Me")

# --- 4. Prediction Logic and Output ---

# Button to trigger the prediction
if st.button("Predict Rent"):
    # Check if the model and other resources were loaded successfully
    if loaded_rf_model and loaded_scaler and loaded_features:
        # Create a dictionary of all user inputs
        input_data = {
            'Size_In_Sqft': size_sqft,
            'Carpet_Area_Sqft': carpet_area_sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Balcony': balcony,
            'Number_Of_Amenities': number_of_amenities,
            'Floor_No': floor_no,
            'Total_floors_In_Building': total_floors_building,
            'Road_Connectivity': road_connectivity,
            # Convert boolean checkboxes to integers (0 or 1)
            'gym': int(gym),
            'gated_community': int(gated_community),
            'intercom': int(intercom),
            'lift': int(lift),
            'pet_allowed': int(pet_allowed),
            'pool': int(pool),
            'security': int(security),
            'wifi': int(wifi),
            'gas_pipeline': int(gas_pipeline),
            'sports_facility': int(sports_facility),
            'kids_area': int(kids_area),
            'power_backup': int(power_backup),
            'Garden': int(garden),
            'Fire_Support': int(fire_support),
            'Parking': int(parking),
            'ATM_Near_me': int(atm_near_me),
            'Airport_Near_me': int(airport_near_me),
            'Bus_Stop__Near_me': int(bus_stop_near_me),
            'Hospital_Near_me': int(hospital_near_me),
            'Mall_Near_me': int(mall_near_me),
            'Market_Near_me': int(market_near_me),
            'Metro_Station_Near_me': int(metro_station_near_me),
            'Park_Near_me': int(park_near_me),
            'School_Near_me': int(school_near_me),
            'Property_Age': property_age,
            # Categorical features
            'Area': area,
            'Zone': zone,
            'Furnishing Status': furnishing_status,
            'Recommended For': recommended_for,
            'Water Supply': water_supply
        }

        try:
            # Preprocess the input data to be in the correct format for the model
            processed_input = preprocess_new_data(
                input_data,
                loaded_features, # Use the feature names saved during training
                loaded_scaler,
                CATEGORICAL_FEATURES
            )

            # Make the prediction
            # The model was trained on the log of the rent, so the prediction is also in log form
            log_predicted_rent = loaded_rf_model.predict(processed_input)[0]

            # Inverse transform the prediction using np.expm1 to get the actual rent amount
            predicted_rent = np.expm1(log_predicted_rent)

            st.subheader("Prediction Result:")
            st.success(f"Estimated Monthly Rent: **₹{predicted_rent:,.2f}**") # Display result with INR symbol

            # --- Price Classification Section ---
            st.markdown("---")
            st.subheader("Price Comparison")
            st.markdown("Enter a listed price to see if the property is underpriced, overpriced, or fairly priced based on our prediction.")

            # Get the listed price from user input, with a default value based on the prediction
            listed_price = st.number_input("Enter Listed Price (₹)", min_value=0, value=int(predicted_rent * 1.05), step=100, key='listed_price_input')

            # Define a tolerance for what is considered a "fair" price
            FAIR_PRICE_TOLERANCE = 0.10 # 10% tolerance
            
            # Calculate the fair price range
            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)

            st.info(f"A fair price for this property would typically be between **₹{lower_bound:,.2f}** and **₹{upper_bound:,.2f}**.")

            # Classify the listed price
            if listed_price < lower_bound:
                st.warning(f"**₹{listed_price:,.2f}**  This property appears to be **Underpriced**! Great deal!")
            elif listed_price > upper_bound:
                st.error(f"**₹{listed_price:,.2f}**  This property appears to be **Overpriced**! Consider negotiating.")
            else:
                st.success(f"**₹{listed_price:,.2f}**  This property appears to be **Fairly Priced**.")

        except Exception as e:
            # Catch and display any errors during the prediction process
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model files are correct.")
    else:
        # Display a warning if the model files couldn't be loaded at startup
        st.warning("Model not loaded. Please check the file paths and restart the app.")
