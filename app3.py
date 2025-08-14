
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Loading the trained model and scaler ---
@st.cache_resource
def load_resources():
    try:
        loaded_rf_model = joblib.load('rf_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        loaded_features = joblib.load('model_features.pkl')
        return loaded_rf_model, loaded_scaler, loaded_features
    except FileNotFoundError:
        st.error("Saved model, scaler, or feature names not found. Please ensure 'rf_model.pkl', 'scaler.pkl', and 'model_features.pkl' are in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None, None

loaded_rf_model, loaded_scaler, loaded_features = load_resources()

# Assuming these lists were defined in your original notebook
categorical_features = ['Area', 'Zone', 'Furnishing Status', 'Recommended For', 'Water Supply']
# This list MUST contain only the continuous numerical features that need to be scaled.
# It should NOT contain binary features (0 or 1).
continuous_numerical_features = [
    'Size_In_Sqft', 'Carpet_Area_Sqft', 'Bedrooms', 'Bathrooms', 'Balcony',
    'Number_Of_Amenities', 'Floor_No', 'Total_floors_In_Building',
]

# Define options for categorical inputs
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

# --- Streamlit App ---
st.set_page_config(page_title="Nagpur House Rent Predictor", layout="wide")

st.title("🏡 Nagpur House Rent Predictor")
st.markdown("Enter the details of a house in Nagpur to get an estimated rental price.")

if loaded_rf_model and loaded_scaler and loaded_features:
    with st.form("prediction_form"):
        st.header("Property Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            size_sqft = st.number_input("Size in Sqft", min_value=100, max_value=10000, value=1000)
            carpet_area = st.number_input("Carpet Area in Sqft", min_value=100, max_value=10000, value=900)
            bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
            bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
            balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1)
            property_age = st.number_input("Property Age (in years)", min_value=0, max_value=100, value=5)
            security_deposite = st.number_input("Security Deposite (in Rs)", min_value=0, value=10000)
        
        with col2:
            floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=2)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=10)
            num_amenities = st.number_input("Number of Amenities", min_value=0, max_value=50, value=5)
            road_connectivity = st.number_input("Road Connectivity (1-5)", min_value=1, max_value=5, value=4)
            listed_price = st.number_input("Listed Price for Comparison (in Rs)", min_value=0, value=0)
            
            # Categorical inputs
            area = st.selectbox("Area", options=area_options)
            zone = st.selectbox("Zone", options=zone_options)
            
        with col3:
            furnishing_status = st.selectbox("Furnishing Status", options=furnishing_status_options)
            recommended_for = st.selectbox("Recommended For", options=recommended_for_options)
            water_supply = st.selectbox("Water Supply", options=water_supply_options)
            
            st.markdown("---")
            st.subheader("Amenities (0 = No, 1 = Yes)")
            amenities_cols = st.columns(4)
            
            new_data_dict_amenities = {}
            # This list contains ALL binary features
            amenity_list = [
                'gym', 'gated_community', 'intercom', 'lift', 'pet_allowed', 'pool', 'security', 
                'wifi', 'gas_pipeline', 'sports_facility', 'kids_area', 'power_backup',
                'Garden', 'Fire_Support', 'Parking', 'ATM_Near_me', 'Airport_Near_me',
                'Bus_Stop__Near_me', 'Hospital_Near_me', 'Mall_Near_me',
                'Market_Near_me', 'Metro_Station_Near_me', 'Park_Near_me', 'School_Near_me'
            ]
            
            for i, amenity in enumerate(amenity_list):
                with amenities_cols[i % 4]:
                    new_data_dict_amenities[amenity] = st.checkbox(amenity.replace('_', ' ').title())

        submitted = st.form_submit_button("Predict Rent")

        if submitted:
            if listed_price == 0:
                st.warning("Please enter a listed price for a meaningful comparison.")
                
            # Create a dictionary from the user inputs
            new_data_dict = {
                'Size_In_Sqft': size_sqft,
                'Carpet_Area_Sqft': carpet_area,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Balcony': balcony,
                'Number_Of_Amenities': num_amenities,
                'Floor_No': floor_no,
                'Total_floors_In_Building': total_floors,
                'Road_Connectivity': road_connectivity,
                'Property_Age': property_age,
                'Area': area,
                'Zone': zone,
                'Furnishing Status': furnishing_status,
                'Recommended For': recommended_for,
                'Water Supply': water_supply
            }
            
            # Add binary amenities to the dictionary
            for amenity, value in new_data_dict_amenities.items():
                new_data_dict[amenity] = 1 if value else 0
                
            # Create a DataFrame
            new_df = pd.DataFrame([new_data_dict])
            
            # Apply one-hot encoding
            df_processed = pd.get_dummies(new_df, columns=categorical_features, drop_first=True)

            # Align columns with the training data
            missing_cols = set(loaded_features) - set(df_processed.columns)
            for c in missing_cols:
                df_processed[c] = 0
            df_processed = df_processed[loaded_features]

            # Scale only the continuous numerical features
            df_processed[continuous_numerical_features] = loaded_scaler.transform(df_processed[continuous_numerical_features])

            # Make prediction
            log_predicted_rent = loaded_rf_model.predict(df_processed)[0]
            predicted_rent = np.expm1(log_predicted_rent)

            st.subheader("Prediction Result")
            st.metric("Predicted Rent", f"Rs{predicted_rent:,.2f}")

            # Price Classification
            FAIR_PRICE_TOLERANCE = 0.10
            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)

            st.markdown(f"A fair price for this property would typically be between **Rs{lower_bound:,.2f}** and **Rs{upper_bound:,.2f}**.")
            
            if listed_price > 0:
                st.markdown("---")
                st.subheader("Price Comparison")
                if listed_price < lower_bound:
                    st.success(f"This property appears to be **Underpriced**! You might be getting a great deal.")
                elif listed_price > upper_bound:
                    st.error(f"This property appears to be **Overpriced**! You might want to negotiate.")
                else:
                    st.info(f"This property appears to be **Fairly Priced**.")
else:
    st.error("Model components could not be loaded. Please check your files.")
