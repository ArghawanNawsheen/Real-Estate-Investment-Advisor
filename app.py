
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_models():
    classification_model = joblib.load(
        "saved_models/classification_model.pkl"
    )
    regression_model = joblib.load(
        "saved_models/regression_model.pkl"
    )
    return classification_model, regression_model

classification_model, regression_model = load_models()

st.title("🏠 Real Estate Investment Advisor")
st.write(
    "Analyze a property's investment potential and estimated "
    "future value using Machine Learning."
)

st.info(
    "Enter property details below. Results are model estimates, "
    "not guarantees of investment returns."
)

st.header("Property Details")

with st.form("property_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.text_input("State", "Maharashtra")
        city = st.text_input("City", "Pune")
        locality = st.text_input("Locality", "Hinjewadi")

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "House", "Villa", "Plot", "Other"]
        )

        bhk = st.number_input("BHK", 1, 20, 2)
        size = st.number_input("Size (Sq. Ft.)", 100, 100000, 1000)
        price = st.number_input(
            "Current Price (Lakhs)", 0.1, 100000.0, 50.0
        )

    with col2:
        current_year = datetime.now().year

        year_built = st.number_input(
            "Year Built", 1800, current_year, 2020
        )

        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"]
        )

        floor = st.number_input("Floor Number", 0, 200, 2)
        total_floors = st.number_input("Total Floors", 1, 200, 10)

        schools = st.number_input("Nearby Schools", 0, 100, 3)
        hospitals = st.number_input("Nearby Hospitals", 0, 100, 2)

        transport = st.selectbox(
            "Public Transport Accessibility",
            ["Low", "Medium", "High"]
        )

    with col3:
        parking = st.selectbox("Parking Space", ["No", "Yes"])
        security = st.selectbox("Security", ["No", "Yes"])

        amenities = st.selectbox(
            "Amenities", ["Basic", "Moderate", "Premium"]
        )

        facing = st.selectbox(
            "Facing", ["North", "South", "East", "West", "Other"]
        )

        owner = st.selectbox(
            "Owner Type", ["Owner", "Builder", "Dealer", "Other"]
        )

        availability = st.selectbox(
            "Availability Status",
            ["Ready_to_Move", "Under_Construction", "Available", "Other"]
        )

    submitted = st.form_submit_button("Analyze Investment")

if submitted:

    if floor > total_floors:
        st.error("Floor number cannot exceed total floors.")

    else:
        age = max(0, current_year - int(year_built))

        property_data = pd.DataFrame([{
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": int(bhk),
            "Size_in_SqFt": int(size),
            "Price_in_Lakhs": float(price),
            "Year_Built": int(year_built),
            "Furnished_Status": furnished,
            "Floor_No": int(floor),
            "Total_Floors": int(total_floors),
            "Age_of_Property": int(age),
            "Nearby_Schools": int(schools),
            "Nearby_Hospitals": int(hospitals),
            "Public_Transport_Accessibility": transport,
            "Parking_Space": parking,
            "Security": security,
            "Amenities": amenities,
            "Facing": facing,
            "Owner_Type": owner,
            "Availability_Status": availability
        }])

        try:
            investment = classification_model.predict(property_data)[0]
            future_price = regression_model.predict(property_data)[0]

            st.header("📊 Prediction Results")

            if int(investment) == 1:
                st.success("Model Prediction: Good Investment")
            else:
                st.warning("Model Prediction: Not a Good Investment")

            st.metric(
                "Estimated Price After 5 Years",
                f"₹ {future_price:.2f} Lakhs"
            )

            st.subheader("Property Summary")
            st.dataframe(property_data, use_container_width=True)

            st.caption(
                "The investment label was derived from the dataset's "
                "investment score. The future-price target uses a fixed "
                "appreciation formula. Predictions are not guaranteed "
                "real-world investment outcomes."
            )

        except Exception as error:
            st.error("Prediction failed. Please check the model inputs.")
            st.exception(error)
