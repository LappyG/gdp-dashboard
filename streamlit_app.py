import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page configuration
st.set_page_config(page_title="Network Anomaly Detection", layout="wide")
st.title("Network Anomaly Detection Dashboard")

def validate_input_data(df, required_features):
    """Validate if the required features are present in the uploaded data."""
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        return False, missing_features
    return True, None

def make_predictions(df, model):
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def show_results(predictions, df):
    if predictions is not None:
        st.success("Detection Complete!")
        
        # Show results summary
        anomaly_count = sum(predictions == 1)
        normal_count = sum(predictions == 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal Traffic", normal_count)
        with col2:
            st.metric("Anomalies Detected", anomaly_count)
            
        # Add results to dataframe
        df['prediction'] = predictions
        st.dataframe(df)
        
        # Allow users to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name='anomaly_detection_results.csv', mime='text/csv')

def main():
    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your network traffic data", type=['csv'])

    if uploaded_file is not None:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.header("Data Preview")
        st.dataframe(df.head())
        
        # Show feature importance if available
        try:
            feature_imp = pd.read_csv('feature_importances.csv')
            st.header("Feature Importance")
            fig = px.bar(feature_imp, x='feature', y='importance',
                        title='Feature Importance in Anomaly Detection')
            st.plotly_chart(fig)
        except Exception as e:
            st.warning("Feature importance file not found.")
        
        # Prediction section
        st.header("Anomaly Detection")
        if st.button("Run Detection"):
            try:
                model = joblib.load('random_forest_model-paa.joblib')
                required_features = model.feature_names_in_  # Model's expected features
                
                # Validate input data
                is_valid, missing_features = validate_input_data(df, required_features)
                if not is_valid:
                    st.error(f"The following required features are missing: {', '.join(missing_features)}")
                    return
                
                # Make predictions
                df = df[required_features]  # Ensure only the required features are used
                predictions = make_predictions(df, model)
                show_results(predictions, df)
            except AttributeError:
                st.error("Model does not have `feature_names_in_` attribute. Please verify the model.")
            except FileNotFoundError:
                st.error("Model file not found. Please ensure random_forest_model-paa.joblib is present.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()
