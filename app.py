import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress Streamlit warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'fraud_detection_model.pkl' is in the same directory.")
        return None

def create_gauge_chart(probability):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Fraud Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered fraud detection system using Random Forest</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This application uses a Random Forest classifier trained with SMOTE 
        to detect fraudulent credit card transactions.
        
        **Features:**
        - Real-time fraud prediction
        - Probability scoring
        - Batch CSV upload
        - Interactive visualizations
        """
    )
    
    st.sidebar.header("Model Stats")
    st.sidebar.metric("Algorithm", "Random Forest")
    st.sidebar.metric("Features", "30 (PCA + Time + Amount)")
    st.sidebar.metric("Training Method", "SMOTE Balanced")
    
    # Main content
    tabs = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "ℹ️ How to Use"])
    
    # Tab 1: Single Prediction
    with tabs[0]:
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Transaction Data")
            
            # Time and Amount
            time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            
            # V1-V28 PCA features
            st.write("**PCA Features (V1-V28)**")
            st.caption("These are principal components from PCA transformation of original features")
            
            v_features = {}
            cols = st.columns(4)
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                with cols[col_idx]:
                    v_features[f'V{i}'] = st.number_input(
                        f'V{i}', 
                        value=0.0, 
                        format="%.6f",
                        key=f'v{i}'
                    )
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Time': [time],
                'Amount': [amount],
                **{k: [v] for k, v in v_features.items()}
            })
            
            # Reorder columns to match training data
            expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            input_data = input_data[expected_columns]
            
            if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
                with st.spinner("Analyzing transaction..."):
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0][1]
                    
                    with col2:
                        st.subheader("Results")
                        
                        # Display result
                        if prediction == 1:
                            st.markdown('<div class="fraud-alert">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="safe-alert">✅ LEGITIMATE</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Gauge chart
                        fig = create_gauge_chart(probability)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional metrics
                        st.metric("Confidence", f"{max(probability, 1-probability)*100:.2f}%")
                        
                        # Risk assessment
                        if probability < 0.3:
                            risk = "Low Risk"
                            color = "🟢"
                        elif probability < 0.7:
                            risk = "Medium Risk"
                            color = "🟡"
                        else:
                            risk = "High Risk"
                            color = "🔴"
                        
                        st.info(f"{color} Risk Level: **{risk}**")
    
    # Tab 2: Batch Prediction (MODIFIED FOR MEMORY EFFICIENCY)
    with tabs[1]:
        st.header("Batch Transaction Analysis")
        st.write("Upload a CSV file with transaction data for batch analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain Time, V1-V28, and Amount columns"
        )
        
        if uploaded_file is not None:
            required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            
            if st.button("🚀 Analyze All Transactions", type="primary"):
                results_list = []
                total_rows = 0
                
                # Use a progress bar to show status
                progress_bar = st.progress(0, text="Starting analysis...")

                try:
                    # Estimate total lines for progress bar without loading all into memory
                    # This is a bit of a workaround but is memory efficient
                    uploaded_file.seek(0)
                    num_lines = sum(1 for _ in uploaded_file)
                    uploaded_file.seek(0) # Reset file pointer to the beginning

                    # Process the file in chunks of 50,000 rows
                    chunk_size = 50000
                    
                    for i, chunk in enumerate(pd.read_csv(uploaded_file, chunksize=chunk_size)):
                        current_rows_processed = i * chunk_size + len(chunk)
                        
                        # Update progress
                        progress_text = f"Analyzing... Processed {current_rows_processed} / {num_lines - 1} transactions"
                        progress_bar.progress(current_rows_processed / (num_lines -1), text=progress_text)

                        # Validate columns in the chunk
                        missing_cols = [col for col in required_cols if col not in chunk.columns]
                        if missing_cols:
                            st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
                            st.stop()
                        
                        # Make predictions on the chunk
                        X = chunk[required_cols]
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1]
                        
                        # Add results to the chunk
                        chunk['Prediction'] = predictions
                        chunk['Fraud_Probability'] = probabilities
                        chunk['Result'] = chunk['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})
                        
                        results_list.append(chunk)

                    progress_bar.empty() # Remove the progress bar
                    
                    # Combine all processed chunks into a single dataframe
                    df = pd.concat(results_list, ignore_index=True)
                    
                    st.success(f"✅ Analysis complete for {len(df)} transactions!")
                    
                    # --- From here, the rest of your visualization code is the same ---
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    fraud_count = df['Prediction'].sum()
                    fraud_pct = (fraud_count / len(df)) * 100
                    
                    col1.metric("Total Transactions", len(df))
                    col2.metric("Fraudulent", fraud_count)
                    col3.metric("Legitimate", len(df) - fraud_count)
                    col4.metric("Fraud Rate", f"{fraud_pct:.2f}%")
                    
                    # Visualizations
                    st.subheader("Analysis Results")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Legitimate', 'Fraud'],
                            values=[len(df) - fraud_count, fraud_count],
                            marker_colors=['#00cc00', '#ff4b4b']
                        )])
                        fig_pie.update_layout(title="Transaction Distribution")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with viz_col2:
                        fig_hist = go.Figure(data=[go.Histogram(
                            x=df['Fraud_Probability'],
                            nbinsx=50,
                            marker_color='#1f77b4'
                        )])
                        fig_hist.update_layout(
                            title="Fraud Probability Distribution",
                            xaxis_title="Probability",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Show results table
                    st.subheader("Detailed Results")
                    filter_option = st.selectbox(
                        "Filter results:",
                        ["All Transactions", "Fraudulent Only", "Legitimate Only"]
                    )
                    
                    if filter_option == "Fraudulent Only":
                        display_df = df[df['Prediction'] == 1]
                    elif filter_option == "Legitimate Only":
                        display_df = df[df['Prediction'] == 0]
                    else:
                        display_df = df
                    
                    display_columns = display_df[['Time', 'Amount', 'Result', 'Fraud_Probability']].copy()
                    display_columns['Amount'] = display_columns['Amount'].apply(lambda x: f'${x:.2f}')
                    display_columns['Fraud_Probability'] = display_columns['Fraud_Probability'].apply(lambda x: f'{x:.4f}')
                    
                    st.dataframe(display_columns, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    # Tab 3: How to Use
    with tabs[2]:
        st.header("How to Use This Application")
        
        st.markdown("""
        ### 📝 Input Format
        
        This model expects transaction data with the following features:
        
        1. **Time**: Seconds elapsed between this transaction and the first transaction
        2. **V1-V28**: PCA-transformed features (anonymized for privacy)
        3. **Amount**: Transaction amount in dollars
        
        ### 🔍 Single Prediction
        
        1. Navigate to the "Single Prediction" tab
        2. Enter transaction details (Time, Amount, and V1-V28 features)
        3. Click "Analyze Transaction"
        4. View the prediction result and fraud probability
        
        ### 📊 Batch Prediction
        
        1. Navigate to the "Batch Prediction" tab
        2. Upload a CSV file with your transaction data
        3. Click "Analyze All Transactions"
        4. Review the analysis and download results
        
        ### 📋 CSV Format Example
        
        Your CSV should have columns in this order:
        ```
        Time,V1,V2,V3,...,V28,Amount
        0.0,-1.359,0.321,...,0.015,149.62
        ```
        
        ### 🎯 Model Information
        
        - **Algorithm**: Random Forest Classifier
        - **Balancing Method**: SMOTE (Synthetic Minority Over-sampling)
        - **Training Features**: 30 features
        - **Output**: Binary classification (Fraud/Legitimate) + Probability score
        
        ### ⚠️ Important Notes
        
        - This is a demonstration model trained on public data
        - Always verify high-risk transactions through additional means
        - The model provides probability scores to help assess confidence
        - False positives may occur - use human judgment for final decisions
        """)
        
        st.info("💡 **Tip**: For best results, ensure your input data is preprocessed the same way as the training data.")

if __name__ == "__main__":
    main()
