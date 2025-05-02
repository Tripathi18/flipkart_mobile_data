import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from data_processor import load_and_clean_data, preprocess_data
from eda import show_eda
from statistical_analysis import show_statistical_analysis
from model import show_price_prediction
from visualizations import show_visualizations
from brand_analysis import show_brand_analysis

# Set page configuration
st.set_page_config(
    page_title="Mobile Phone Market Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📱 Mobile Phone Market Analysis Platform")
st.markdown("""
This platform provides comprehensive analysis of the mobile phone market based on Flipkart data.
Explore pricing trends, specifications, brand comparisons, and even predict phone prices!
""")

# Load and process data
@st.cache_data
def get_data():
    df1 = pd.read_csv("Flipkart_Mobiles.csv")
    df2 = pd.read_csv("Flipkart_mobile_brands_scraped_data.csv")
    
    # Combine the two datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Clean and preprocess data
    cleaned_df = load_and_clean_data(combined_df)
    processed_df = preprocess_data(cleaned_df)
    
    return processed_df

# Loading message
with st.spinner("Loading and processing data..."):
    df = get_data()

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["Overview", 
     "Exploratory Data Analysis", 
     "Statistical Analysis", 
     "Price Prediction Model", 
     "Interactive Visualizations",
     "Brand Analysis"]
)

# Display content based on selection
if page == "Overview":
    st.header("Dataset Overview")
    
    # Display dataset information
    st.subheader("Dataset Structure")
    st.dataframe(df.head())
    
    # Display dataset statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", df.shape[0])
        st.metric("Unique Brands", df['Brand'].nunique())
        st.metric("Unique Models", df['Model'].nunique())
    
    with col2:
        st.metric("Price Range", f"₹{df['Selling Price'].min():,} - ₹{df['Selling Price'].max():,}")
        st.metric("Average Rating", round(df['Rating'].mean(), 2))
        st.metric("Average Price", f"₹{int(df['Selling Price'].mean()):,}")
    
    # Create distribution of prices
    st.subheader("Price Distribution")
    fig = px.histogram(df, x="Selling Price", nbins=50, 
                       title="Distribution of Mobile Phone Prices",
                       labels={"Selling Price": "Price (₹)"})
    fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Create bar chart of brands
    st.subheader("Brand Distribution")
    brand_counts = df['Brand'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Count']
    fig = px.bar(brand_counts.head(15), x='Brand', y='Count',
                title="Top 15 Mobile Phone Brands",
                labels={"Count": "Number of Models", "Brand": "Brand Name"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key insights
    st.subheader("Key Insights")
    st.markdown("""
    - The dataset contains information about various mobile phone models available on Flipkart.
    - Key attributes include brand, model, specifications (color, memory, storage), rating, and pricing.
    - The data can be used to analyze pricing trends, brand positioning, and factors affecting prices.
    - Use the navigation panel to explore detailed analyses and interactive visualizations.
    """)

elif page == "Exploratory Data Analysis":
    show_eda(df)

elif page == "Statistical Analysis":
    show_statistical_analysis(df)

elif page == "Price Prediction Model":
    show_price_prediction(df)

elif page == "Interactive Visualizations":
    show_visualizations(df)

elif page == "Brand Analysis":
    show_brand_analysis(df)

# Footer
st.markdown("---")
st.markdown("📱 Mobile Phone Market Analysis Platform | Data sourced from Flipkart")
