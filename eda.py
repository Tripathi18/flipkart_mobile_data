import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_eda(df):
    """
    Show exploratory data analysis section.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed DataFrame
    """
    st.header("Exploratory Data Analysis")
    
    # Overview of the data
    st.subheader("Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["Data Sample", "Data Types", "Summary Statistics"])
    
    with tab1:
        st.dataframe(df.head(10))
    
    with tab2:
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtypes_df)
    
    with tab3:
        st.dataframe(df.describe().T)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    missing_values['Missing Percentage'] = (missing_values['Missing Count'] / len(df) * 100).round(2)
    
    # Create a bar chart of missing values
    fig = px.bar(
        missing_values[missing_values['Missing Count'] > 0], 
        x='Column', 
        y='Missing Percentage',
        title='Missing Values by Column (%)',
        labels={'Missing Percentage': 'Missing Percentage (%)', 'Column': 'Column Name'},
        color='Missing Percentage',
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand distribution
    st.subheader("Brand Distribution")
    
    # Get top brands by count
    top_brands = df['Brand'].value_counts().reset_index()
    top_brands.columns = ['Brand', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a bar chart
        fig = px.bar(
            top_brands.head(15), 
            x='Brand', 
            y='Count',
            title='Top 15 Brands by Number of Models',
            labels={'Count': 'Number of Models', 'Brand': 'Brand Name'},
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a pie chart
        fig = px.pie(
            top_brands.head(10),
            values='Count',
            names='Brand',
            title='Top 10 Brands by Market Share',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution
    st.subheader("Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a histogram
        fig = px.histogram(
            df,
            x='Selling Price',
            nbins=50,
            title='Distribution of Selling Prices',
            labels={'Selling Price': 'Price (₹)'},
            color_discrete_sequence=['blue']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a box plot for price distribution by price category
        fig = px.box(
            df,
            x='Price Category',
            y='Selling Price',
            title='Price Distribution by Category',
            labels={'Price Category': 'Category', 'Selling Price': 'Price (₹)'},
            color='Price Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Memory and storage analysis
    st.subheader("Memory and Storage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a histogram for memory
        memory_counts = df['Memory'].value_counts().reset_index()
        memory_counts.columns = ['Memory (GB)', 'Count']
        memory_counts = memory_counts.sort_values('Memory (GB)')
        
        fig = px.bar(
            memory_counts, 
            x='Memory (GB)', 
            y='Count',
            title='Distribution of RAM Size',
            labels={'Count': 'Number of Models', 'Memory (GB)': 'RAM Size (GB)'},
            color='Count',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a histogram for storage
        storage_counts = df['Storage'].value_counts().reset_index()
        storage_counts.columns = ['Storage (GB)', 'Count']
        storage_counts = storage_counts.sort_values('Storage (GB)')
        
        fig = px.bar(
            storage_counts, 
            x='Storage (GB)', 
            y='Count',
            title='Distribution of Storage Size',
            labels={'Count': 'Number of Models', 'Storage (GB)': 'Storage Size (GB)'},
            color='Count',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pricing by RAM and storage
    st.subheader("Pricing by RAM and Storage")
    
    # Create a scatter plot
    fig = px.scatter(
        df, 
        x='Memory', 
        y='Storage', 
        size='Selling Price', 
        color='Brand Category',
        hover_name='Model',
        hover_data=['Selling Price', 'Rating'],
        title='Price Relationship with RAM and Storage',
        labels={'Memory': 'RAM (GB)', 'Storage': 'Storage (GB)'},
        opacity=0.7
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating analysis
    st.subheader("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a histogram for ratings
        fig = px.histogram(
            df,
            x='Rating',
            nbins=20,
            title='Distribution of Ratings',
            labels={'Rating': 'Rating (out of 5)'},
            color_discrete_sequence=['orange']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a box plot for ratings by price category
        fig = px.box(
            df,
            x='Price Category',
            y='Rating',
            title='Ratings by Price Category',
            labels={'Price Category': 'Category', 'Rating': 'Rating (out of 5)'},
            color='Price Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Create a correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    
    # Create a heatmap
    fig = px.imshow(
        correlation,
        text_auto=True,
        title='Correlation Matrix of Numeric Features',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights from EDA")
    
    st.markdown("""
    Based on the exploratory data analysis, here are some key insights:
    
    1. **Brand Distribution**: There is a significant variation in the number of models offered by different brands, with some brands having a dominant market presence.
    
    2. **Price Distribution**: The mobile phone market shows a right-skewed price distribution, indicating that most phones are in the budget to mid-range categories, with fewer premium and flagship models.
    
    3. **Memory and Storage**: There appears to be a clear relationship between RAM/storage specifications and price, with higher specifications generally commanding higher prices.
    
    4. **Ratings**: Most phones have ratings between 4.0 and 4.5, suggesting overall customer satisfaction with the majority of models.
    
    5. **Correlation Insights**: There are positive correlations between price, memory, and storage, confirming that these specifications are key determinants of a phone's price.
    """)
