import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_visualizations(df):
    """
    Show interactive visualizations section.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed DataFrame
    """
    st.header("Interactive Visualizations")
    
    # Introduction
    st.markdown("""
    This section provides interactive visualizations that allow you to explore the mobile phone market in depth.
    You can filter and customize the visualizations to focus on specific aspects of interest.
    """)
    
    # Brand filters
    st.sidebar.subheader("Brand Filters")
    top_brands = df['Brand'].value_counts().head(10).index.tolist()
    selected_brands = st.sidebar.multiselect(
        "Select brands to include:",
        options=top_brands,
        default=top_brands[:5]
    )
    
    # Check if brands are selected
    if not selected_brands:
        st.warning("Please select at least one brand from the sidebar.")
        return
    
    # Filter data based on selected brands
    filtered_df = df[df['Brand'].isin(selected_brands)]
    
    # Price range filter
    st.sidebar.subheader("Price Range")
    min_price, max_price = st.sidebar.slider(
        "Select price range (₹):",
        min_value=int(df['Selling Price'].min()),
        max_value=int(df['Selling Price'].max()),
        value=(int(df['Selling Price'].min()), int(df['Selling Price'].max()))
    )
    
    # Filter by price range
    filtered_df = filtered_df[(filtered_df['Selling Price'] >= min_price) & 
                             (filtered_df['Selling Price'] <= max_price)]
    
    # RAM filter
    st.sidebar.subheader("RAM (Memory)")
    ram_options = sorted(df['Memory'].unique())
    selected_ram = st.sidebar.multiselect(
        "Select RAM sizes (GB):",
        options=ram_options,
        default=ram_options
    )
    
    # Check if RAM sizes are selected
    if not selected_ram:
        st.warning("Please select at least one RAM size from the sidebar.")
        return
    
    # Filter by RAM
    filtered_df = filtered_df[filtered_df['Memory'].isin(selected_ram)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return
    
    # Display count of phones after filtering
    st.info(f"Showing data for {len(filtered_df)} phones based on your selection.")
    
    # Price distribution by brand
    st.subheader("Price Distribution by Brand")
    
    # Create a box plot
    fig = px.box(
        filtered_df, 
        x='Brand', 
        y='Selling Price',
        title='Price Distribution by Brand',
        color='Brand',
        labels={'Selling Price': 'Price (₹)', 'Brand': 'Brand Name'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand market share
    st.subheader("Brand Market Share (Based on Number of Models)")
    
    brand_counts = filtered_df['Brand'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Count']
    brand_counts['Percentage'] = brand_counts['Count'] / brand_counts['Count'].sum() * 100
    
    # Create a pie chart
    fig = px.pie(
        brand_counts, 
        values='Count', 
        names='Brand',
        title='Brand Market Share by Number of Models',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive scatter plot
    st.subheader("Interactive Scatter Plot: RAM vs Storage vs Price")
    
    # Create a scatter plot
    fig = px.scatter(
        filtered_df, 
        x='Memory', 
        y='Storage', 
        size='Selling Price', 
        color='Brand',
        hover_name='Model',
        hover_data=['Color', 'Rating', 'Selling Price', 'Original Price'],
        title='Relationship between RAM, Storage, and Price',
        labels={'Memory': 'RAM (GB)', 'Storage': 'Storage (GB)'},
        size_max=50
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Rating
    st.subheader("Price vs Rating")
    
    # Create a scatter plot
    fig = px.scatter(
        filtered_df, 
        x='Selling Price', 
        y='Rating',
        color='Brand',
        size='Memory',
        hover_name='Model',
        hover_data=['Storage', 'Color', 'Discount %'],
        title='Relationship between Price and Rating',
        labels={'Selling Price': 'Price (₹)', 'Rating': 'Rating (out of 5)'},
        trendline='ols'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory vs Price by Brand
    st.subheader("Memory (RAM) vs Price by Brand")
    
    # Create a scatter plot
    fig = px.scatter(
        filtered_df, 
        x='Memory', 
        y='Selling Price',
        color='Brand',
        facet_col='Brand',
        hover_name='Model',
        hover_data=['Storage', 'Color', 'Rating'],
        title='Relationship between RAM and Price by Brand',
        labels={'Memory': 'RAM (GB)', 'Selling Price': 'Price (₹)'},
        trendline='ols',
        height=500
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Storage vs Price by Brand
    st.subheader("Storage vs Price by Brand")
    
    # Create a scatter plot
    fig = px.scatter(
        filtered_df, 
        x='Storage', 
        y='Selling Price',
        color='Brand',
        facet_col='Brand',
        hover_name='Model',
        hover_data=['Memory', 'Color', 'Rating'],
        title='Relationship between Storage and Price by Brand',
        labels={'Storage': 'Storage (GB)', 'Selling Price': 'Price (₹)'},
        trendline='ols',
        height=500
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount distribution
    st.subheader("Discount Distribution")
    
    # Create a histogram
    fig = px.histogram(
        filtered_df, 
        x='Discount %',
        color='Brand',
        title='Distribution of Discount Percentages',
        labels={'Discount %': 'Discount (%)'},
        marginal='box',
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap: RAM vs Storage vs Average Price
    st.subheader("Heatmap: RAM vs Storage vs Average Price")
    
    # Calculate average price for each RAM-Storage combination
    avg_price_by_specs = filtered_df.groupby(['Memory', 'Storage'])['Selling Price'].mean().reset_index()
    
    # Create a pivot table
    pivot_table = avg_price_by_specs.pivot_table(
        values='Selling Price',
        index='Memory',
        columns='Storage',
        aggfunc='mean'
    )
    
    # Create a heatmap
    fig = px.imshow(
        pivot_table,
        title='Average Price by RAM and Storage',
        labels=dict(x='Storage (GB)', y='RAM (GB)', color='Avg Price (₹)'),
        color_continuous_scale='Viridis',
        text_auto='.0f'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Color distribution by brand
    st.subheader("Color Distribution by Brand")
    
    # Get color counts for each brand
    color_counts = filtered_df.groupby(['Brand', 'Color']).size().reset_index(name='Count')
    
    # Create a grouped bar chart
    fig = px.bar(
        color_counts, 
        x='Brand', 
        y='Count',
        color='Color',
        title='Color Distribution by Brand',
        labels={'Count': 'Number of Models', 'Brand': 'Brand Name'},
        barmode='group'
    )
    
    # Handle large number of colors by limiting the height
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis section (simulated based on available data)
    st.subheader("Price Range Analysis by Specification Category")
    
    # Create Memory-Storage specification categories
    filtered_df['Spec Category'] = filtered_df['Memory'].astype(str) + "GB RAM, " + filtered_df['Storage'].astype(str) + "GB Storage"
    
    # Get average price by specification category
    spec_prices = filtered_df.groupby(['Spec Category', 'Brand'])['Selling Price'].mean().reset_index()
    
    # Sort by average price
    spec_prices = spec_prices.sort_values('Selling Price')
    
    # Create a grouped bar chart
    fig = px.bar(
        spec_prices, 
        x='Spec Category', 
        y='Selling Price',
        color='Brand',
        title='Average Price by Specification Category and Brand',
        labels={'Selling Price': 'Average Price (₹)', 'Spec Category': 'Specification Category'},
        barmode='group'
    )
    
    # Handle potentially large number of specification categories
    fig.update_layout(height=600, xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive 3D scatter plot
    st.subheader("Interactive 3D Scatter Plot: RAM vs Storage vs Price")
    
    # Create a 3D scatter plot
    fig = px.scatter_3d(
        filtered_df, 
        x='Memory', 
        y='Storage', 
        z='Selling Price',
        color='Brand',
        size='Rating',
        hover_name='Model',
        hover_data=['Color', 'Discount %'],
        title='3D Relationship between RAM, Storage, and Price',
        labels={'Memory': 'RAM (GB)', 'Storage': 'Storage (GB)', 'Selling Price': 'Price (₹)'},
        opacity=0.8
    )
    
    # Adjust layout
    fig.update_layout(scene=dict(
        xaxis_title='RAM (GB)',
        yaxis_title='Storage (GB)',
        zaxis_title='Price (₹)'
    ), height=700)
    
    st.plotly_chart(fig, use_container_width=True)
