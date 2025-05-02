import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_brand_analysis(df):
    """
    Show brand analysis section.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed DataFrame
    """
    st.header("Brand Analysis")
    
    # Introduction
    st.markdown("""
    This section provides detailed analysis of mobile phone brands, their market positioning, 
    pricing strategies, and competitive landscape.
    """)
    
    # Get top brands by count
    brand_counts = df['Brand'].value_counts()
    top_brands = brand_counts[brand_counts >= 20].index.tolist()
    
    # Brand selection
    selected_brands = st.multiselect(
        "Select brands to analyze:",
        options=top_brands,
        default=top_brands[:5]
    )
    
    # Check if brands are selected
    if not selected_brands:
        st.warning("Please select at least one brand to analyze.")
        return
    
    # Filter data based on selected brands
    filtered_df = df[df['Brand'].isin(selected_brands)]
    
    # Market share analysis
    st.subheader("Market Share Analysis")
    
    # Calculate market share based on number of models
    market_share = filtered_df['Brand'].value_counts().reset_index()
    market_share.columns = ['Brand', 'Count']
    market_share['Percentage'] = (market_share['Count'] / market_share['Count'].sum() * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a pie chart
        fig = px.pie(
            market_share, 
            values='Count', 
            names='Brand',
            title='Market Share by Number of Models',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a bar chart
        fig = px.bar(
            market_share, 
            x='Brand', 
            y='Percentage',
            title='Market Share Percentage',
            labels={'Percentage': 'Market Share (%)', 'Brand': 'Brand Name'},
            color='Brand',
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Price segment analysis
    st.subheader("Price Segment Analysis")
    
    # Calculate count by price category and brand
    price_segment = filtered_df.groupby(['Brand', 'Price Category']).size().reset_index(name='Count')
    
    # Create a grouped bar chart
    fig = px.bar(
        price_segment, 
        x='Brand', 
        y='Count',
        color='Price Category',
        title='Distribution of Price Segments by Brand',
        labels={'Count': 'Number of Models', 'Brand': 'Brand Name'},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate percentage by price category for each brand
    brand_totals = price_segment.groupby('Brand')['Count'].sum().reset_index()
    price_segment = price_segment.merge(brand_totals, on='Brand', suffixes=('', '_Total'))
    price_segment['Percentage'] = (price_segment['Count'] / price_segment['Count_Total'] * 100).round(2)
    
    # Create a 100% stacked bar chart
    fig = px.bar(
        price_segment, 
        x='Brand', 
        y='Percentage',
        color='Price Category',
        title='Percentage Distribution of Price Segments by Brand',
        labels={'Percentage': 'Percentage (%)', 'Brand': 'Brand Name'},
        barmode='stack',
        category_orders={"Price Category": ["Budget", "Entry", "Mid-range", "Premium", "Flagship"]}
    )
    fig.update_layout(yaxis=dict(tickformat='.0f', range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)
    
    # Average price by brand
    st.subheader("Average Price Analysis")
    
    # Calculate average price by brand
    avg_price = filtered_df.groupby('Brand')['Selling Price'].agg(['mean', 'min', 'max', 'std']).reset_index()
    avg_price.columns = ['Brand', 'Average Price', 'Minimum Price', 'Maximum Price', 'Price StdDev']
    avg_price = avg_price.round(2)
    
    # Create a bar chart with error bars
    fig = px.bar(
        avg_price, 
        x='Brand', 
        y='Average Price',
        title='Average Price by Brand',
        labels={'Average Price': 'Average Price (₹)', 'Brand': 'Brand Name'},
        color='Brand',
        error_y='Price StdDev'
    )
    fig.update_traces(texttemplate='₹%{y:.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Price range by brand
    fig = go.Figure()
    
    for i, brand in enumerate(avg_price['Brand']):
        fig.add_trace(go.Scatter(
            x=[brand, brand],
            y=[avg_price.loc[i, 'Minimum Price'], avg_price.loc[i, 'Maximum Price']],
            mode='lines',
            line=dict(width=4),
            name=brand
        ))
        fig.add_trace(go.Scatter(
            x=[brand],
            y=[avg_price.loc[i, 'Average Price']],
            mode='markers',
            marker=dict(size=12, symbol='diamond'),
            name=f"{brand} (Avg)",
            showlegend=False
        ))
    
    fig.update_layout(
        title='Price Range by Brand',
        xaxis_title='Brand',
        yaxis_title='Price (₹)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Specification analysis
    st.subheader("Specification Analysis")
    
    # Memory analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate average memory by brand
        avg_memory = filtered_df.groupby('Brand')['Memory'].mean().reset_index()
        avg_memory.columns = ['Brand', 'Average RAM']
        avg_memory['Average RAM'] = avg_memory['Average RAM'].round(2)
        
        # Create a bar chart
        fig = px.bar(
            avg_memory, 
            x='Brand', 
            y='Average RAM',
            title='Average RAM by Brand',
            labels={'Average RAM': 'Average RAM (GB)', 'Brand': 'Brand Name'},
            color='Brand'
        )
        fig.update_traces(texttemplate='%{y:.2f} GB', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate average storage by brand
        avg_storage = filtered_df.groupby('Brand')['Storage'].mean().reset_index()
        avg_storage.columns = ['Brand', 'Average Storage']
        avg_storage['Average Storage'] = avg_storage['Average Storage'].round(2)
        
        # Create a bar chart
        fig = px.bar(
            avg_storage, 
            x='Brand', 
            y='Average Storage',
            title='Average Storage by Brand',
            labels={'Average Storage': 'Average Storage (GB)', 'Brand': 'Brand Name'},
            color='Brand'
        )
        fig.update_traces(texttemplate='%{y:.0f} GB', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Rating analysis
    st.subheader("Rating Analysis")
    
    # Calculate average rating by brand
    avg_rating = filtered_df.groupby('Brand')['Rating'].agg(['mean', 'min', 'max', 'count']).reset_index()
    avg_rating.columns = ['Brand', 'Average Rating', 'Minimum Rating', 'Maximum Rating', 'Number of Ratings']
    avg_rating['Average Rating'] = avg_rating['Average Rating'].round(2)
    
    # Create a bar chart
    fig = px.bar(
        avg_rating, 
        x='Brand', 
        y='Average Rating',
        title='Average Rating by Brand',
        labels={'Average Rating': 'Average Rating (out of 5)', 'Brand': 'Brand Name'},
        color='Brand',
        hover_data=['Minimum Rating', 'Maximum Rating', 'Number of Ratings']
    )
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, 5]))
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount analysis
    st.subheader("Discount Analysis")
    
    # Calculate average discount by brand
    avg_discount = filtered_df.groupby('Brand')['Discount %'].mean().reset_index()
    avg_discount.columns = ['Brand', 'Average Discount']
    avg_discount['Average Discount'] = avg_discount['Average Discount'].round(2)
    
    # Create a bar chart
    fig = px.bar(
        avg_discount, 
        x='Brand', 
        y='Average Discount',
        title='Average Discount Percentage by Brand',
        labels={'Average Discount': 'Average Discount (%)', 'Brand': 'Brand Name'},
        color='Brand'
    )
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount distribution by brand
    fig = px.box(
        filtered_df, 
        x='Brand', 
        y='Discount %',
        title='Discount Percentage Distribution by Brand',
        labels={'Discount %': 'Discount (%)', 'Brand': 'Brand Name'},
        color='Brand'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Color analysis
    st.subheader("Color Analysis")
    
    # Get color counts for each brand
    color_counts = filtered_df.groupby(['Brand', 'Color']).size().reset_index(name='Count')
    
    # Get top colors for each brand
    top_colors_by_brand = color_counts.sort_values(['Brand', 'Count'], ascending=[True, False])
    top_colors_by_brand = top_colors_by_brand.groupby('Brand').head(5)
    
    # Create a grouped bar chart
    fig = px.bar(
        top_colors_by_brand, 
        x='Brand', 
        y='Count',
        color='Color',
        title='Top 5 Colors by Brand',
        labels={'Count': 'Number of Models', 'Brand': 'Brand Name'},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitive pricing analysis
    st.subheader("Competitive Pricing Analysis")
    
    # Prepare data for competitive analysis
    # Group by brand and specs
    competitive_df = filtered_df.copy()
    competitive_df['Specs'] = competitive_df['Memory'].astype(str) + "GB RAM, " + competitive_df['Storage'].astype(str) + "GB Storage"
    
    # Get common specs across selected brands
    specs_by_brand = competitive_df.groupby('Brand')['Specs'].unique()
    common_specs = set.intersection(*[set(specs) for specs in specs_by_brand])
    
    if common_specs:
        # Filter for common specs
        common_specs_df = competitive_df[competitive_df['Specs'].isin(common_specs)]
        
        # Average price by brand and specs
        avg_price_by_specs = common_specs_df.groupby(['Brand', 'Specs'])['Selling Price'].mean().reset_index()
        avg_price_by_specs['Selling Price'] = avg_price_by_specs['Selling Price'].round(2)
        
        # Create a grouped bar chart
        fig = px.bar(
            avg_price_by_specs, 
            x='Specs', 
            y='Selling Price',
            color='Brand',
            title='Price Comparison for Common Specifications',
            labels={'Selling Price': 'Average Price (₹)', 'Specs': 'Specifications'},
            barmode='group'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common specifications found across the selected brands.")
    
    # Price to specification value analysis
    st.subheader("Price to Specification Value Analysis")
    
    # Calculate price per GB of RAM
    value_df = filtered_df.copy()
    value_df['Price per GB RAM'] = value_df['Selling Price'] / value_df['Memory']
    value_df['Price per GB Storage'] = value_df['Selling Price'] / value_df['Storage']
    
    # Calculate average by brand
    ram_value = value_df.groupby('Brand')['Price per GB RAM'].mean().reset_index()
    ram_value['Price per GB RAM'] = ram_value['Price per GB RAM'].round(2)
    
    storage_value = value_df.groupby('Brand')['Price per GB Storage'].mean().reset_index()
    storage_value['Price per GB Storage'] = storage_value['Price per GB Storage'].round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a bar chart for RAM value
        fig = px.bar(
            ram_value, 
            x='Brand', 
            y='Price per GB RAM',
            title='Average Price per GB of RAM',
            labels={'Price per GB RAM': 'Price per GB RAM (₹)', 'Brand': 'Brand Name'},
            color='Brand'
        )
        fig.update_traces(texttemplate='₹%{y:.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a bar chart for storage value
        fig = px.bar(
            storage_value, 
            x='Brand', 
            y='Price per GB Storage',
            title='Average Price per GB of Storage',
            labels={'Price per GB Storage': 'Price per GB Storage (₹)', 'Brand': 'Brand Name'},
            color='Brand'
        )
        fig.update_traces(texttemplate='₹%{y:.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Overall value index
    value_df['Value Index'] = (value_df['Memory'] * value_df['Storage']) / value_df['Selling Price'] * 100
    avg_value_index = value_df.groupby('Brand')['Value Index'].mean().reset_index()
    avg_value_index['Value Index'] = avg_value_index['Value Index'].round(2)
    
    # Sort by value index (higher is better)
    avg_value_index = avg_value_index.sort_values('Value Index', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        avg_value_index, 
        x='Brand', 
        y='Value Index',
        title='Value Index by Brand (Higher is Better)',
        labels={'Value Index': 'Value Index', 'Brand': 'Brand Name'},
        color='Value Index',
        color_continuous_scale='Viridis'
    )
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand positioning summary
    st.subheader("Brand Positioning Summary")
    
    # Create a scatter plot
    fig = px.scatter(
        avg_price.merge(avg_rating, on='Brand'), 
        x='Average Price', 
        y='Average Rating',
        size='Number of Ratings',
        color='Brand',
        hover_name='Brand',
        hover_data=['Minimum Price', 'Maximum Price'],
        title='Brand Positioning: Price vs. Rating',
        labels={'Average Price': 'Average Price (₹)', 'Average Rating': 'Average Rating (out of 5)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights about Selected Brands")
    
    # Calculate overall stats
    overall_stats = pd.DataFrame({
        'Brand': selected_brands,
        'Market Share (%)': [market_share[market_share['Brand'] == brand]['Percentage'].values[0] if not market_share[market_share['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Avg Price (₹)': [avg_price[avg_price['Brand'] == brand]['Average Price'].values[0] if not avg_price[avg_price['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Avg Rating': [avg_rating[avg_rating['Brand'] == brand]['Average Rating'].values[0] if not avg_rating[avg_rating['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Avg Discount (%)': [avg_discount[avg_discount['Brand'] == brand]['Average Discount'].values[0] if not avg_discount[avg_discount['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Avg RAM (GB)': [avg_memory[avg_memory['Brand'] == brand]['Average RAM'].values[0] if not avg_memory[avg_memory['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Avg Storage (GB)': [avg_storage[avg_storage['Brand'] == brand]['Average Storage'].values[0] if not avg_storage[avg_storage['Brand'] == brand].empty else 0 for brand in selected_brands],
        'Value Index': [avg_value_index[avg_value_index['Brand'] == brand]['Value Index'].values[0] if not avg_value_index[avg_value_index['Brand'] == brand].empty else 0 for brand in selected_brands]
    })
    
    # Round values
    numeric_cols = overall_stats.columns[1:]
    overall_stats[numeric_cols] = overall_stats[numeric_cols].round(2)
    
    # Display table
    st.dataframe(overall_stats)
    
    # Generate insights text for each brand
    st.markdown("### Brand-specific Insights")
    
    for brand in selected_brands:
        brand_stats = overall_stats[overall_stats['Brand'] == brand].iloc[0]
        
        # Get main price segment
        brand_segments = price_segment[price_segment['Brand'] == brand].sort_values('Percentage', ascending=False)
        main_segment = brand_segments.iloc[0]['Price Category'] if not brand_segments.empty else "Unknown"
        
        st.markdown(f"#### {brand}")
        st.markdown(f"""
        - **Market Position**: {brand} holds a {brand_stats['Market Share (%)']:.2f}% market share among the selected brands.
        - **Pricing Strategy**: The average price of {brand} phones is ₹{brand_stats['Avg Price (₹)']:,.2f}, with an average discount of {brand_stats['Avg Discount (%)']:.2f}%.
        - **Target Segment**: {brand} primarily targets the {main_segment} price segment.
        - **Specifications**: On average, {brand} phones offer {brand_stats['Avg RAM (GB)']:.2f}GB of RAM and {brand_stats['Avg Storage (GB)']:.0f}GB of storage.
        - **Customer Satisfaction**: {brand} phones have an average rating of {brand_stats['Avg Rating']:.2f} out of 5.
        - **Value Proposition**: {brand} has a value index of {brand_stats['Value Index']:.2f}, indicating {'excellent' if brand_stats['Value Index'] > 1.5 else 'good' if brand_stats['Value Index'] > 1.0 else 'average' if brand_stats['Value Index'] > 0.5 else 'poor'} value for money.
        """)
