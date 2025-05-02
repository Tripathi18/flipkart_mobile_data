import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def show_statistical_analysis(df):
    """
    Show statistical analysis section.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed DataFrame
    """
    st.header("Statistical Analysis")
    
    # Introduction
    st.markdown("""
    This section performs various statistical analyses to identify factors affecting phone ratings and prices,
    tests hypotheses about the mobile phone market, and provides insights into market trends.
    """)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    correlation = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation,
        text_auto=True,
        title='Correlation Matrix of Numeric Features',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hypothesis testing
    st.subheader("Hypothesis Testing")
    
    # Hypothesis 1: Is there a significant difference in prices between memory categories?
    st.markdown("### Hypothesis 1: Is there a significant difference in prices between memory categories?")
    
    # Perform ANOVA test
    memory_groups = [df[df['Memory Category'] == cat]['Selling Price'] for cat in df['Memory Category'].unique()]
    memory_groups = [group.dropna() for group in memory_groups]
    
    # Only perform the test if we have at least two groups with data
    valid_groups = [group for group in memory_groups if len(group) > 0]
    
    if len(valid_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*valid_groups)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", round(f_stat, 2))
        with col2:
            st.metric("p-value", f"{p_value:.10f}")
        
        # Interpret the result
        if p_value < 0.05:
            st.success("The p-value is less than 0.05, so we reject the null hypothesis. There is a significant difference in prices between different memory categories.")
        else:
            st.info("The p-value is greater than 0.05, so we fail to reject the null hypothesis. There is no significant difference in prices between different memory categories.")
        
        # Show boxplot
        fig = px.box(
            df, 
            x='Memory Category', 
            y='Selling Price',
            title='Selling Price by Memory Category',
            color='Memory Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough valid groups to perform ANOVA test.")
    
    # Hypothesis 2: Is there a correlation between phone price and rating?
    st.markdown("### Hypothesis 2: Is there a correlation between phone price and rating?")
    
    # Calculate Pearson correlation
    price_rating_corr = df['Selling Price'].corr(df['Rating'])
    
    # Calculate p-value
    corr, p_value = stats.pearsonr(df['Selling Price'].dropna(), df['Rating'].dropna())
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Correlation coefficient", round(corr, 2))
    with col2:
        st.metric("p-value", f"{p_value:.10f}")
    
    # Interpret the result
    if p_value < 0.05:
        if corr > 0:
            st.success("The p-value is less than 0.05, so we reject the null hypothesis. There is a significant positive correlation between phone price and rating.")
        else:
            st.success("The p-value is less than 0.05, so we reject the null hypothesis. There is a significant negative correlation between phone price and rating.")
    else:
        st.info("The p-value is greater than 0.05, so we fail to reject the null hypothesis. There is no significant correlation between phone price and rating.")
    
    # Show scatter plot
    fig = px.scatter(
        df, 
        x='Selling Price', 
        y='Rating',
        title='Relationship between Price and Rating',
        trendline='ols',
        color='Brand Category',
        hover_data=['Model', 'Memory', 'Storage']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Hypothesis 3: Do discounts vary significantly across price categories?
    st.markdown("### Hypothesis 3: Do discounts vary significantly across price categories?")
    
    # Perform ANOVA test
    discount_groups = [df[df['Price Category'] == cat]['Discount %'].dropna() for cat in df['Price Category'].unique()]
    
    # Only perform the test if we have at least two groups with data
    valid_groups = [group for group in discount_groups if len(group) > 0]
    
    if len(valid_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*valid_groups)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", round(f_stat, 2))
        with col2:
            st.metric("p-value", f"{p_value:.10f}")
        
        # Interpret the result
        if p_value < 0.05:
            st.success("The p-value is less than 0.05, so we reject the null hypothesis. Discounts vary significantly across price categories.")
        else:
            st.info("The p-value is greater than 0.05, so we fail to reject the null hypothesis. Discounts do not vary significantly across price categories.")
        
        # Show boxplot
        fig = px.box(
            df, 
            x='Price Category', 
            y='Discount %',
            title='Discount Percentage by Price Category',
            color='Price Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough valid groups to perform ANOVA test.")
    
    # Multiple regression analysis
    st.subheader("Multiple Regression Analysis")
    
    st.markdown("""
    Let's perform a multiple regression analysis to understand which factors significantly influence the price of mobile phones.
    """)
    
    # Prepare data for regression (remove missing values)
    reg_df = df[['Selling Price', 'Memory', 'Storage', 'Rating']].dropna()
    
    # Fit the model
    X = sm.add_constant(reg_df[['Memory', 'Storage', 'Rating']])
    y = reg_df['Selling Price']
    model = sm.OLS(y, X).fit()
    
    # Display regression results
    st.text("Multiple Regression Results:")
    st.text(model.summary().as_text())
    
    # Extract and display coefficients
    coefficients = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'p-value': model.pvalues.values
    })
    
    # Create a bar chart of coefficients
    fig = px.bar(
        coefficients[1:],  # Exclude constant
        x='Variable', 
        y='Coefficient',
        title='Regression Coefficients (Impact on Price)',
        color='p-value',
        color_continuous_scale='RdBu_r',
        labels={'Coefficient': 'Impact on Price (₹)', 'Variable': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    **Interpretation of Regression Results:**
    
    - **Memory (RAM)**: The coefficient represents the average increase in price for each additional GB of RAM, holding other variables constant.
    - **Storage**: The coefficient represents the average increase in price for each additional GB of storage, holding other variables constant.
    - **Rating**: The coefficient represents the average change in price for each one-unit increase in rating, holding other variables constant.
    
    Statistical significance is indicated by the p-value:
    - p-value < 0.05: The variable has a statistically significant effect on price
    - p-value ≥ 0.05: The variable does not have a statistically significant effect on price
    """)
    
    # Top 5 brands analysis
    st.subheader("Top 5 Brands: Statistical Comparison")
    
    # Get top 5 brands by count
    top_5_brands = df['Brand'].value_counts().head(5).index.tolist()
    top_brands_df = df[df['Brand'].isin(top_5_brands)]
    
    # Calculate statistics for top 5 brands
    brand_stats = top_brands_df.groupby('Brand').agg({
        'Selling Price': ['mean', 'std', 'min', 'max'],
        'Rating': ['mean', 'std', 'min', 'max'],
        'Discount %': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-level columns
    brand_stats.columns = ['Brand', 'Avg Price', 'Price StdDev', 'Min Price', 'Max Price',
                          'Avg Rating', 'Rating StdDev', 'Min Rating', 'Max Rating',
                          'Avg Discount', 'Discount StdDev']
    
    # Round the values
    numeric_cols = brand_stats.columns[1:]
    brand_stats[numeric_cols] = brand_stats[numeric_cols].round(2)
    
    # Display the statistics
    st.dataframe(brand_stats)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Average price by brand
        fig = px.bar(
            brand_stats, 
            x='Brand', 
            y='Avg Price',
            title='Average Price by Brand',
            color='Brand',
            error_y='Price StdDev'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average rating by brand
        fig = px.bar(
            brand_stats, 
            x='Brand', 
            y='Avg Rating',
            title='Average Rating by Brand',
            color='Brand',
            error_y='Rating StdDev'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA test for price differences among top 5 brands
    st.markdown("### ANOVA Test: Price Differences Among Top 5 Brands")
    
    # Perform ANOVA test
    brand_price_groups = [top_brands_df[top_brands_df['Brand'] == brand]['Selling Price'].dropna() 
                          for brand in top_5_brands]
    
    # Only perform the test if we have at least two groups with data
    valid_groups = [group for group in brand_price_groups if len(group) > 0]
    
    if len(valid_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*valid_groups)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", round(f_stat, 2))
        with col2:
            st.metric("p-value", f"{p_value:.10f}")
        
        # Interpret the result
        if p_value < 0.05:
            st.success("The p-value is less than 0.05, so we reject the null hypothesis. There is a significant difference in prices among the top 5 brands.")
        else:
            st.info("The p-value is greater than 0.05, so we fail to reject the null hypothesis. There is no significant difference in prices among the top 5 brands.")
    else:
        st.warning("Not enough valid groups to perform ANOVA test.")
    
    # Key findings
    st.subheader("Key Statistical Findings")
    
    st.markdown("""
    Based on the statistical analysis, here are the key findings:
    
    1. **Price Determinants**: Memory (RAM) and storage are significant determinants of mobile phone prices, with higher specifications leading to higher prices.
    
    2. **Rating-Price Relationship**: There is a statistically significant relationship between price and rating, indicating that higher-priced phones tend to have slightly higher ratings.
    
    3. **Discount Variation**: Discounts vary significantly across price categories, with different pricing strategies applied to different market segments.
    
    4. **Brand Differentiation**: There are statistically significant differences in pricing among the top brands, indicating different market positioning and target segments.
    
    5. **Regression Model**: The multiple regression model provides a statistical framework for understanding how different specifications contribute to the price of a mobile phone.
    """)
