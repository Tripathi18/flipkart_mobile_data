import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

def show_price_prediction(df):
    """
    Show price prediction model section.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed DataFrame
    """
    st.header("Mobile Phone Price Prediction Model")
    
    # Introduction
    st.markdown("""
    This section builds and evaluates machine learning models to predict mobile phone prices based on their specifications.
    The models can help understand which features have the most impact on price and predict prices for new phone configurations.
    """)
    
    # Data preparation
    st.subheader("Data Preparation")
    
    # Select relevant features
    features = ['Brand Category', 'Memory', 'Storage', 'Rating']
    target = 'Selling Price'
    
    # Prepare the data
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle missing values
    X = X.fillna({'Rating': X['Rating'].median()})
    data_for_model = pd.concat([X, y], axis=1).dropna()
    
    X = data_for_model[features]
    y = data_for_model[target]
    
    # Display data sample
    st.write("Sample data for modeling:")
    st.dataframe(data_for_model.head())
    
    # Show feature distributions
    st.subheader("Feature Distributions")
    
    # Create distribution plots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Memory Distribution", "Storage Distribution", 
                                                      "Rating Distribution", "Price Distribution"))
    
    fig.add_trace(go.Histogram(x=data_for_model['Memory'], name='Memory'), row=1, col=1)
    fig.add_trace(go.Histogram(x=data_for_model['Storage'], name='Storage'), row=1, col=2)
    fig.add_trace(go.Histogram(x=data_for_model['Rating'], name='Rating'), row=2, col=1)
    fig.add_trace(go.Histogram(x=data_for_model['Selling Price'], name='Price'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Use SelectKBest to find the best features
    X_numeric = data_for_model[['Memory', 'Storage', 'Rating']]
    
    # Create a feature selector
    selector = SelectKBest(f_regression, k=3)
    selector.fit(X_numeric, y)
    
    # Get scores and p-values
    feature_scores = pd.DataFrame({
        'Feature': X_numeric.columns,
        'Score': selector.scores_,
        'P-value': selector.pvalues_
    })
    
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Display feature scores
    st.write("Feature importance scores:")
    st.dataframe(feature_scores)
    
    # Create a bar chart of feature scores
    fig = px.bar(
        feature_scores, 
        x='Feature', 
        y='Score',
        title='Feature Importance Scores',
        color='Score',
        labels={'Score': 'F-Score', 'Feature': 'Feature Name'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model building and evaluation
    st.subheader("Model Building and Evaluation")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_features = ['Memory', 'Storage', 'Rating']
    categorical_features = ['Brand Category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create model pipelines
    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }
    
    # Fit models and store results
    results = {}
    
    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ['Model', 'RMSE', 'MAE', 'R-squared']
    
    # Round values
    results_df[['RMSE', 'MAE', 'R-squared']] = results_df[['RMSE', 'MAE', 'R-squared']].round(2)
    
    # Display results
    st.write("Model Evaluation Results:")
    st.dataframe(results_df)
    
    # Create visualization of model performance
    fig = make_subplots(rows=1, cols=3, subplot_titles=("RMSE (Lower is Better)", 
                                                      "MAE (Lower is Better)", 
                                                      "R-squared (Higher is Better)"))
    
    fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE'), row=1, col=1)
    fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE'), row=1, col=2)
    fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['R-squared'], name='R-squared'), row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    
    # Extract feature importance from Random Forest model
    rf_model = models['Random Forest'].named_steps['model']
    
    # Get the preprocessor
    preprocessor = models['Random Forest'].named_steps['preprocessor']
    
    # Get feature names
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out([categorical_features[0]])
    feature_names = np.concatenate([numeric_features, cat_feature_names])
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    })
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Display feature importance
    st.write("Feature Importance from Random Forest Model:")
    st.dataframe(feature_importance)
    
    # Create a bar chart of feature importance
    fig = px.bar(
        feature_importance.head(10), 
        x='Feature', 
        y='Importance',
        title='Top 10 Features by Importance',
        color='Importance',
        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price prediction tool
    st.subheader("Price Prediction Tool")
    
    st.markdown("""
    Use this tool to predict the price of a mobile phone based on its specifications.
    Adjust the parameters below and see the predicted price from different models.
    """)
    
    # Get the best model based on R-squared
    best_model_name = results_df.loc[results_df['R-squared'].idxmax(), 'Model']
    
    # Create input widgets
    col1, col2 = st.columns(2)
    
    with col1:
        brand_category = st.selectbox(
            "Brand Category",
            options=df['Brand Category'].unique()
        )
        memory = st.slider(
            "Memory (RAM) in GB",
            min_value=1,
            max_value=16,
            value=6,
            step=1
        )
    
    with col2:
        storage = st.slider(
            "Storage in GB",
            min_value=16,
            max_value=512,
            value=128,
            step=16
        )
        rating = st.slider(
            "Rating (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'Brand Category': [brand_category],
        'Memory': [memory],
        'Storage': [storage],
        'Rating': [rating]
    })
    
    # Make predictions with all models
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = round(pred, 2)
    
    # Display predictions
    st.subheader("Price Predictions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Linear Regression", f"₹{predictions['Linear Regression']:,.2f}")
    
    with col2:
        st.metric("Ridge Regression", f"₹{predictions['Ridge Regression']:,.2f}")
    
    with col3:
        st.metric("Random Forest", f"₹{predictions['Random Forest']:,.2f}")
    
    with col4:
        st.metric("Gradient Boosting", f"₹{predictions['Gradient Boosting']:,.2f}")
    
    # Highlight the best model
    st.info(f"Based on R-squared scores, the {best_model_name} model shows the best performance for price prediction.")
    
    # Model limitations
    st.subheader("Model Limitations and Considerations")
    
    st.markdown("""
    **Limitations of the Price Prediction Model:**
    
    1. **Limited Features**: The model only considers a few features (RAM, storage, rating, brand category) and may miss other important factors like processor, camera quality, display type, etc.
    
    2. **Market Dynamics**: The model doesn't account for temporal market dynamics, such as new product launches, sales events, or changes in consumer preferences.
    
    3. **Brand Specificity**: Grouping smaller brands as 'Other' may lead to less accurate predictions for those brands.
    
    4. **Outliers**: Extremely high-end or low-end phones might not be predicted accurately due to their unique positioning in the market.
    
    5. **External Factors**: The model doesn't consider external factors like supply chain issues, component shortages, or currency fluctuations that might affect pricing.
    
    For more accurate predictions, the model would need to be regularly updated with new data and potentially expanded to include more features.
    """)
