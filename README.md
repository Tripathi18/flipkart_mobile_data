# 📱 Flipkart Mobile Market Analysis & Price Prediction

This project analyzes the Indian mobile phone market using real-world data scraped from Flipkart. It includes detailed exploratory data analysis (EDA), brand comparisons, and a machine learning model to predict mobile prices based on product features.

An interactive dashboard is also available via **Streamlit**, allowing users to explore the dataset visually and make predictions.

---

## 📊 Features

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data cleaning and preprocessing
- 📈 Brand-wise analysis and visualization
- 🧠 Price prediction using ML models (Random Forest, Gradient Boosting, etc.)
- 🖥️ Streamlit app interface for real-time interaction

---

## 📁 Project Structure

```

.
├── Flipkart\_Mobiles.csv                          # Primary dataset
├── Flipkart\_mobile\_brands\_scraped\_data.csv      # Supplementary brand dataset
├── data\_processor.py                            # Data cleaning and preprocessing
├── eda.py                                       # Exploratory Data Analysis script
├── brand\_analysis.py                            # Brand-wise insights
├── model.py                                     # Machine Learning modeling
├── app.py                                       # Streamlit application
├── pyproject.toml                               # Project dependencies/config
└── README.md                                    # Project documentation

````

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/flipkart_mobile_data.git
cd flipkart_mobile_data
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*Alternatively, use `pyproject.toml` with Poetry.*

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 📈 Model Overview

The model pipeline includes:

* Feature selection using `SelectKBest`
* Preprocessing with `ColumnTransformer`
* Algorithms: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
* Evaluation metrics: RMSE, MAE, R²

---

## 📌 Key Insights

* Brands like Samsung, Realme, and Xiaomi dominate Flipkart listings.
* Selling prices show significant variation from original prices.
* RAM, Storage, and Battery are strong predictors of price.

---

## 🧠 Future Improvements

* Integrate customer review sentiment
* Add time-series price tracking
* Deploy model as API
* Improve UI/UX of dashboard

---

## 📚 References

* Flipkart.com (Data Source)
* Scikit-learn
* Streamlit
* Pandas, NumPy, Matplotlib, Seaborn, Plotly

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
