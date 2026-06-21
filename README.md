🚗 Used Car Price Prediction — Regression Pipeline

This project explores the 2025 used car market by cleaning, analyzing, and modeling real-world vehicle listing data using Python. It includes a fully modular 9-stage pipeline that predicts car prices using Linear Regression, Decision Trees, and Random Forest, with model comparison and visual reporting built in.

📊 Project Description

This project analyzes used car listing data sourced from [Kaggle's 2025 Used Car Market Dataset](https://www.kaggle.com/datasets), focusing on:

* Vehicle pricing trends across brands and model years,
* The effect of mileage and title status on resale value,
* Comparing regression algorithms to find the most accurate predictor.

Key tasks include:

1. Loading and inspecting raw listing data,
2. Performing exploratory data analysis (EDA),
3. Cleaning and preprocessing the dataset (outlier removal, encoding, scaling),
4. Training and evaluating three regression models,
5. Comparing model performance and visualizing the results.

🧠 Features

📌 Data Cleaning

* Removes invalid listings (price = 0, pre-2000 vehicles),
* Detects and removes extreme mileage outliers using the IQR method,
* Drops non-predictive columns (VIN, lot number, country, auction timer),
* Groups rare car models into an "other" category to reduce noise.

📌 Exploratory Data Analysis

* Distribution plots for price, year, and mileage,
* Correlation heatmap of numeric features,
* Average price comparisons by brand and title status,
* Boxplots showing the price impact of salvage vs. clean titles.

📌 Predictive Modeling

* Linear Regression as a baseline model,
* Decision Tree Regressor with depth control to reduce overfitting,
* Random Forest Regressor (ensemble of 100 trees),
* Evaluation using MSE and R² for every model,
* Feature importance ranking to identify what drives used car prices.

📁 Project Structure

* `used_car_regression.py`: Complete 9-stage pipeline (loading → EDA → preprocessing → modeling → reporting),
* `data/`: Contains the raw dataset and a saved raw copy,
* `outputs/`: Contains the cleaned dataset, model comparison table, and all generated plots,
* `outputs/plots/`: All visualizations produced by the pipeline,
* `README.md`: Project documentation (this file).

🚀 Getting Started

✅ Prerequisites

* Python 3.8 or higher
* Recommended: Cursor, VS Code, or Jupyter Notebook

🛠️ Installation Steps

```
# 1. Clone the repository
git clone https://github.com/DataCrafter20/used-car-price-prediction.git

# 2. Navigate to the project directory
cd used-car-price-prediction

# 3. Install the dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 4. Run the pipeline
python used_car_regression.py
```

The script will automatically create the required folders, clean the data, train all three models, and save every plot and result to the `outputs/` directory.

📊 Sample Results

| Model | MSE | R² (%) |
|---|---|---|
| Linear Regression | 95,299,014.57 | 31.41% |
| Decision Tree | 109,493,748.91 | 21.20% |
| **Random Forest** | **76,848,160.53** | **44.69%** |

Random Forest was the best-performing model, explaining 44.7% of the variance in used car prices. This indicates that while mileage, year, and title status are meaningful predictors, other unrecorded factors (engine condition, accident history, number of previous owners) likely account for the remaining price variation.

🧩 Future Enhancements

* Add hyperparameter tuning (GridSearchCV) to improve Random Forest performance,
* Try gradient boosting models (XGBoost, LightGBM) for comparison,
* Incorporate additional features like engine size or accident history if available,
* Build a Streamlit dashboard for interactive price prediction.

👤 Author

NDIVHUWO MUNYAI

* 📧 nmunyai11@gmail.com
* 🔗 https://github.com/DataCrafter20
* 🔗 https://linkedin.com/in/ndivhuwo-munyai-390a58337

📄 License

This project is licensed under the MIT License.
