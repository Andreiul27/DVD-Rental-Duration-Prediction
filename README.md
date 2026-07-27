# DVD Rental Duration Prediction

## Overview
Built regression models to help a DVD rental company predict how many days 
a customer will rent a DVD. Accurate predictions support better inventory 
planning and operational efficiency. The target was to achieve an MSE of 3 
or less on the test set.

## Goals / Questions I explored
- Can we accurately predict DVD rental duration from customer and movie data?
- Which features are most relevant for predicting rental length?
- Which regression model performs best: Linear Regression or Random Forest?

## Tools & Libraries
Python, pandas, NumPy, scikit-learn (Lasso, LinearRegression, 
RandomForestRegressor, RandomizedSearchCV)

## Methodology
1. Loaded rental data and parsed rental/return dates to calculate rental length in days
2. Engineered features from special_features column (deleted scenes, behind the scenes)
3. Removed data leakage columns (return date, rental date)
4. Applied **Lasso regression** for feature selection that reduced 14 features down to 4
5. Trained and compared **Linear Regression** and **Random Forest** models
6. Used **RandomizedSearchCV** with 5-fold cross-validation to tune Random Forest

## Key Findings
- Lasso selected 4 key features: `amount`, `amount_2`, `length_2`, `rental_rate_2`
- Linear Regression MSE: **3.09** (did not meet target)
- Random Forest MSE: **2.06** ✅ (beat the target of < 3)
- Best Random Forest parameters: 150 estimators, max depth 14, min samples leaf 2

## Features Used
| Feature | Description |
|---------|-------------|
| amount | Amount paid by the customer |
| rental_rate | Rate at which the DVD is rented |
| length | Movie length in minutes |
| replacement_cost | Cost to replace the DVD |
| special_features | Encoded as deleted scenes / behind the scenes dummies |
| NC-17, PG, PG-13, R | Movie rating dummy variables |

## Dataset
Source: rental_info.csv (provided as part of DataCamp project)
