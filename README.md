# fater-churn-prediction

## Overview
Churn prediction model for Fater's Coccole Pampers loyalty app. 
Built for the LUISS Grand Challenge 2025 in collaboration with Fater (Angelini Industries & P&G).

## Problem
53% of app users were churning — stopping engagement with the loyalty program. 
The goal was to predict which users are at risk and support targeted marketing campaigns.

## Dataset
- 6,205 unique users
- Sources: user profiles, app logins, product codes, missions, rewards

## Methodology
- Data integration across 6 tables
- Churn definition: inactive for more than 90 days
- Model: XGBoost Classifier
- Train/Test split: 80/20

## Results
- Accuracy: 91%
- ROC-AUC: 0.97
- Cross-Validation ROC-AUC: 0.971 (+/- 0.004)

## User Segments
- Loyal: 47% (2,907 users)
- Medium Risk: 39% (2,428 users)  
- High Risk: 14% (870 users)

## Files
- `grand_challenge.ipynb` — Python n
