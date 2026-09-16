# Fater Coccole Pampers: Churn Prediction & Real-Time Inference API

An end-to-end Machine Learning pipeline and production-grade REST microservice predicting customer churn for Fater's Coccole Pampers loyalty app. Built for the **LUISS Grand Challenge 2025** in collaboration with **Fater (Angelini Industries & P&G)**.

---

## Overview
* **Business Problem:** 53% of app users were churning (defined as inactivity exceeding 90 days). The goal was to identify at-risk users early to enable data-driven targeted retention campaigns.
* **Dataset Scale:** 6,205 unique users consolidated across 6 relational sources (user profiles, app logins, scanned product codes, missions, and rewards).
* **Architecture:** Offline feature engineering & model benchmarking in Jupyter, packaged into an asynchronous FastAPI inference service containerized with Docker.

---

## Model Performance

Trained using an optimized **XGBoost Classifier** with an 80/20 train/test split.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 91% |
| **ROC-AUC** | 0.970 |
| **5-Fold Cross-Validation ROC-AUC** | 0.971 (+/- 0.004) |

### Customer Segmentation
Based on model churn probability and platform engagement:
* **Loyal (47% / 2,907 users):** Active engagement, low risk of churn.
* **Medium Risk (39% / 2,428 users):** Declining app interactions; eligible for re-engagement promotions.
* **High Risk (14% / 870 users):** Zero recent logins and high churn probability; immediate retention intervention required.

---

## Service Architecture & Features

The inference service accepts 6 core behavioral and demographic features:
1. `ETA_MM_BambinoTODAY` (Baby age in months)
2. `total_points` (Total accumulated loyalty points)
3. `total_codes` (Count of scanned product codes)
4. `total_logins` (Application session count)
5. `total_missions` (Completed engagement challenges)
6. `tenure_days` (Days since registration)

---

## Quickstart & Deployment

### Option 1: Docker (Recommended)

1. **Build the container image:**
```bash
docker build -t fater-churn-service .
