# 🕵️‍♂️ Financial Fraud Detection with XGBoost

## 1. Introduction

This project documents my learning journey in Python and its application to a real-world business problem: **identifying financial fraud using XGBoost**.

**Why XGBoost?**  
[XGBoost](https://xgboost.readthedocs.io/en/stable/) is a top-performing, scalable machine learning library that is widely used across industries—including finance, marketing, and healthcare. Mastery of XGBoost unlocks opportunities for robust predictive modeling in diverse domains.

**Dataset:**  
[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)  
- **179,014** transaction records
- **11 features:** transaction type, amount, account balances, and a binary fraud indicator (`isFraud`)
- **Class imbalance:** Only **0.0776%** (139 out of 179,014) are fraudulent—a challenging but common reality in fraud detection.

## 2. Initial Approach

A simple baseline XGBoost model to establish the project foundation.

### 2.1 Feature Engineering
- **Basic features:** Raw transaction fields (amount, balances)
- **Simple transformations:** Example—`origBalanceDelta = newbalanceOrig - oldbalanceOrg`

### 2.2 Data Preprocessing
- Fill missing values (zeros or medians)
- Scale with `StandardScaler`
- Encode categoricals (`OneHotEncoder` for `type`)

### 2.3 Model Training
- **Algorithm:** XGBoost classifier (default hyperparameters)
- **Handling imbalance:** Relied on XGBoost’s built-in weighting (no explicit resampling)

### 2.4 Evaluation Metrics
- **Metrics:** Accuracy, precision, recall, F1-score
- **Result:** OK accuracy, but **low recall for fraud**—the model missed many rare cases due to class imbalance.

---

## 3. Enhanced Approach

Driven by the initial model’s low fraud recall, a more advanced version was created using better features, preprocessing, and balance correction.

### 3.1 Limitations of Initial Approach

- Low recall (misses many frauds)
- Limited features
- Basic preprocessing insufficient for skewed, imbalanced data

### 3.2 Advanced Feature Engineering

- **Transaction Context:** Extract origin/destination types from account IDs  
- **Ratios:** E.g., `origAmountToBalanceRatio`
- **Temporal Features:** Hour, day, weekday from `step`
- **Behavioral Flags:** `isAccountEmptied`, `isNewAccount`, `isLargeTransaction`, `isToMerchant`
- **Velocity Features:** Number and amount of transactions in past 24h
- **Log Transforms:** To reduce skew

### 3.3 Improved Preprocessing

- **Pipeline:**  
    - Numerical: `SimpleImputer` (median), `PowerTransformer` (Yeo-Johnson), `StandardScaler`
    - Categorical: `SimpleImputer` (mode), `OneHotEncoder`

### 3.4 Resampling Techniques

Tested: **SMOTE**, **ADASYN**, **SMOTEENN**, and **SMOTETomek**  
- **Best:** SMOTE (F1=0.6972)  
- Others: ADASYN (0.6902), SMOTEENN (0.6838), SMOTETomek (0.6945)

### 3.5 Model Training and Tuning

- **XGBoost enhancements:** Hyperparameter tuning using `RandomizedSearchCV`
- **Feature selection:** Keep features contributing >90% importance

### 3.6 Evaluation Metrics

- Added **ROC-AUC**, precision-recall curve, and average precision
- Adjusted thresholds for F1 and business cost impact

---

## 4. Results and Discussion

### 4.1 Initial Model

- **High global accuracy (99%)**—but misleading due to severe class imbalance  
- Fraud class: Only 4% precision, 8% F1—**lots of false alarms**  
- Non-fraud class: 99% F1  
- **Business problem:** Too many labor-intensive false positives. Employees would review 100 flagged transactions to find only 4 actual frauds.

### 4.2 Enhanced Model

- **Perfect accuracy on non-fraud (100%)**
- Fraud class: 96% precision, 95% F1—**drastic improvement**
- Only 4 in 100 fraudulent transactions mislabeled
- Overall, the model is now far more useful for practical fraud detection

### 4.3 Key Takeaways

- **Advanced features** like `isToMerchant` and `origAmountToBalanceRatio` are top predictors
- **Resampling** fixed the imbalance, boosting fraud detection
- **Hyperparameter tuning** improved model precision without sacrificing recall

---

## 5. Conclusion

This project demonstrates the transformative impact of iterative feature engineering, preprocessing, and model tuning in financial fraud detection. The enhanced XGBoost pipeline delivers **practical and business-ready performance**, sharply reducing false alarms and increasing true positive fraud detection.

**Future Directions:**  
- Try ensemble or deep learning models  
- Detect new types of fraud as patterns evolve  
- Explore model explainability for regulatory and operational transparency

---

## 🚀 Get Started

1. **Clone the repo & install requirements**
    ```bash
    pip install xgboost pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
    ```
2. **Download the dataset:**  
   [Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)

3. **Run the notebook or script** for step-by-step code and outputs.

---

## 📚 References

- [Kaggle: Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [SMOTE for Imbalanced Learning](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## 🤝 Contributions

Have improvements or questions?  
Open an issue or submit a pull request!

---

_This project was inspired by a desire to advance practical Python skills and apply them to high-stakes business problems using state-of-the-art machine learning techniques._
