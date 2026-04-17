import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
d = pd.read_csv('/Users/pushkarkhanna/Downloads/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

"""# Exploratory Data Analysis (EDA)
print("First 5 rows of the dataset:")
print(d.head())
print("Columns in the dataset:")
print(d.columns)
print("Shape of the dataset:")
print(d.shape)
print("Information about the dataset:")
print(d.info())
print("Description of the dataset:")
print(d.describe())
print("Missing values in the dataset:")
print(d.isnull().sum())"""

# Data Cleaning
d.drop(["transaction_id","user_id"],axis = 1, inplace=True)

"""print("Value counts for addiction_level:")
print(d["addiction_level"].value_counts(), "\n")
print("Unique values in addiction_level:")
print(d["addiction_level"].unique(), "\n")
print("Value counts for addicted_label:")
print(d["addicted_label"].value_counts(), "\n")"""

# Impute missing values in addiction_level based on addicted_label
d["addiction_level"] = np.where(d["addiction_level"].isnull(),np.where(d["addicted_label"] == 0, "Mild", "Moderate"), d["addiction_level"])

# Drop the addiction_level column as it's now redundant
d.drop("addiction_level", axis = 1, inplace=True)

# Encoding Categorical Variables
# Dictionary to store encoders for each column
encoders = {}
categorical_cols = ["gender", "stress_level", "academic_work_impact"]

print("--- Category Mappings ---")
for col in categorical_cols:
    if col in d.columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))
        encoders[col] = le 
        
        # This part shows you exactly what 0, 1, and 2 mean
        print(f"\n{col.upper()}:")
        for index, label in enumerate(le.classes_):
            print(f"  {index} = {label}")
#exit() # Run this once to see the mappings, then comment it out to proceed with modeling4

# Prepare data for modeling
x = d.drop("addicted_label", axis = 1)
y = d["addicted_label"]
feature_list = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""# Check data types of features
print("Data Types of Features:")
print(x.dtypes)"""
# exit() 

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
"""plt.figure(figsize=(20,10))
plot_tree(rf_model.estimators_[0], 
          max_depth=3, 
          filled=True, 
          feature_names=x.columns, 
          class_names=["Not Addicted", "Addicted"])
plt.show()"""

# Evaluate the model
print("--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

"""# Run this to get a clean summary of what the tree just showed you
importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Factors Leading to Addiction")
plt.show()"""

# 8. Save all components for the "Natural" Prediction Script
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(feature_list, 'feature_names.pkl')

print("\nAll components successfully saved to your folder.")
print("="*30) # This line is just a separator for better readability in the output