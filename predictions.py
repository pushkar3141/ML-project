import joblib
import pandas as pd
import numpy as np  

# Now you can run the "Natural" Prediction Script to interactively analyze smartphone addiction risk based on user input!

# Load your pre-trained components

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

print("--- Smartphone Addiction Risk Analyzer ---")
print("Please enter the following details:")

# Collect Inputs from Terminal
raw_input = {}
raw_input['age'] = float(input("Enter Age: "))
raw_input['gender'] = input("Enter Gender (Male/Female/Other): ")
raw_input['daily_screen_time_hours'] = float(input("Daily Screen Time (hours): "))
raw_input['social_media_hours'] = float(input("Social Media Usage (hours): "))
raw_input['gaming_hours'] = float(input("Gaming Usage (hours): "))
raw_input['work_study_hours'] = float(input("Work/Study Usage (hours): "))
raw_input['sleep_hours'] = float(input("Sleep Duration (hours): "))
raw_input['notifications_per_day'] = float(input("Notifications per Day: "))
raw_input['app_opens_per_day'] = float(input("App Opens per Day: "))
raw_input['weekend_screen_time'] = float(input("Weekend Screen Time (hours): "))
raw_input['stress_level'] = input("Stress Level (Low/Medium/High): ")
raw_input['academic_work_impact'] = input("Does it impact your work/studies? (Yes/No): ")


df = pd.DataFrame([raw_input])

# Apply the saved LabelEncoders
for col, le in encoders.items():
    # .strip() handles accidental spaces in user input
    df[col] = le.transform([str(raw_input[col]).strip().capitalize()])

# Match the training feature order
df = df[feature_names]

# Scale the numeric values
df_scaled = scaler.transform(df)

# Make prediction and calculate confidence
prediction = model.predict(df_scaled)
probability = model.predict_proba(df_scaled)

# Display results with clear explanations
status = "You are at high risk of smartphone addiction." if prediction[0] == 1 else "You are not at high risk of smartphone addiction."
confidence = probability[0][prediction[0]] * 100

# This part provides a clear and user-friendly output, showing the risk status and confidence level, along with specific risk factors if applicable.

print("\n" + "="*30)
print(f"RESULT: {status}")
print(f"CONFIDENCE: {confidence:.2f}%")


if prediction[0] == 1:
    print("\nPrimary Risk Factors Identified:")
    
    # We check the values inside our raw_input dictionary
    if raw_input['social_media_hours'] > 4:
        print(f"  🚩 High Social Media: {raw_input['social_media_hours']} hrs/day (Threshold: 4hrs)")
        
    if raw_input['sleep_hours'] < 6:
        print(f"  🚩 Sleep Deprivation: {raw_input['sleep_hours']} hrs (Minimum suggested: 6hrs)")
        
    if raw_input['weekend_screen_time'] > 8:
        print(f"  🚩 Weekend Overuse: {raw_input['weekend_screen_time']} hrs (High recreational usage)")
        
    if raw_input['notifications_per_day'] > 150:
        print(f"  🚩 Notification Overload: {raw_input['notifications_per_day']} alerts/day")

else:
    print("\nPositive Habits Observed:")
    if raw_input['sleep_hours'] >= 7:
        print("  ✅ Healthy sleep schedule maintained.")
    if raw_input['social_media_hours'] < 2:
        print("  ✅ Controlled social media usage.")

if confidence > 80:
    risk = "High Risk 🚨"
elif confidence > 50:
    risk = "Moderate Risk ⚠️"
else:
    risk = "Low Risk ✅"

print(f"Risk Level: {risk}")


print("\nSuggested Actions:")

if raw_input['sleep_hours'] < 6:
    print("  👉 Increase sleep to at least 6-7 hours")

if raw_input['social_media_hours'] > 4:
    print("  👉 Reduce social media usage")

if raw_input['notifications_per_day'] > 150:
    print("  👉 Turn off non-essential notifications")

if raw_input['weekend_screen_time'] > 8:
    print("  👉 Limit weekend screen time")


print("="*30)
