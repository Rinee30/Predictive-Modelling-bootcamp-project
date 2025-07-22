import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\kanik\Downloads\heart_failure_clinical_records_dataset (1).csv")

# Step 2: Feature Engineering
df['creatinine_per_age'] = df['serum_creatinine'] / df['age']
df['platelet_ratio'] = df['platelets'] / df['serum_creatinine']

# Step 3: Split features and label
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 6: Build voting ensemble model
model1 = LogisticRegression(max_iter=1000)
model2 = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1)
model3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

voting_model = VotingClassifier(estimators=[
    ('lr', model1),
    ('gb', model2),
    ('xgb', model3)
], voting='soft')

# Step 7: Train the model
voting_model.fit(X_resampled, y_resampled)

# Step 8: Evaluate model
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save the model to a known folder
output_path = os.path.join(os.getcwd(), 'model.pkl')
pickle.dump(voting_model, open(output_path, 'wb'))
print(f"\n🎯 Model saved successfully as: {output_path}")