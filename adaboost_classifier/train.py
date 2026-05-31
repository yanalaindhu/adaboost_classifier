import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/insurance.csv")

# Example:
# age, gender, bmi, children, smoker, region, insurance

df = pd.get_dummies(df, drop_first=True)

# Target Column
X = df.drop("insurance", axis=1)
y = df["insurance"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
base_model = DecisionTreeClassifier(max_depth=1)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save Model
with open("models/adaboost_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully")