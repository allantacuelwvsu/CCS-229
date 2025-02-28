import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load
train_df = pd.read_csv("Titanic-Survival/datasets/train.csv")

# Preprocess
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
train_df = pd.get_dummies(train_df, columns=["Embarked"], drop_first=True)
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)

# Train
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# App
st.title("Titanic Survival Prediction")

# Model Assessment
st.header("Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_val, y_pred):.2f}")
st.write(f"**Precision:** {precision_score(y_val, y_pred):.2f}")
st.write(f"**Recall:** {recall_score(y_val, y_pred):.2f}")
st.write(f"**F1 Score:** {f1_score(y_val, y_pred):.2f}")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Predictor
st.header("Predict Survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Process Input
sex = 1 if sex == "Female" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
family_size = sibsp + parch + 1
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, family_size, embarked_C, embarked_Q]])

# Predict and Output
survival_probability = model.predict_proba(input_data)[0][1]
st.subheader("Survival Probability")
st.write(f"{survival_probability:.2%}")