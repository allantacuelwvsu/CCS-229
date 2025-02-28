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

# Preset Character Data
def set_character(pclass, sex, age, sibsp, parch, fare, embarked):
    st.session_state.pclass = pclass
    st.session_state.sex = sex
    st.session_state.age = age
    st.session_state.sibsp = sibsp
    st.session_state.parch = parch
    st.session_state.fare = fare
    st.session_state.embarked = embarked
    
with st.expander("Movie Characters"):
    col1, col2 = st.columns(2)
    with col1:
        st.button("Rose DeWitt Bukater", on_click=set_character, args=(1, "Female", 17, 0, 1, 300.0, "S")) # Rose was 17, embarked from South Hampton, was in first class and paid $300 as an estimate of the average of the 1st class fare, maybe a slightly less luxurious cabin. . She was with her mother.
        st.button("Jack Dawson", on_click=set_character, args=(3, "Male", 20, 0, 0, 8.05, "S")) # Jack was 20, embarked from South Hampton, was in third class and paid a random estimate of $8.05 as the bet for the poker game where he won is 3rd class ticket. Of course, he was alone.
    with col2:
        st.button("Caledon Hockley", on_click=set_character, args=(1, "Male", 30, 0, 0, 500.0, "C")) # Caledon was 30, embarked from Cherbourg, was in first class and paid $500 as an estimate of the high average of the 1st class fare. He was alone.
        st.button("Ruth DeWitt Bukater", on_click=set_character, args=(1, "Female", 45, 0, 1, 300.0, "S")) # Ruth was 45, embarked from South Hampton, was in first class and paid $300 as an estimate of the average of the 1st class fare. She was with her daughter.

pclass = st.selectbox("Passenger Class", [1, 2, 3], index=[1, 2, 3].index(st.session_state.get("pclass", 3)))
sex = st.radio("Sex", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.get("sex", "Male")))
age = st.slider("Age", 0, 100, st.session_state.get("age", 30))
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, st.session_state.get("sibsp", 0))
parch = st.slider("Parents/Children Aboard", 0, 6, st.session_state.get("parch", 0))
fare = st.number_input("Fare", 0.0, 500.0, st.session_state.get("fare", 30.0))
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], index=["C", "Q", "S"].index(st.session_state.get("embarked", "S")))

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