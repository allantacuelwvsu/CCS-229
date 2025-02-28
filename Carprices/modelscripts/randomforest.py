import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Extract / clean anomalies
df = pd.read_csv('carprices.csv')
df = df[df['price'] <= 100000]
df = df[df['price'] >= 1000]

# Preprocess
columns = ['year', 'manufacturer', 'odometer', 'fuel', 'transmission', 'state', 'price']
df = df[columns]
df = df[df['price'] > 0].dropna(subset=['price'])
df['odometer'] = df['odometer'].fillna(0)
df[['manufacturer', 'fuel', 'transmission', 'state']] = df[['manufacturer', 'fuel', 'transmission', 'state']].fillna('Unknown')
# 'Tokenize' / Turn string data into machine readable numerical values by mapping
mapping = {
    "manufacturer": {m: i for i, m in enumerate(df['manufacturer'].unique())},
    "fuel": {f: i for i, f in enumerate(df['fuel'].unique())},
    "transmission": {t: i for i, t in enumerate(df['transmission'].unique())},
    "state": {s: i for i, s in enumerate(df['state'].unique())},
}
for col, map_dict in mapping.items():
    df[col] = df[col].map(map_dict)
df = df.dropna()

# Load preprocessed data
X = df.drop(columns=['price'])  # Features (everything except price)
y = df['price']  # Target variable (price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/randomforest.pkl')