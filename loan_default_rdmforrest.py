import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\callu\OneDrive\Documents\Coding Projects\Loan_Data\Task_3_and_4_Loan_Data.csv")

df = df.drop(columns=["customer_id"])

X = df.drop(columns=["default"])
Y = df["default"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_scaled, Y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(Y_test, y_pred))

def expected_loss(customer_details, loan_amount):
    customer_df = pd.DataFrame([customer_details])
    customer_scaled = scaler.transform(customer_df)
    pd_prob = model.predict_proba(customer_scaled)[0][1]
    loss = pd_prob * loan_amount * 0.9
    print(f"Probability of Default {pd_prob:.2%}")
    return loss
