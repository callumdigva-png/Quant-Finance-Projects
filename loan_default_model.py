import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load the data
df = pd.read_csv(r"C:\Users\callu\Downloads\Loan_Data\Task_3_and_4_Loan_Data.csv")

# 2. Drop the customer_id column
df = df.drop(columns=["customer_id"])

# 3. Split into features (X) and target (y)
X = df.drop(columns=["default"])
y = df["default"]

# 4. Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 8. Expected loss function
def expected_loss(customer_details, loan_amount):
    customer_df = pd.DataFrame([customer_details])
    customer_scaled = scaler.transform(customer_df)
    pd_prob = model.predict_proba(customer_scaled)[0][1]
    loss = pd_prob * loan_amount * 0.9
    print(f"Probability of default: {pd_prob:.2%}")
    return loss

# 9. Test it
test_customer = {
    "credit_lines_outstanding": 2,
    "loan_amt_outstanding": 10000,
    "total_debt_outstanding": 20000,
    "income": 25000,
    "years_employed": 3,
    "fico_score": 600
}

print("running expected loss...")
loss = expected_loss(test_customer, 5000)
print(f"Expected loss: £{loss:.2f}")

print(df['fico_score'].min())
print(df['fico_score'].max())
print(df['fico_score'].describe())