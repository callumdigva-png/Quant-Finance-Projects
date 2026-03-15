import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#import data from provided datasheet on kaggle
df = pd.read_csv(r"C:\Users\callu\OneDrive\Documents\Coding Projects\Credit_loans\UCI_Credit_Card.csv.zip")

#get rid of columns that are unimportant to the function
df = df.drop(columns=["SEX", "MARRIAGE", "EDUCATION"])

#create a payment to bill ratio, as an additional indicater 
df["avg_bill"] = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
df["avg_payment"] = df[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(axis=1)

df["payment_to_bill_ratio"] = df["avg_payment"]/df["avg_bill"].replace(0,1)

#define all features that matter to the function
features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'avg_bill', 'avg_payment', 'payment_to_bill_ratio']

#spllit into X and Y
X = df[features]
Y = df["default.payment.next.month"]

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size= 0.2, random_state= 42)

#scale the data, so they can be properly interpreted
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#define the functiom, note this can also be done with logisticregression(scale = balenced) and randomforestclassifier, as shown by the library imports
model = XGBClassifier(scale_pos_weight = 3.52, random_state= 42)
#train the model with the training data provided
model.fit(X_train_scaled, Y_train)

#use the model to predict the outcomes of a new customer
y_pred = model.predict(X_test_scaled)
print(classification_report(Y_test, y_pred))

#confusion matrix
cm = confusion_matrix(Y_test,y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

#expected loss function
def expected_loss(customer_details, loan_amount):
    customer_df = pd.DataFrame([customer_details])
    customer_scaled = scaler.transform(customer_df)
    pd_prob = model.predict_proba(customer_scaled)[0][1]
    loss = pd_prob * loan_amount * 0.8
    print(f"The Probability of default: {pd_prob:.2%}")
    return loss

#risky customer
test_customer = {
    "LIMIT_BAL": 20000,
    "AGE": 22,
    "PAY_0": 0,
    "PAY_2": 2,
    "PAY_3": 1,
    "PAY_4": 1,
    "PAY_5": 2,
    "PAY_6": 2,
    "avg_bill": 15000,
    "avg_payment": 3000,
    "payment_to_bill_ratio": 0.2
}

print("running expected loss...")
loss = expected_loss(test_customer, 3000)
print(f"Expected loss: £{loss:.2f}")

