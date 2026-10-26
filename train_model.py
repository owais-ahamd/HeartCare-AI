import pandas as pd
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


# Load Dataset

data = pd.read_csv(
    "data/heart.csv"
)


print(data.head())


# Separate input and output

X = data.drop("target", axis=1)

y = data["target"]


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Models

models = {

"Logistic Regression":
LogisticRegression(),

"Decision Tree":
DecisionTreeClassifier(),

"Random Forest":
RandomForestClassifier(),

"KNN":
KNeighborsClassifier()

}



best_model = None
best_accuracy = 0



for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    print(
        name,
        accuracy
    )


    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model



# Save model

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/heart_model.pkl"
)


joblib.dump(
    scaler,
    "models/scaler.pkl"
)



print(
    "Best Model Accuracy:",
    best_accuracy
)

print(
    "Model Saved"
)