import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import os

RANDOM_STATE = 42

train = pd.read_csv("Datasets/Rachycentron/Train.csv")
test = pd.read_csv("Datasets/Rachycentron/Test.csv")

X_train = train.drop(columns=["label3"])
y_train = train["label3"]

X_test = test.drop(columns=["label3"])
y_test = test["label3"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=1000,
        early_stopping=True
    ))
])

param_grid = {

    "model__hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (256,),
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32)
    ],

    "model__activation": [
        "relu",
	    "leaky_relu"
    ],

    "model__solver": [
        "adam"
    ],

    "model__alpha": [
        1e-5,
        1e-4,
        1e-3
    ],

    "model__learning_rate_init": [
        1e-4,
        5e-4,
        1e-3,
        1e-2
    ],

    "model__batch_size": [
        32,
        64,
        128
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=2
)

grid.fit(X_train, y_train)

print("=" * 100)
print("BEST PARAMETERS")
print("=" * 100)
print(grid.best_params_)

print("\nBest CV Macro F1")
print(f"{grid.best_score_:.4f}")

best_model = grid.best_estimator_

test_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)
recall = recall_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 100)
print("TEST RESULT")
print("=" * 100)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report")
print(classification_report(
    y_test,
    test_pred,
    digits=4,
    zero_division=0
))

result = pd.DataFrame([{

    "Model": "MLP",

    "Best Parameters": str(grid.best_params_),

    "CV Macro F1": grid.best_score_,

    "Test Accuracy": accuracy,

    "Test Precision": precision,

    "Test Recall": recall,

    "Test Macro F1": macro_f1

}])


cv_result = pd.DataFrame(grid.cv_results_)

file_path = "mlp_result_rachycentron.xlsx"
if os.path.exists(file_path):
    with pd.ExcelWriter(file_path, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
        result.to_excel(writer, sheet_name="Summary", index=False)
        cv_result.to_excel(writer, sheet_name="GridSearchDetails", index=False)
else:
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Summary", index=False)
        cv_result.to_excel(writer, sheet_name="GridSearchDetails", index=False)

print("\nResults saved.")
