import pandas as pd
import numpy as np
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

RULES = {
    "Air temperature": {
        "HS": [(25, 29, True, True)],
        "S": [(20, 25, True, False), (29, 35, False, True)],
    },
    "Rainfall": {
        "HS": [(1900, 2600, True, True)],
        "S": [(1500, 1900, True, False), (2600, 3000, False, True)],
    },
    "Salinity": {
        "HS": [(13, 21, True, True)],
        "S": [(8, 13, True, False), (21, 25, False, True)],
    },
    "Alkalinity": {
        "HS": [(80, 160, True, True)],
        "S": [(40, 80, True, False), (160, 200, False, True)],
    },
    "pH": {
        "HS": [(6.2, 7.7, True, True)],
        "S": [(5.5, 6.2, True, False), (7.7, 8.5, False, True)],
    },
    "NH3": {
        "HS": [(0.28, 0.42, True, True)],
        "S": [(0.2, 0.28, True, False), (0.42, 0.5, False, True)],
    },
    "H2S": {
        "HS": [(0.0, 0.05, True, True)],
        "S": [(0.05, 0.1, False, True)],
    },
    "Sea water temperature": {
        "HS": [(22, 28, True, True)],
        "S": [(17, 22, True, False), (28, 32, False, True)],
    },
    "BOD5": {
        "HS": [(0, 25, True, True)],
        "S": [(25, 50, False, True)],
    },
    "COD": {
        "HS": [(0, 75, True, True)],
        "S": [(75, 150, False, True)],
    },
    "TSS": {
        "HS": [(0, 25, True, True)],
        "S": [(25, 50, False, True)],
    },
    "As": {
        "HS": [(0, 12, True, True)],
        "S": [(12, 15, False, True)],
    },
    "Cd": {
        "HS": [(0, 2, True, True)],
        "S": [(2, 5, False, True)],
    },
    "Pb": {
        "HS": [(0, 100, True, True)],
        "S": [(100, 150, False, True)],
    },
    "Cu": {
        "HS": [(0, 70, True, True)],
        "S": [(70, 100, False, True)],
    },
    "Zn": {
        "HS": [(0, 200, True, True)],
        "S": [(200, 250, False, True)],
    },
}

RANKS = {
    "Air temperature": 5,
    "Rainfall": 6,
    "Salinity": 7,
    "Alkalinity": 8,
    "pH": 1,
    "NH3": 3,
    "H2S": 4,
    "Sea water temperature": 2,
    "BOD5": 9,
    "COD": 10,
    "TSS": 11,
    "As": 14,
    "Cd": 12,
    "Pb": 13,
    "Cu": 15,
    "Zn": 16
}


def calculate_weights(ranks_dict):
    inverse_ranks = {k: 1.0 / v for k, v in ranks_dict.items()}
    total_inv = sum(inverse_ranks.values())
    return {k: v / total_inv for k, v in inverse_ranks.items()}

WEIGHTS = calculate_weights(RANKS)


def get_interval_info(x, intervals):
    if not intervals:
        return False, np.inf, 1.0, 0, 1.0

    is_in = False
    min_dist = np.inf
    closest_width = 1.0
    center = 0
    radius = 1.0

    for lo, hi, lc, rc in intervals:
        left_cond = x >= lo if lc else x > lo
        right_cond = x <= hi if rc else x < hi

        if left_cond and right_cond:
            is_in = True
            min_dist = 0.0
            closest_width = hi - lo
            center = (hi + lo) / 2.0
            radius = (hi - lo) / 2.0
            break

        dist = lo - x if x < lo else (x - hi if x > hi else 0)

        if dist < min_dist:
            min_dist = dist
            closest_width = hi - lo
            center = (hi + lo) / 2.0
            radius = (hi - lo) / 2.0

    if closest_width <= 0: closest_width = 1e-6
    if radius <= 0: radius = 1e-6

    return is_in, min_dist, closest_width, center, radius


def calculate_suitability_score(x, hs_intervals, s_intervals, gamma_us=1.0):
    gamma_s = -np.log(0.675 / 0.875)

    in_hs, dist_to_hs, _, center_hs, radius_hs = get_interval_info(x, hs_intervals)

    if in_hs:
        score = 1.0 - 0.125 * (abs(x - center_hs) / radius_hs)
        return score

    in_s, dist_to_s, width_s, _, _ = get_interval_info(x, s_intervals)

    if in_s:
        score = 0.875 * np.exp(-gamma_s * (dist_to_hs / width_s))
        return score

    if not s_intervals:
        dist_to_boundary = dist_to_hs
        width_scaling = radius_hs * 2
    else:
        dist_to_boundary = dist_to_s
        width_scaling = width_s

    score = 0.675 * np.exp(-gamma_us * (dist_to_boundary / width_scaling))
    return score


def enrich_features_with_expert_knowledge(df, rules, weights, gamma_us=1.0):
    df_enhanced = df.copy()

    for col, rule in rules.items():
        if col in df.columns:
            hs_intervals = rule.get("hs", [])
            s_intervals = rule.get("s", [])
            w_i = weights.get(col, 0)

            scores = df[col].apply(
                lambda x: calculate_suitability_score(x, hs_intervals, s_intervals, gamma_us)
            )

            penalty_col_name = f"{col}_Risk"
            df_enhanced[penalty_col_name] = w_i * (1.0 - scores)

    return df_enhanced


train = pd.read_csv("Datasets/Sonneratia/Train.csv")
test = pd.read_csv("Datasets/Sonneratia/Test.csv")

X_train_raw = train.drop(columns=["label3"])
y_train = train["label3"]

X_test_raw = test.drop(columns=["label3"])
y_test = test["label3"]


X_train = enrich_features_with_expert_knowledge(X_train_raw, RULES, WEIGHTS, gamma_us=1.0)
X_test = enrich_features_with_expert_knowledge(X_test_raw, RULES, WEIGHTS, gamma_us=1.0)



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

file_path = "enhanced_mlp_result_sonneratia.xlsx"
if os.path.exists(file_path):
    with pd.ExcelWriter(file_path, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
        result.to_excel(writer, sheet_name="Summary", index=False)
        cv_result.to_excel(writer, sheet_name="GridSearchDetails", index=False)
else:
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Summary", index=False)
        cv_result.to_excel(writer, sheet_name="GridSearchDetails", index=False)

print("\nResults saved.")
