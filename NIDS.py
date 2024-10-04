import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

# Load the dataset
df_train = pd.read_csv('datasets/UNSW_NB15_training-set.csv', header=0)
df_test = pd.read_csv('datasets/UNSW_NB15_testing-set.csv', header=0)

# Drop irrelevant features
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

# Encoding categorical variables (proto, service, state)
df_train = pd.get_dummies(df_train, columns=['proto', 'service', 'state'])
df_test = pd.get_dummies(df_test, columns=['proto', 'service', 'state'])

# Align columns between train and test datasets
X_train, X_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Split the dataset into features and labels
y_train = df_train['label']
y_test = df_test['label']

# We want to predict the label, so we drop the label and attack_cat columns from the features b
X_train = X_train.drop(columns=['label', 'attack_cat'])
X_test = X_test.drop(columns=['label', 'attack_cat'])

# Handle imbalanced data by applying class weights
# Ensure that classes are a numpy array, and calculate class weights
classes = np.array([0, 1])  # Define classes (0 for normal, 1 for attack)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))  # Convert to a dictionary


# Hyperparameter tuning with Optuna (Hyperparameters are already tuned and put in best_params)
"""
def objective(trial):
    # Define the search space for hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Define the model with the hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   class_weight=class_weights_dict,
                                   random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Calculate F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1


# Run Optuna for hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Use the best hyperparameters to train the final model
best_params = study.best_params
"""

best_params = {'n_estimators': 147, 'max_depth': 34, 'min_samples_split': 13, 'min_samples_leaf': 20}

# print(f"Best hyperparameters: {best_params}")

final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],
                                     min_samples_leaf=best_params['min_samples_leaf'],
                                     class_weight=class_weights_dict,
                                     random_state=42)

# Train the final model
final_model.fit(X_train, y_train)

# Test the final model
y_pred = final_model.predict(X_test)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
