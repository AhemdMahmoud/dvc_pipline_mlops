## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from imblearn.over_sampling import SMOTE
from PIL import Image

## skelarn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## skelarn -- models
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

## sklearn -- metrics
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score





## --------------------- Data Preparation ---------------------------- ##
data_path = os.path.join(os.getcwd(), "outs_data")

X_train_final = pd.read_csv(os.path.join(data_path, "X_train_final.csv"))
X_test_final = pd.read_csv(os.path.join(data_path, "X_test_final.csv")) 
y_train = pd.read_csv(os.path.join(data_path, "y_train_final.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test_final.csv"))

# --------------------- Impalancing ---------------------------- ##

## 1. use algorithm without taking the effect of imbalancing

y_train = y_train.iloc[:, 0]  ## convert to 1D array
y_test = y_test.iloc[:, 0]  ## convert to 1D array
## 2. prepare class_weights for solving imbalance dataset
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  ## normalizing


dict_weights = {}
for i in range(2):  ## 2 classes (0, 1)
    dict_weights[i] = vals_count[i]

## 3. Using SMOTE for over sampling
over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)


## --------------------- Modeling ---------------------------- ##

## Clear metrics.json file at the beginning
# with open('metrics.json', 'w') as f:
#     pass

all_metrics = {}

import json

# Global flat dictionary to hold all metrics
all_metrics = {}

def train_model(X_train, y_train, plot_name='', class_weight=None):
    """ A function to train model given the required train data """

    global clf_name, all_metrics

    clf = RandomForestClassifier(n_estimators=800, max_depth=14, random_state=45, class_weight=class_weight)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test_final)

    f1_test = f1_score(y_test, y_pred_test)
    acc_test = accuracy_score(y_test, y_pred_test)

    clf_name = clf.__class__.__name__

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Save to flat dictionary
    all_metrics[f"{plot_name}_f1_score"] = round(f1_test, 4)
    all_metrics[f"{plot_name}_accuracy"] = round(acc_test, 4)

    print(f"Updated metrics: {plot_name}_f1_score = {round(f1_test, 4)}, {plot_name}_accuracy = {round(acc_test, 4)}")  # Debugging output

    return True

# Call this after training all models
def save_all_metrics():
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to 'metrics.json': {all_metrics}")  # Debugging output





## 1. without considering the imabalancing data
train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance', class_weight=None)

## 2. with considering the imabalancing data using class_weights
train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', class_weight=dict_weights)

## 3. with considering the imabalancing data using oversampled data (SMOTE)
train_model(X_train=X_train_resmapled, y_train=y_train_resampled, plot_name=f'with-SMOTE', class_weight=None)

save_all_metrics()

## Combine all conf matrix in one
confusion_matrix_paths = [f'./without-imbalance.png', f'./with-class-weights.png', f'./with-SMOTE.png']

## Load and plot each confusion matrix
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization


## Save combined plot locally
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

## Delete old image files
for path in confusion_matrix_paths:
    os.remove(path)
