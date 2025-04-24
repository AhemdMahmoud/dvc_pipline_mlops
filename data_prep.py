## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from PIL import Image


outs_data_path = os.path.join(os.getcwd(), "outs_data")
if not os.path.exists(outs_data_path):
    os.makedirs(outs_data_path)


## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), "dataset.csv")
df = pd.read_csv(TRAIN_PATH)


## Drop first 3 features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

## Filtering using Age Feature using threshold
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

# save data

df.to_csv(os.path.join(outs_data_path, "prepared_data.csv"), index=False)