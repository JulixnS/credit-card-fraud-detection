import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path)
    top_features = ['V17', 'V12', 'V14', 'V16', 'V10', 'V11', 'V18', 'V9', 'V7', 'V4', 'Class']
    return df[top_features]

# print(load_data("../data/creditcard.csv").head())

def split_data(df):
    X = df.drop("Class", axis = 1)
    y = df["Class"]

    X_train,  X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size = 0.8,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    smote = SMOTE(random_state = 42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, y_train_smote, X_test, y_test




