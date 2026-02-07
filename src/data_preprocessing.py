import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    trans = pd.read_csv("data/raw/train_transaction.csv")
    identity = pd.read_csv("data/raw/train_identity.csv")
    df = trans.merge(identity, on="TransactionID", how="left")
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])
    X = X.fillna(-999)
    X = X.select_dtypes(include=["int64", "float64"])
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names
def preprocess_single_sample(index):
    import pandas as pd

    trans = pd.read_csv("data/raw/train_transaction.csv")
    identity = pd.read_csv("data/raw/train_identity.csv")

    df = trans.merge(identity, on="TransactionID", how="left")

    X = df.drop(columns=["isFraud", "TransactionID"])
    X = X.fillna(-999)
    X = X.select_dtypes(include=["int64", "float64"])

    sample = X.iloc[[index]]  
    return sample
def get_bulk_samples(n=300, random_state=None):
    import pandas as pd

    trans = pd.read_csv("data/raw/train_transaction.csv")
    identity = pd.read_csv("data/raw/train_identity.csv")

    df = trans.merge(identity, on="TransactionID", how="left")

    X = df.drop(columns=["isFraud", "TransactionID"])
    X = X.fillna(-999)
    X = X.select_dtypes(include=["int64", "float64"])

    return X.sample(
        n=min(n, len(X)),
        random_state=random_state
    )



