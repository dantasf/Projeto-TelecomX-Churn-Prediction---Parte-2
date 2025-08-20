import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_data(path_csv="data/telecomx_processed.csv"):
    try:
        return pd.read_csv(path_csv)
    except:
        return pd.read_parquet(path_csv.replace(".csv", ".parquet"))

def drop_irrelevant(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = [c for c in ["customerid","id","clientid"] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")

def get_feature_target(df):
    target = None
    for t in ["churn","Churn","evasao","evasão"]:
        if t in df.columns:
            target = t; break
    if target is None:
        for c in df.columns:
            if "churn" in c.lower() or "evas" in c.lower():
                target = c; break
    X = df.drop(columns=[target])
    y = df[target]
    if y.dtype.kind not in "biufc":
        y = y.astype(str).str.lower().isin(["1","sim","yes","true"]).astype(int)
    return X, y, target

def build_preprocessor(X):
    num_cols = [c for c in X.columns if X[c].dtype.kind in "biufc"]
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, cat_cols

def split_scale(X, y, test_size=0.3, preprocessor=None, apply_smote=False, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if preprocessor is not None:
        pipe = Pipeline([("pre", preprocessor)])
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)
    if apply_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test