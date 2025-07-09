import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import balance_classes, remove_ambiguous_smiles, filter_and_log, is_valid_organic_smiles, train_test_split, form_data, featurize_ecfp4


def filter_data(input_path: str):
    df = pd.read_csv(input_path)
    df, _ = filter_and_log(df)
    df_filtered = remove_ambiguous_smiles(df)
    df_filtered = df_filtered.sample(frac=1, random_state=42)
    df_filtered = df_filtered[df_filtered['SMILES'].apply(is_valid_organic_smiles)]
    df_filtered = df_filtered.dropna().reset_index(drop=True)
    return df_filtered


def split_data(df: pd.DataFrame, n_splits: int = 6):
    X = np.vstack([featurize_ecfp4(m) for m in df['mol']])
    y = df['toxicity'].values
    train_idx_random, test_idx_random = train_test_split(X, y, n_splits=n_splits)
    df_train_random_split, df_test_random_split = form_data(df, train_idx_random, test_idx_random)
    return df_train_random_split, df_test_random_split


def get_features(df):
    X_ecfp = np.vstack([featurize_ecfp4(m) for m in df['mol']])
    y = df['toxicity'].values
    return X_ecfp, y


def train(df_train):
    df_train_oversampled = balance_classes(df_train, target_col='toxicity', method='oversample')

    model = RandomForestClassifier(n_estimators=500, max_depth=None,
                                   min_samples_split=3,
                                   random_state=42, criterion='log_loss',
                                   min_samples_leaf=2,
                                   n_jobs=-1)

    X_train, y_train = get_features(df_train_oversampled)
    model.fit(X_train, y_train)
    return model


def predict(X_test, model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def collect_results(df_test, y_pred, y_proba):
    df_test['y_pred'] = y_pred
    df_test['y_proba'] = y_proba
    return df_test


def main():
    parser = argparse.ArgumentParser(description="Run inference on molecular dataset with pretrained MLP model")
    parser.add_argument("--input_path", required=True, help="Path to input CSV with SMILES and toxicity")
    parser.add_argument("--output_path", required=True, help="Path to save prediction CSV")

    args = parser.parse_args()

    df_filtered = filter_data(args.input_path)

    df_train, df_test = split_data(df_filtered)

    model = train(df_train)
    X_test, _ = get_features(df_test)
    y_pred, y_proba = predict(X_test, model)
    df_test = collect_results(df_test, y_pred, y_proba)
    df_test.to_csv(args.output_path, index=False, sep='\t')


if __name__ == "__main__":
    main()
