import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from rdkit.Chem import SaltRemover, MolStandardize
import io
from contextlib import redirect_stderr


def filter_and_log(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Process a DataFrame of SMILES strings, stripping salts and canonicalizing tautomers.
    Returns a new DataFrame with valid molecules (including the original index) and a dict of per-row logs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'SMILES' and 'toxicity'.

    Returns
    -------
    processed_df : pd.DataFrame
        Columns:
          - 'orig_idx': original DataFrame index
          - 'SMILES': cleaned canonical SMILES
          - 'toxicity': original labels
          - 'mol': RDKit Mol objects
    logs : dict[int, str]
        Mapping from original DataFrame index to captured stderr log.
    """
    remover = SaltRemover.SaltRemover()
    canonicalizer = MolStandardize.rdMolStandardize.TautomerEnumerator()

    log_kekulize = "Can't kekulize mol."
    log_remove_hydrogen = "not removing hydrogen atom without neighbors"
    log_tautomer_enumeration_stopped = "Tautomer enumeration stopped"

    clean_data = {
        'orig_idx': [],
        'SMILES': [],
        'toxicity': [],
        'mol': []
    }
    logs = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering molecules"):
        smi = row['SMILES']
        label = row['toxicity']

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                stderr_buffer.write(f"[{idx}] Invalid SMILES: {smi}\n")
            else:
                mol = remover.StripMol(mol, dontRemoveEverything=True)
                mol = canonicalizer.Canonicalize(mol)
                smi_clean = Chem.MolToSmiles(mol, canonical=True)

        log_text = stderr_buffer.getvalue()
        stderr_buffer.close()

        # Save log if any
        if log_text:
            logs[idx] = log_text

        # Only keep molecules that parsed correctly and have no kekulization errors
        if mol is not None and log_kekulize not in log_text and log_remove_hydrogen not in log_text and log_tautomer_enumeration_stopped not in log_text:
            clean_data['orig_idx'].append(idx)
            clean_data['SMILES'].append(smi_clean)
            clean_data['toxicity'].append(label)
            clean_data['mol'].append(mol)

    processed_df = pd.DataFrame(clean_data)
    processed_df = processed_df.set_index('orig_idx')
    delta = len(processed_df) / len(df) * 100
    print(f"Delta: {delta:.2f}%")
    return processed_df, logs


def is_organic(mol):
    # Проверим, содержит ли молекула углерод
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def is_valid_organic_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return mol.GetNumAtoms() > 1 and is_organic(mol)


def load_embeddings_from_hdf5(file_path: str) -> dict:
    embeddings = {}
    with h5py.File(file_path, 'r') as h5f:
        for smi_id in h5f.keys():
            embeddings[smi_id] = h5f[smi_id][:]
    return embeddings


def remove_ambiguous_smiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove ambiguous SMILES from the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing 'SMILES' and 'toxicity' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ambiguous SMILES removed and duplicates dropped.
    """
    duplicated_smiles = dataframe[dataframe.duplicated('SMILES', keep=False)]
    smiles_counts = duplicated_smiles.groupby('SMILES')['toxicity'].nunique()
    ambiguous_smiles = smiles_counts[smiles_counts > 1].index

    dataframe = dataframe[~dataframe['SMILES'].isin(ambiguous_smiles)]
    return dataframe.drop_duplicates(subset='SMILES')


def form_data(df, train_idx, test_idx):
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    return df_train, df_test


def compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, pos_label=0),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }
    return metrics


def balance_classes(df: pd.DataFrame, target_col: str, method: str = 'oversample') -> pd.DataFrame:
    """
    Балансирует классы в датафрейме.

    Параметры:
    - df: pd.DataFrame — входной датафрейм
    - target_col: str — имя колонки с целевой переменной
    - method: str — 'oversample' (по умолчанию) или 'undersample'

    Возвращает:
    - сбалансированный DataFrame
    """
    # Разделим по классам
    classes = df[target_col].unique()
    dfs = [df[df[target_col] == c] for c in classes]

    if method == 'oversample':
        max_len = max(len(subdf) for subdf in dfs)
        balanced_dfs = [
            resample(subdf, replace=True, n_samples=max_len, random_state=42)
            for subdf in dfs
        ]
    elif method == 'undersample':
        min_len = min(len(subdf) for subdf in dfs)
        balanced_dfs = [
            resample(subdf, replace=False, n_samples=min_len, random_state=42)
            for subdf in dfs
        ]
    else:
        raise ValueError("method должен быть 'oversample' или 'undersample'")

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)


def balance_with_interpolate(train_df, target_name='toxicity', class_balancer=None, random_state=42):
    """
    Балансирует обучающие данные с помощью ADASYN или SMOTE.
    """
    X_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    balancer = class_balancer(random_state=random_state)
    X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)
    train_df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
    return train_df_balanced


def featurize_ecfp4(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
    prefer_generator: bool = True
) -> np.ndarray:
    """
    Generate an ECFP4 (Morgan radius=2) fingerprint for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to featurize.
    radius : int, optional
        The radius for Morgan fingerprint (ECFP diameter = 2*radius).
        Default is 2 for ECFP4.
    n_bits : int, optional
        Length of the bit vector. Default is 2048.
    use_chirality : bool, optional
        Whether to include chirality information. Default is False.
    prefer_generator : bool, optional
        If True, uses the new rdFingerprintGenerator API; otherwise the classic helper.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of 0/1 ints of length `n_bits`.
    """

    # 1) generate the RDKit bitvector
    if prefer_generator:
        # modern generator API
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality
        )
        bitvect = gen.GetFingerprint(mol)
    else:
        # classic helper function
        bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, useChirality=use_chirality
        )

    # 2) convert to NumPy array
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits=5,    # для valid ≈ 1/3
    random_state=42,
    shuffle=True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает train_idx, test_idx, valid_idx для трёх наборов данных,
    стратифицированных по y и разбитых по группам groups.

    - valid: выделяется первым (n_splits_outer)
    - train/test: выделяются из оставшихся (n_splits_inner)
    """
    if groups is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y))

    return train_idx, test_idx


def embeddings_dict_to_df(embeddings_dict):
    data = []
    for key, embedding in embeddings_dict.items():
        data.append((key, embedding))
    return pd.DataFrame(data, columns=['SMILES', 'Embedding'])


def expand_embeddings(df: pd.DataFrame, id_col='SMILES', embedding_col='Embedding') -> pd.DataFrame:
    embedding_matrix = np.stack(df[embedding_col].values)
    component_columns = [f'Component_{i}' for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=component_columns)
    return pd.concat([df[[id_col]], embedding_df], axis=1)


class BinaryDataset(Dataset):
    def __init__(self, df, target_col='toxicity', id_col='SMILES'):
        self.X = torch.tensor(df.drop(columns=[target_col, id_col]).values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
        self.id = df[id_col].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InferenceDataset(Dataset):
    def __init__(self, df, target_col='toxicity', id_col='SMILES'):
        self.X = torch.tensor(df.drop(columns=[target_col, id_col]).values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                n_epochs=50, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Валидация
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"[{epoch+1}] Train loss: {np.mean(train_losses):.4f}, Val loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_score.extend(probs.flatten())
            y_pred.extend(preds.flatten())
            y_true.extend(y_batch.numpy().flatten())

    metrics = compute_metrics(y_true, y_pred, y_score)
    return metrics


def predict_by_nn(model, loader, device):
    model.eval()
    y_pred, y_score = [], []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_score.extend(probs.flatten())
            y_pred.extend(preds.flatten())

    return pd.DataFrame({'pred': y_pred, 'score': y_score})
