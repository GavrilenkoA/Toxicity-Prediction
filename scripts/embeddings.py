import hashlib
import base64
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import h5py


def generate_hash(s: str, length: int = 8) -> str:
    """
    Возвращает хэш фиксированной длины (по умолчанию 5 символов) для строки `s`.
    """
    # Получаем sha256-хэш
    full_hash = hashlib.sha256(s.encode()).digest()

    # Кодируем в base64 и оставляем только нужное количество символов
    short = base64.urlsafe_b64encode(full_hash).decode()[:length]
    return short


def load_data(df_path: str):
    df = pd.read_csv(df_path)
    mapper_df = df[['SMILES']].copy()
    mapper_df['SMILES ID'] = mapper_df['SMILES'].apply(generate_hash)
    return df, mapper_df


def load_model_and_tokenizer(model_name: str = 'DeepChem/ChemBERTa-10M-MLM'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_embeddings(df: pd.DataFrame, mapper_df: pd.DataFrame, tokenizer, model, device) -> dict:
    embeddings = {}
    model = model.to(device)
    for smi in tqdm(df['SMILES'], total=len(df)):
        smi_id = mapper_df.loc[mapper_df['SMILES'] == smi, 'SMILES ID'].values[0]
        encoded_inputs = tokenizer(smi, padding=False, return_tensors="pt")
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
        embeddings[smi_id] = embedding
    return embeddings


def save_embeddings_to_hdf5(embeddings: dict, file_path: str):
    with h5py.File(file_path, 'w') as h5f:
        for smi_id, embedding in embeddings.items():
            h5f.create_dataset(smi_id, data=embedding)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    df, mapper_df = load_data('data/df_copied.csv')
    tokenizer, model = load_model_and_tokenizer()
    embeddings = generate_embeddings(df, mapper_df, tokenizer, model, device)
    save_embeddings_to_hdf5(embeddings, 'data/chemberta_pooler_embeddings.h5')
    mapper_df.to_csv('data/mapper_df.csv', index=False)


if __name__ == "__main__":
    main()
