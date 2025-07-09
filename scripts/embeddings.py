import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str = 'DeepChem/ChemBERTa-10M-MLM'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_embeddings(df: pd.DataFrame, tokenizer, model, device) -> dict:
    embeddings = {}
    model = model.to(device)
    for smi in tqdm(df['SMILES'], total=len(df)):
        encoded_inputs = tokenizer(smi, padding=False, return_tensors="pt")
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
        embeddings[smi] = embedding
    return embeddings


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/df_organic_cleaned.csv')
    tokenizer, model = load_model_and_tokenizer()
    embeddings = generate_embeddings(df, tokenizer, model, device)
    torch.save(embeddings, 'data/chemberta_embeddings.pt')


if __name__ == "__main__":
    main()
