import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class TransliterationDataset(Dataset):
    def __init__(self, latin_samples, devanagari_samples, char_to_idx_latin, char_to_idx_devanagari):
        self.latin_samples = latin_samples
        self.devanagari_samples = devanagari_samples
        self.char_to_idx_latin = char_to_idx_latin
        self.char_to_idx_devanagari = char_to_idx_devanagari

    def __len__(self):
        return len(self.latin_samples)

    def __getitem__(self, idx):
        latin = str(self.latin_samples[idx])
        devanagari = str(self.devanagari_samples[idx])
        
        latin_indices = [1] + [self.char_to_idx_latin[char] for char in latin] + [2]
        devanagari_indices = [1] + [self.char_to_idx_devanagari[char] for char in devanagari] + [2]
        
        return torch.tensor(latin_indices, dtype=torch.long), torch.tensor(devanagari_indices, dtype=torch.long)

def build_vocab(samples):
    chars = set()
    for sample in samples:
        sample = str(sample)
        chars.update(sample)
    char_to_idx = {char: idx+3 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<SOS>'] = 1
    char_to_idx['<EOS>'] = 2
    return char_to_idx

def load_data(lang_code="hi", data_dir="dakshina_dataset_v1.0"):
    base_path = os.path.join(data_dir, lang_code, "lexicons")
    
    train_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.train.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"], 
        usecols=[0, 1], dtype=str
    )
    dev_df = pd.read_csv(os.path.join(base_path, f"{lang_code}.translit.sampled.dev.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )
    test_df = pd.read_csv(os.path.join(base_path, f"{lang_code}.translit.sampled.test.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )

    char_to_idx_latin = build_vocab(train_df["latin"].tolist())
    char_to_idx_devanagari = build_vocab(train_df["devanagari"].tolist())

    train_dataset = TransliterationDataset(
        train_df["latin"].tolist(),
        train_df["devanagari"].tolist(),
        char_to_idx_latin,
        char_to_idx_devanagari
    )
    dev_dataset = TransliterationDataset(
        dev_df["latin"].tolist(),
        dev_df["devanagari"].tolist(),
        char_to_idx_latin,
        char_to_idx_devanagari
    )
    test_dataset = TransliterationDataset(
        test_df["latin"].tolist(),
        test_df["devanagari"].tolist(),
        char_to_idx_latin,
        char_to_idx_devanagari
    )

    return train_dataset, dev_dataset, test_dataset, char_to_idx_latin, char_to_idx_devanagari

def collate_fn(batch):
    latin_batch, devanagari_batch = zip(*batch)
    latin_padded = torch.nn.utils.rnn.pad_sequence(latin_batch, batch_first=True, padding_value=0)
    devanagari_padded = torch.nn.utils.rnn.pad_sequence(devanagari_batch, batch_first=True, padding_value=0)
    return latin_padded, devanagari_padded

def get_data_loaders(batch_size=32, lang_code="hi"):
    train_dataset, dev_dataset, test_dataset, char_to_idx_latin, char_to_idx_devanagari = load_data(lang_code)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, char_to_idx_latin, char_to_idx_devanagari

if __name__ == "__main__":
    train_loader, dev_loader, test_loader, char_to_idx_latin, char_to_idx_devanagari = get_data_loaders()
    print(f"Latin vocab size: {len(char_to_idx_latin)}")
    print(f"Devanagari vocab size: {len(char_to_idx_devanagari)}")
    
    # Print sample pairs
    latin, devanagari = next(iter(train_loader))
    idx_to_latin = {v: k for k, v in char_to_idx_latin.items()}
    idx_to_devanagari = {v: k for k, v in char_to_idx_devanagari.items()}
    
    print("\nSample pairs:")
    for i in range(3):
        latin_str = ''.join([idx_to_latin[idx.item()] for idx in latin[i] if idx > 2])
        devanagari_str = ''.join([idx_to_devanagari[idx.item()] for idx in devanagari[i] if idx > 2])
        print(f"{latin_str} → {devanagari_str}")