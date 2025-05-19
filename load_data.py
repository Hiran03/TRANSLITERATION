import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class TransliterationDataset(Dataset):
    def __init__(self, devanagari_samples, latin_samples, char_to_idx_devanagari, char_to_idx_latin):
        self.devanagari_samples = devanagari_samples
        self.latin_samples = latin_samples
        self.char_to_idx_devanagari = char_to_idx_devanagari
        self.char_to_idx_latin = char_to_idx_latin

    def __len__(self):
        return len(self.devanagari_samples)

    def __getitem__(self, idx):
        devanagari = str(self.devanagari_samples[idx])
        latin = str(self.latin_samples[idx])
        
        # Convert characters to indices (add <SOS>=1 and <EOS>=2)
        devanagari_indices = [1] + [self.char_to_idx_devanagari[char] for char in devanagari] + [2]
        latin_indices = [1] + [self.char_to_idx_latin[char] for char in latin] + [2]
        
        return torch.tensor(devanagari_indices, dtype=torch.long), torch.tensor(latin_indices, dtype=torch.long)

def build_vocab(samples):
    chars = set()
    for sample in samples:
        sample = str(sample)  # Ensure string type
        chars.update(sample)
    char_to_idx = {char: idx+3 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<SOS>'] = 1
    char_to_idx['<EOS>'] = 2
    return char_to_idx

def load_data(lang_code="hi", data_dir="dakshina_dataset_v1.0"):
    base_path = os.path.join(data_dir, lang_code, "lexicons")
    
    # Load TSV files with proper column order (Devanagari first, then Latin)
    train_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.train.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"], 
        usecols=[0, 1], dtype=str
    )
    dev_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.dev.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )
    test_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.test.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )

    # Build vocabularies
    char_to_idx_devanagari = build_vocab(train_df["devanagari"].tolist())
    char_to_idx_latin = build_vocab(train_df["latin"].tolist())

    # Create datasets
    train_dataset = TransliterationDataset(
        train_df["devanagari"].tolist(),
        train_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )
    dev_dataset = TransliterationDataset(
        dev_df["devanagari"].tolist(),
        dev_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )
    test_dataset = TransliterationDataset(
        test_df["devanagari"].tolist(),
        test_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )

    return train_dataset, dev_dataset, test_dataset, char_to_idx_devanagari, char_to_idx_latin

def collate_fn(batch):
    devanagari_batch, latin_batch = zip(*batch)
    
    # Pad sequences
    devanagari_padded = torch.nn.utils.rnn.pad_sequence(devanagari_batch, batch_first=True, padding_value=0)
    latin_padded = torch.nn.utils.rnn.pad_sequence(latin_batch, batch_first=True, padding_value=0)
    
    return devanagari_padded, latin_padded

def get_data_loaders(batch_size=32, lang_code="hi"):
    train_dataset, dev_dataset, test_dataset, char_to_idx_devanagari, char_to_idx_latin = load_data(lang_code)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, dev_loader, test_loader, char_to_idx_devanagari, char_to_idx_latin

def print_samples(loader, num_samples=5, char_to_idx_devanagari=None, char_to_idx_latin=None):
    devanagari_to_char = {v: k for k, v in char_to_idx_devanagari.items()} if char_to_idx_devanagari else None
    latin_to_char = {v: k for k, v in char_to_idx_latin.items()} if char_to_idx_latin else None
    
    print("\nSample pairs from dataset:")
    for i, (devanagari, latin) in enumerate(loader):
        if i >= num_samples:
            break
            
        # Convert indices back to characters for display
        devanagari_str = ''.join([devanagari_to_char.get(idx.item(), '?') for idx in devanagari[0] if idx not in {0, 1, 2}])
        latin_str = ''.join([latin_to_char.get(idx.item(), '?') for idx in latin[0] if idx not in {0, 1, 2}])
        
        print(f"{devanagari_str} → {latin_str}")

if __name__ == "__main__":
    train_loader, dev_loader, test_loader, char_to_idx_devanagari, char_to_idx_latin = get_data_loaders()
    
    print(f"Devanagari vocab size: {len(char_to_idx_devanagari)}")
    print(f"Latin vocab size: {len(char_to_idx_latin)}")
    
    # Print sample pairs
    print_samples(train_loader, 5, char_to_idx_devanagari, char_to_idx_latin)
    
    # Example batch
    devanagari_batch, latin_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Devanagari: {devanagari_batch.shape}")  # (batch_size, max_seq_len)
    print(f"Latin: {latin_batch.shape}")