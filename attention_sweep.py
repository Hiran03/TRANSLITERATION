import wandb
import torch
import math
from attention import DualAttentionSeq2Seq
from load_data import get_data_loaders
import torch.nn as nn

sweep_config = {
    'name': 'attention_sweep',
    'method': 'grid',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'embedding_dim': {'values': [256, 512]},
        'num_layers': {'values': [1,2]},
        'num_heads': {'values': [4, 8]},
        'dropout': {'values': [0.2]},
        'batch_size': {'values': [32]},
        'learning_rate': {'values': [10e-4]},
        
    }
}

def train():
    wandb.init()
    config = wandb.config
    
    # Load data (Latin → Devanagari)
    train_loader, dev_loader, test_loader, char_to_idx_latin, char_to_idx_devanagari = get_data_loaders()

    # Vocabulary info
    input_vocab_size = len(char_to_idx_latin)
    output_vocab_size = len(char_to_idx_devanagari)
    idx_to_input = {v: k for k, v in char_to_idx_latin.items()}
    idx_to_output = {v: k for k, v in char_to_idx_devanagari.items()}
    # Index maps
    idx_to_latin = {v: k for k, v in char_to_idx_latin.items()}
    idx_to_devanagari = {v: k for k, v in char_to_idx_devanagari.items()}
    device = torch.device("cuda")
    # Model
    model = DualAttentionSeq2Seq(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:, :-1])  # teacher forcing
            loss = criterion(
                output.reshape(-1, output_vocab_size),
                trg[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in dev_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg[:, :-1])
                val_loss += criterion(
                    output.reshape(-1, output_vocab_size),
                    trg[:, 1:].reshape(-1)
                ).item()
        
        val_loss /= len(dev_loader)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}')
    wandb.log({
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(dev_loader)
        })


sweep_id = wandb.sweep(sweep_config, project="transliteration-sweep")
wandb.agent(sweep_id, function=train)