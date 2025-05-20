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
        'epochs': {'values': [30]},
        'embedding_dim': {'values': [512]},
        'hidden_dim': {'values': [512]},
        'num_layers': {'values': [1,2]},
        'num_heads': {'values': [4, 8, 16]},
        'dropout': {'values': [0.2]},
        'batch_size': {'values': [32]},
        'learning_rate': {'values': [10e-4]},
        
    }
}

def train():
    wandb.init()
    config = wandb.config
    
    # Initialize data loaders and model
    train_loader, dev_loader, test_loader, char_to_idx_devanagari, char_to_idx_latin = get_data_loaders()
    input_vocab_size = len(char_to_idx_devanagari)
    output_vocab_size = len(char_to_idx_latin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = DualAttentionSeq2Seq(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for src, trg in train_loader:
            # Move data to device
            src = src.to(device)
            trg = trg.to(device)
            
            # Forward pass
            output = model(src, trg[:, :-1])  # Teacher forcing with shifted target
            
            # Calculate loss (ignore padding)
            loss = criterion(output.reshape(-1, output_vocab_size), 
                            trg[:, 1:].reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for src, trg in dev_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg[:, :-1])
                loss = criterion(output.reshape(-1, output_vocab_size), 
                            trg[:, 1:].reshape(-1))
                val_loss += loss.item()
            print(f'Validation Loss: {val_loss/len(dev_loader):.4f}')
    wandb.log({
            'epoch': epoch,
            'train_loss': total_loss/len(train_loader),
            'val_loss': val_loss/len(dev_loader)
        })


sweep_id = wandb.sweep(sweep_config, project="transliteration-sweep")
wandb.agent(sweep_id, function=train)