import wandb
import torch
from seq2seq_model import Seq2SeqModel
import torch.nn as nn
from load_data import get_data_loaders

# Sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'embedding_dim': {'values': [16, 32, 64, 128, 256]},
        'hidden_dim': {'values': [64, 128, 256, 512]},
        'cell_type': {'values': ['RNN', 'GRU', 'LSTM']},
        'num_layers_encoder': {'values': [1, 2, 3]},
        'num_layers_decoder': {'values': [1, 2, 3]},
        'dropout': {'values': [0.2, 0.3, 0.5]},
        'batch_size': {'values': [32, 64, 128]},
        'learning_rate': {'distribution': 'log_uniform', 'min': -5, 'max': -3}
    }
}

def train():
    # Initialize W&B run
    wandb.init()
    config = wandb.config
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, dev_loader, _, char_to_idx_devanagari, char_to_idx_latin = get_data_loaders(
        batch_size=config.batch_size
    )
    
    # Model initialization
    model = Seq2SeqModel(
        input_vocab_size=len(char_to_idx_devanagari),
        output_vocab_size=len(char_to_idx_latin),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        cell_type=config.cell_type,
        num_layers_encoder=config.num_layers_encoder,
        num_layers_decoder=config.num_layers_decoder,
        dropout=config.dropout,
        device=device
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(10):  # Fixed epoch number for sweep
        model.train()
        train_loss = 0
        
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            
            # Forward with teacher forcing
            output = model(src, trg[:, :-1])
            loss = criterion(output.reshape(-1, len(char_to_idx_latin)), 
                           trg[:, 1:].reshape(-1))
            
            # Backprop
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
                loss = criterion(output.reshape(-1, len(char_to_idx_latin)), 
                              trg[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(dev_loader)
        })

if __name__ == '__main__':
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="seq2seq-transliteration")
    wandb.agent(sweep_id, function=train, count=50)  # Run 50 iterations