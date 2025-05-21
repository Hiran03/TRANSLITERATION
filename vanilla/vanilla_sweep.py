from seq2seq_model import Seq2SeqModel 
from load_data import get_data_loaders 
import torch
import torch.nn as nn
import wandb

sweep_config = {
    'name': 'fine_search',
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {'values': [10, 15]},
        'embedding_dim': {'values': [64]},
        'hidden_dim': {'values': [256]},
        'cell_type': {'values': ['LSTM']},
        'num_layers_encoder': {'values': [1]},
        'dropout': {'values': [0.2]},
        'batch_size': {'values': [32, 64]},
        'beam_size': {'values': [1, 2]},
        'learning_rate': {'values' : [5e-3, 10e-3, 15e-3]}
        }
}




def train():
    wandb.init()
    config = wandb.config
    
    
    # Load data
    train_loader, dev_loader, test_loader, char_to_idx_latin, char_to_idx_devanagari = get_data_loaders()

    # Input is Latin, Output is Devanagari
    input_vocab_size = len(char_to_idx_latin)
    output_vocab_size = len(char_to_idx_devanagari)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        cell_type=config.cell_type,
        num_layers_encoder=config.num_layers_encoder,
        num_layers_decoder=config.num_layers_encoder,
        dropout=config.dropout,
        device=device,
        beam_size=config.beam_size
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    

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