# Transliteration using Attention-based Seq2Seq Models

This repository contains implementations of vanilla and attention-based sequence-to-sequence (Seq2Seq) models for transliteration from **Latin script to Devanagari**, using the [Dakshina dataset](https://github.com/google-research/dakshina).

##  Environment Setup

1. **Clone this repository** and navigate into the folder.
2. **Create a virtual environment** and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
3. **Download the Dakshina Dataset v1.0** and place it in the root directory or update the path in `load_data.py`.

---

##  Project Structure

````

├── requirements.txt              # Required Python packages
├── load\_data.py                  # Loads Dakshina data and returns train/dev/test DataLoaders
├── Attention/
│   ├── attention.py              # Dual-attention model, cross-attention plotting (static + interactive)
│   ├── attention\_sweep.py       # Sweep config for training attention models with W\&B
│   ├── attention.ipynb          # Demo notebook showing model training, prediction, and visualization
│   └── predictions\_attention.tsv # Predictions on test set using attention model
├── Vanilla/
│   ├── seq2seq\_model.py         # Vanilla Seq2Seq model (RNN/LSTM/GRU configurable)
│   ├── vanilla\_sweep.py         # Sweep config for vanilla model training with W\&B
│   ├── vanilla.ipynb            # Demo notebook showing usage of the vanilla model
│   └── vanilla\_attention.tsv    # Predictions on test set using vanilla model

````

---

##  Data Loading

```python
from load_data import get_data_loaders

train_loader, dev_loader, test_loader, char_to_idx_latin, char_to_idx_devanagari = get_data_loaders()
````

---

##  Attention Model: Usage

### Initialize

```python
from attention import DualAttentionSeq2Seq

model = DualAttentionSeq2Seq(
    input_vocab_size=input_vocab_size,
    output_vocab_size=output_vocab_size,
    embedding_dim=256,
    hidden_dim=256,
    num_layers=1,
    num_heads=4,
    dropout=0.2,
    device=device
).to(device)
```

### Training Loop

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg[:, :-1])
        loss = criterion(output.reshape(-1, output_vocab_size), trg[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

##  Visualization

### Static Cross-Attention Heatmap

```python
from attention import plot_cross_attention_heatmaps

fig = plot_cross_attention_heatmaps(model, src, trg, idx_to_input, idx_to_output)
```

### Interactive Cross-Attention Heatmap

```python
from attention import interactive_dualline_attention

fig = interactive_dualline_attention(model, src, trg, idx_to_input, idx_to_devanagari, idx_to_latin)
```

> Hover over Devanagari characters to highlight contributing Latin characters based on attention scores.

---

##  Vanilla Seq2Seq Model: Usage

```python
from Vanilla.seq2seq_model import Seq2SeqModel

model = Seq2SeqModel(
    input_vocab_size=input_vocab_size,
    output_vocab_size=output_vocab_size,
    embedding_dim=64,
    hidden_dim=256,
    cell_type="lstm",
    num_layers_encoder=1,
    num_layers_decoder=1,
    dropout=0.2,
    device=device,
    beam_size=2
).to(device)
```

### Training & Validation

```python
# Same structure as attention model training
# Add validation and save best model
torch.save(model.state_dict(), 'best_vanilla_model.pt')
```

---

##  WandB Logging

Both models support hyperparameter sweeps via `wandb`.

```bash
# Edit sweep_config inside attention_sweep.py or vanilla_sweep.py
python attention/attention_sweep.py
python Vanilla/vanilla_sweep.py
```

