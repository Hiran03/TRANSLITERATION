import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_vocab_size,       # Size of Latin character vocabulary
        output_vocab_size,      # Size of Devanagari character vocabulary
        embedding_dim=128,      # Dimension of character embeddings
        hidden_dim=256,         # Hidden state dimension
        cell_type="lstm",       # RNN cell type: "rnn", "lstm", or "gru"
        num_layers_encoder=1,   # Number of encoder layers
        num_layers_decoder=1,   # Number of decoder layers
        dropout=0.1,            # Dropout rate
        device="cpu"            # Device (cpu/cuda)
    ):
        super(Seq2SeqModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder

        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self._init_rnn_cell(
            encoder=True,
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_encoder,
            dropout=dropout,
        )

        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self._init_rnn_cell(
            encoder=False,
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def _init_rnn_cell(self, encoder, input_dim, hidden_dim, num_layers, dropout):
        if self.cell_type == "rnn":
            rnn_class = nn.RNN
        elif self.cell_type == "lstm":
            rnn_class = nn.LSTM
        elif self.cell_type == "gru":
            rnn_class = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {self.cell_type}")

        if encoder:
            self.encoder_rnn = rnn_class(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.decoder_rnn = rnn_class(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )

    def forward(self, src, trg=None, max_len=None, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_seq_len)
        # trg: (batch_size, trg_seq_len) (only used in training)
        batch_size = src.shape[0]

        # --- Encoder ---
        encoder_embedded = self.encoder_embedding(src)  # (batch_size, src_seq_len, embedding_dim)
        encoder_outputs, hidden = self._run_rnn(self.encoder_rnn, encoder_embedded)

        # --- Decoder ---
        if trg is not None:  # Training mode
            trg_embedded = self.decoder_embedding(trg)  # (batch_size, trg_seq_len, embedding_dim)
            decoder_outputs, _ = self._run_rnn(
                self.decoder_rnn,
                trg_embedded,
                hidden,
                teacher_forcing=True,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            outputs = self.fc_out(decoder_outputs)
        else:  # Inference mode
            outputs = self._decode_greedy(hidden, max_len, batch_size)

        return outputs  # (batch_size, seq_len, output_vocab_size)

    def _run_rnn(self, rnn, embedded, hidden=None, teacher_forcing=False, teacher_forcing_ratio=0.5):
        if hidden is None:
            if self.cell_type == "lstm":
                h0 = torch.zeros(rnn.num_layers, embedded.size(0), self.hidden_dim).to(self.device)
                c0 = torch.zeros(rnn.num_layers, embedded.size(0), self.hidden_dim).to(self.device)
                hidden = (h0, c0)
            else:
                hidden = torch.zeros(rnn.num_layers, embedded.size(0), self.hidden_dim).to(self.device)

        output, hidden = rnn(embedded, hidden)
        return output, hidden

    def _decode_greedy(self, hidden, max_len, batch_size):
        # Initialize with <SOS> token (assuming 0 is SOS)
        input = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
        outputs = []

        for _ in range(max_len):
            embedded = self.decoder_embedding(input)  # (batch_size, 1, embedding_dim)
            output, hidden = self._run_rnn(self.decoder_rnn, embedded, hidden)
            output = self.fc_out(output.squeeze(1))  # (batch_size, output_vocab_size)
            _, topi = output.topk(1)
            outputs.append(output.unsqueeze(1))
            input = topi.detach()  # Next input is current prediction

        return torch.cat(outputs, dim=1)  # (batch_size, max_len, output_vocab_size)