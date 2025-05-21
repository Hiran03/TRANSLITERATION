import torch
import torch.nn as nn
import math

class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        cell_type="lstm",
        num_layers_encoder=1,
        num_layers_decoder=1,
        dropout=0.3,
        device="cpu",
        beam_size = 2
    ):
        super(Seq2SeqModel, self).__init__()
        self.device = device
        self.output_vocab_size = output_vocab_size
        self.cell_type = cell_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.beam_size = beam_size
        
        # Dropout layers
        self.embedding_dropout = nn.Dropout(dropout * 0.5)
        self.output_dropout = nn.Dropout(dropout * 0.8)


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
        rnn_class = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }.get(self.cell_type.lower(), nn.LSTM)

        rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        setattr(self, "encoder_rnn" if encoder else "decoder_rnn", rnn)

    def forward(self, src, trg=None, max_len=None, teacher_forcing_ratio=0.5):
        if self.beam_size > 1 and trg is None:
            return self._decode_beam(src, max_len, self.beam_size)
            
        # Original forward pass for training/greedy decoding
        encoder_embedded = self.embedding_dropout(self.encoder_embedding(src))
        _, hidden = self._run_rnn(self.encoder_rnn, encoder_embedded)

        if trg is not None:
            trg_embedded = self.embedding_dropout(self.decoder_embedding(trg))
            decoder_outputs, _ = self._run_rnn(
                self.decoder_rnn, trg_embedded, hidden, teacher_forcing_ratio
            )
            return self.fc_out(self.output_dropout(decoder_outputs))
        else:
            return self._decode_greedy(hidden, max_len)

    def _decode_beam(self, src, max_len, beam_size):
        """Beam search decoding"""
        # Encode source
        encoder_embedded = self.embedding_dropout(self.encoder_embedding(src))
        _, hidden = self._run_rnn(self.encoder_rnn, encoder_embedded)

        # Initialize beam
        batch_size = src.size(0)
        beams = [{
            'sequence': [1],  # Start with <SOS>
            'score': 0.0,
            'hidden': hidden
        } for _ in range(beam_size)]

        for _ in range(max_len):
            candidates = []
            for beam in beams:
                if beam['sequence'][-1] == 2:  # <EOS>
                    candidates.append(beam)
                    continue

                # Prepare decoder input
                input_tensor = torch.tensor([beam['sequence'][-1]], device=self.device).unsqueeze(0)
                embedded = self.embedding_dropout(self.decoder_embedding(input_tensor))
                
                # Forward pass
                output, new_hidden = self._run_rnn(
                    self.decoder_rnn, embedded, beam['hidden']
                )
                logits = self.fc_out(self.output_dropout(output.squeeze(1)))
                log_probs = torch.log_softmax(logits, dim=1)

                # Top k candidates
                topk_probs, topk_indices = torch.topk(log_probs, beam_size)
                for i in range(beam_size):
                    candidates.append({
                        'sequence': beam['sequence'] + [topk_indices[0, i].item()],
                        'score': beam['score'] + topk_probs[0, i].item(),
                        'hidden': new_hidden
                    })

            # Select top beams
            beams = sorted(candidates, key=lambda x: x['score'] / len(x['sequence']), reverse=True)[:beam_size]

        # Return best sequence (excluding <SOS>)
        best_sequence = beams[0]['sequence'][1:-1]  # Remove <SOS> and last token
        return torch.tensor(best_sequence, device=self.device).unsqueeze(0)

    def _decode_greedy(self, hidden, max_len):
        batch_size = hidden[0].size(1) if isinstance(hidden, tuple) else hidden.size(1)
        input = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
        outputs = []

        for _ in range(max_len):
            embedded = self.embedding_dropout(self.decoder_embedding(input))
            output, hidden = self._run_rnn(self.decoder_rnn, embedded, hidden)
            output = self.fc_out(self.output_dropout(output.squeeze(1)))
            _, topi = output.topk(1)
            outputs.append(output.unsqueeze(1))
            input = topi.detach()

        return torch.cat(outputs, dim=1)

    def _run_rnn(self, rnn, embedded, hidden=None, teacher_forcing_ratio=0.5):
        if hidden is None:
            shape = (rnn.num_layers, embedded.size(0), self.hidden_dim)
            if self.cell_type == "lstm":
                hidden = (torch.zeros(shape, device=self.device), 
                         torch.zeros(shape, device=self.device))
            else:
                hidden = torch.zeros(shape, device=self.device)
        return rnn(embedded, hidden)
    