## Encoder-Decoder Seq2Seq

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Are Seq2Seq Models?

### Motivation and Intuition

Sequence-to-sequence (Seq2Seq) models transform one sequence into another when input and output lengths differ. Classic examples: machine translation ("let's go" -> "vamos"), text summarization, and speech recognition. The core challenge is that neither the input nor output length is fixed, and they can differ.

The solution is an **encoder-decoder architecture**: an encoder reads the entire input sequence and compresses it into a fixed-size context vector; a decoder then uses that context vector to generate the output sequence token by token.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Seq2Seq | Model that maps variable-length input to variable-length output | Enables tasks like translation where input/output lengths differ |
| Encoder | RNN that reads the input sequence | Produces a context vector summarizing the input |
| Decoder | RNN that generates the output sequence from the context vector | Predicts one token at a time until an end-of-sequence token |

---

## 2. The Encoder: Reading the Input Sequence

### Motivation and Intuition

The encoder reads input words one at a time, updating a hidden state that accumulates information. Each input token is converted to a dense embedding vector, then fed into an RNN cell (LSTM or GRU). After processing all tokens, the final hidden state(s) form the **context vector**.

$$
h_t = \text{RNN}(x_t, h_{t-1})
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $x_t$ | Embedding of input token at time step $t$ | Dense vector representing the current word |
| $h_t$ | Hidden state at time step $t$ | Accumulates information from all tokens seen so far |
| $\text{RNN}$ | Recurrent cell (LSTM or GRU) with learned weights | Same cell reused at every time step (weight sharing) |
| $c$ | Context vector: final hidden/cell states from encoder | Fixed-size summary of the entire input sequence |

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)           # (batch, seq_len, embedding_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell  # context vector
```

---

## 3. The Context Vector: Bottleneck

### Motivation and Intuition

The context vector is the only connection between encoder and decoder. It must encode the entire meaning of the input sequence into a fixed-size vector — this is both the strength and the weakness of basic Seq2Seq. For long sequences, early words can be forgotten, motivating the attention mechanism.

$$
c = \{(h_1^{(L)}, c_1^{(L)}), (h_2^{(L)}, c_2^{(L)}), \dots, (h_N^{(L)}, c_N^{(L)})\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Context vector | Final set of hidden/cell states from all encoder LSTM layers | Carries full input meaning to decoder |
| Bottleneck | Fixed-size vector must encode arbitrarily long input | Long sequences lose detail — this motivates attention |

---

## 4. The Decoder: Generating Output

### Motivation and Intuition

The decoder takes the context vector as its initial state and generates the output sequence one token at a time. At each step, it receives the previous token, updates its hidden state, and produces a probability distribution over the output vocabulary via a linear layer with softmax. Generation stops when the decoder outputs `<EOS>`.

$$
P(y_t \mid y_{<t}, c) = \text{softmax}(W_o h_t^{(dec)} + b_o)
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $y_t$ | Output token at time step $t$ | The predicted word |
| $h_t^{(dec)}$ | Decoder hidden state at time step $t$ | Conditioned on context vector and all previous outputs |
| $W_o, b_o$ | Output projection weights and bias | Map decoder hidden state to vocabulary-sized logits |

```python
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x, hidden, cell):
        # x: (batch, 1) -- single token index
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell
```

---

## 5. Teacher Forcing

### Motivation and Intuition

During training, the decoder is fed the **ground-truth** token at each step instead of its own prediction. This stabilizes training and speeds convergence. Without teacher forcing, early wrong predictions would cascade and compound.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Teacher forcing | Using true target token as next decoder input during training | Stabilizes training; prevents error propagation |
| Scheduled sampling | Gradually reducing teacher forcing ratio as training progresses | Bridges gap between training and inference behavior |

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input_token = trg[:, 0].unsqueeze(1)  # <SOS> token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs
```

---

> **Check your intuition:** If you had to summarize a 50-word paragraph into a single sentence, you would choose the most important details and discard the rest. A Seq2Seq encoder does the same — the context vector is a lossy compression. Why does this work for short phrases like "let's go" but fail for long paragraphs? What architectural change would let the decoder inspect individual input words directly?

---

## Prerequisites and Further Reading

- [StatQuest: RNNs and LSTMs](https://www.youtube.com/watch?v=AsNTP8Kwu80)
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=D-ekhWxubTs)
- [StatQuest: Backpropagation](https://www.youtube.com/watch?v=i94OvYb6noo)
- Original Seq2Seq paper: Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)
- Attention paper: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
