"""Train a transformer built from scratch in PyTorch.

The "create" half of the experiment: the architecture is defined here with
torch.nn instead of being imported. Counterpart: import_transformer.py (same
flow, but imports a pretrained Hugging Face LLM). Shared infra (tokenizer, data,
training, eval, plotting) lives in training_utils.py; compare.py trains both
side by side.

Run this file alone to train just this model.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import training_utils as tu


class TransformerBlock(nn.Module):
    """Multi-head self-attention + feed-forward, with residual connections."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention sublayer.
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward sublayer.
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class TransformerModel(nn.Module):
    """Causal transformer: embedding + positional encoding + blocks + head."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)

        # Embed tokens and add positional encoding.
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Causal mask (True = blocked) prevents attending to future tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for block in self.transformer_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        return self.output_layer(x)


def create_model(vocab_size, d_model=256, num_heads=4, d_ff=512,
                 num_layers=2, max_seq_len=64, dropout=0.1):
    """Factory for a small transformer (sized for a fast comparison run)."""
    return TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout)


def make_step_fn():
    """Per-batch step for the scratch model -> (loss, num_correct, num_tokens)."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def step_fn(model, batch, device):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Predict token t+1 from tokens up to t; ignore padding in the loss.
        inputs = input_ids[:, :-1]
        in_mask = attention_mask[:, :-1]
        targets = input_ids[:, 1:].clone()
        target_mask = attention_mask[:, 1:].bool()
        targets[~target_mask] = -100

        logits = model(inputs, attention_mask=in_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        preds = logits.argmax(dim=-1)
        correct = ((preds == input_ids[:, 1:]) & target_mask).sum().item()
        total = target_mask.sum().item()
        return loss, correct, total

    return step_fn


def build(tokenizer, device, max_seq_len=64, learning_rate=1e-3):
    """Create the from-scratch model and return a ModelSpec (model on device).

    Vocabulary comes from the shared tokenizer so both models see the same ids.
    """
    vocab_size = len(tokenizer)
    print(f"Building from-scratch transformer (vocab_size={vocab_size}, max_seq_len={max_seq_len})")
    model = create_model(vocab_size, max_seq_len=max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return tu.ModelSpec(
        name="scratch:pytorch-transformer",
        model=model,
        optimizer=optimizer,
        step_fn=make_step_fn(),
    )


def main():
    model_name = "distilgpt2"  # used only for its tokenizer / vocabulary
    batch_size, epochs, max_length = 8, 12, 64
    num_train, num_eval = 300, 60
    use_real_data = False  # set True to use real_data.py (needs `datasets`)

    device = tu.get_device()
    tokenizer = tu.load_tokenizer(model_name)
    train_loader, eval_loader, _ = tu.load_data(
        tokenizer, batch_size, max_length, num_train, num_eval, use_real_data
    )

    spec = build(tokenizer, device, max_seq_len=max_length)
    history, metrics = tu.run_single(spec, train_loader, eval_loader, device, epochs)
    tu.plot_history(spec.name, history, "create_transformer_curves.png")
    tu.save_model(spec.model, "scratch_transformer_model.pt")
    return spec.model, history, metrics


if __name__ == "__main__":
    main()
