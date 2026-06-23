"""Train an external (third-party) LLM imported from Hugging Face.

The "import" half of the experiment: architecture and weights come from the
transformers library. Counterpart: create_transformer.py (same flow, but builds
the network from scratch). Shared infra (tokenizer, data, training, eval,
plotting) lives in training_utils.py; compare.py trains both side by side.

Run this file alone to train just this model.
"""

import torch.optim as optim
from transformers import AutoModelForCausalLM

import training_utils as tu


def make_step_fn():
    """Per-batch step for a HF causal LM -> (loss, num_correct, num_tokens)."""
    def step_fn(model, batch, device):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # The model computes its own loss; padding is ignored via -100.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Next-token accuracy: prediction at t targets token t+1.
        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        correct = ((preds == targets) & valid).sum().item()
        total = valid.sum().item()
        return outputs.loss, correct, total

    return step_fn


def build(tokenizer, device, model_name="distilgpt2", learning_rate=5e-5):
    """Load the pretrained model and return a ModelSpec (model moved to device)."""
    print(f"Loading pretrained model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return tu.ModelSpec(
        name=f"external:{model_name}",
        model=model,
        optimizer=optimizer,
        step_fn=make_step_fn(),
    )


def main():
    model_name = "distilgpt2"  # any HF causal LM, e.g. "gpt2", "gpt2-medium"
    batch_size, epochs, max_length = 8, 12, 64
    num_train, num_eval = 300, 60
    use_real_data = False  # set True to use real_data.py (needs `datasets`)

    device = tu.get_device()
    tokenizer = tu.load_tokenizer(model_name)
    train_loader, eval_loader, _ = tu.load_data(
        tokenizer, batch_size, max_length, num_train, num_eval, use_real_data
    )

    spec = build(tokenizer, device, model_name=model_name)
    history, metrics = tu.run_single(spec, train_loader, eval_loader, device, epochs)
    tu.plot_history(spec.name, history, "import_transformer_curves.png")
    tu.save_model(spec.model, "pretrained_transformer_model.pt")
    return spec.model, history, metrics


if __name__ == "__main__":
    main()
