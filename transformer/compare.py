"""Entry point: train both transformers on identical data and graph the result.

This is the top of the dependency graph — it imports the shared library and both
model modules. Keeping the orchestration here (rather than in training_utils)
avoids an import cycle:

    fake_data / real_data        (leaves)
            training_utils       (pure library)
    import_transformer / create_transformer
            compare              (this file)
"""

import training_utils as tu
import import_transformer
import create_transformer


def run_comparison(model_name="distilgpt2", epochs=12, batch_size=8, max_length=64,
                   num_train=300, num_eval=60, use_real_data=False):
    """Train both models on the same data and save comparison graphs."""
    device = tu.get_device()
    tokenizer = tu.load_tokenizer(model_name)
    train_loader, eval_loader, _ = tu.load_data(
        tokenizer, batch_size, max_length, num_train, num_eval, use_real_data
    )

    # Both models share the tokenizer/vocab so they see identical token ids.
    specs = [
        import_transformer.build(tokenizer, device, model_name=model_name),
        create_transformer.build(tokenizer, device, max_seq_len=max_length),
    ]

    histories, evals = {}, {}
    for spec in specs:
        print(f"\n=== {spec.name} ===")
        history, metrics = tu.run_single(spec, train_loader, eval_loader, device, epochs)
        histories[spec.name] = history
        evals[spec.name] = metrics

    tu.print_summary(evals)
    suffix = "real" if use_real_data else "fake"
    tu.plot_comparison(histories, evals, f"model_comparison_{suffix}.png")
    return histories, evals


if __name__ == "__main__":
    run_comparison()
