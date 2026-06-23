"""The REAL policy: a torch + transformers language model.

This is the real-model counterpart to ../test/generate_policy.py (MockPolicy). It
exposes the SAME generate(prompt, num_samples, correct_answer) interface, so the
shared scoring/advantage code is reused unchanged. Two differences from the mock:

    * generate() actually samples completions from a language model (it ignores
      `correct_answer` -- a real model must reason, it isn't handed the answer).
    * train_step() applies one real GRPO policy-gradient update from the group's
      advantages -- the part the simulation can only describe.

Libraries: torch (tensors, autograd, optimizer) and transformers (the model +
tokenizer). Everything else (parse_tags, score_rewards, compute_advantages) lives
loose in grpo/.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Few-shot preamble: shows the tag format so even a small base model emits tags.
_PREAMBLE = (
    "Answer each question. Put reasoning in <think></think> and the final number "
    "in <answer></answer>.\n"
    "Q: What is 2 + 3?\n<think>2 plus 3 is 5.</think><answer>5</answer>\n"
)


class RealPolicy:
    def __init__(self, model_name="distilgpt2", device=None, learning_rate=1e-5,
                 max_new_tokens=40, temperature=1.0, top_k=50):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading real model '{model_name}' on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._cache = None  # (sequences, attention_mask, prompt_len) from last generate()

    def _build_prompt(self, task_prompt):
        return f"{_PREAMBLE}Q: {task_prompt}\n"

    def generate(self, prompt, num_samples, correct_answer=None):
        """Sample `num_samples` completions for one prompt; return the texts.

        The full token sequences are cached so train_step() can reuse them. The
        `correct_answer` argument is accepted for interface parity but ignored --
        a real model has to produce the answer itself.
        """
        self.model.eval()
        enc = self.tokenizer(self._build_prompt(prompt), return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            sequences = self.model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=num_samples,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
            )  # [num_samples, L] including the prompt

        attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        self._cache = (sequences, attention_mask, prompt_len)

        # Decode only the completion part (after the prompt) for scoring.
        return [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                for seq in sequences]

    def train_step(self, advantages):
        """One GRPO policy-gradient update from the last group's advantages.

        loss = -(advantage * sum_of_completion_token_logprobs).mean()
        Outputs with positive advantage are made more likely; negative, less likely.
        """
        sequences, attention_mask, prompt_len = self._cache
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)

        self.model.train()
        logits = self.model(input_ids=sequences, attention_mask=attention_mask).logits

        # Log-prob the model assigned to each actually-sampled next token.
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = sequences[:, 1:]
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [M, L-1]

        # Keep only completion tokens (skip the prompt) and skip padding.
        positions = torch.arange(sequences.size(1) - 1, device=self.device)
        completion_mask = (positions >= prompt_len - 1).unsqueeze(0) & (attention_mask[:, 1:] == 1)
        seq_logp = (token_logp * completion_mask).sum(dim=1)             # [M]

        loss = -(advantages * seq_logp).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
