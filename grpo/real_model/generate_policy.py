"""The REAL policy: a torch + transformers instruction-tuned language model.

This is the real-model counterpart to ../test/generate_policy.py (MockPolicy). It
exposes the SAME generate(prompt, num_samples, correct_answer) interface, so the
shared scoring/advantage code is reused unchanged. Two differences from the mock:

    * generate() actually samples completions from a language model (it ignores
      `correct_answer` -- a real model must reason, it isn't handed the answer).
    * train_step() applies one real GRPO policy-gradient update from the group's
      advantages -- the part the simulation can only describe.

Uses Qwen2.5-0.5B-Instruct: small enough to run locally, but already capable of
basic arithmetic and tag-following, so GRPO has correct samples to reinforce.

Stabilizers that stop the policy from collapsing (plain REINFORCE drifts off the
pretrained manifold and forgets how to answer):
    * KL penalty to a FROZEN reference (the initial model) keeps the policy close
      to its competent starting point -- the defining GRPO regularizer.
    * Length-normalized log-probs (mean per completion token, not sum) so the
      update magnitude doesn't scale with completion length.
    * A gentle learning rate.

Libraries: torch (tensors, autograd, optimizer) and transformers (the model +
tokenizer). Everything else (parse_tags, score_rewards, compute_advantages) lives
loose in grpo/.
"""

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SYSTEM = (
    "You solve arithmetic problems. Put your step-by-step reasoning inside "
    "<think></think>, then give ONLY the final integer inside <answer></answer>."
)
# One-shot example (in chat form) so the model reliably emits the tag format.
_EXAMPLE_USER = "What is 2 + 3?"
_EXAMPLE_ASSISTANT = "<think>2 plus 3 is 5.</think><answer>5</answer>"


class RealPolicy:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None,
                 learning_rate=5e-6, kl_coef=0.1,
                 max_new_tokens=64, temperature=1.0, top_k=50):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading real model '{model_name}' on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Frozen reference = the initial policy; the KL leash pulls back toward it.
        self.reference = copy.deepcopy(self.model).to(self.device)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._cache = None  # (sequences, attention_mask, prompt_len) from last generate()

    def _build_prompt(self, task_prompt):
        # Format as a chat with a one-shot example; ask for an assistant turn.
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _EXAMPLE_USER},
            {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
            {"role": "user", "content": task_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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

    def _token_logp(self, model, sequences, attention_mask):
        """Log-prob each model assigns to the actually-sampled next token. [M, L-1]"""
        logits = model(input_ids=sequences, attention_mask=attention_mask).logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = sequences[:, 1:]
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def train_step(self, advantages):
        """One regularized GRPO update from the last group's advantages.

        loss = -(advantage * mean_logp) + kl_coef * KL(policy || reference)
        * mean_logp is the length-normalized completion log-prob (mean per token).
        * KL uses the k3 estimator on the sampled tokens (>= 0), pulling the policy
          back toward the frozen reference so it cannot drift into incoherence.
        """
        sequences, attention_mask, prompt_len = self._cache
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)

        self.model.train()
        policy_logp = self._token_logp(self.model, sequences, attention_mask)        # grad
        with torch.no_grad():
            ref_logp = self._token_logp(self.reference, sequences, attention_mask)

        # Average over completion tokens only (skip prompt + padding).
        positions = torch.arange(sequences.size(1) - 1, device=self.device)
        mask = ((positions >= prompt_len - 1).unsqueeze(0) & (attention_mask[:, 1:] == 1)).float()
        tokens = mask.sum(dim=1).clamp(min=1.0)                                       # [M]

        mean_logp = (policy_logp * mask).sum(dim=1) / tokens                          # [M]

        # k3 KL estimator (Schulman): exp(r) - r - 1 with r = ref_logp - policy_logp.
        log_ratio = ref_logp - policy_logp
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
        mean_kl = (kl_per_token * mask).sum(dim=1) / tokens                           # [M]

        loss = -(advantages * mean_logp).mean() + self.kl_coef * mean_kl.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
