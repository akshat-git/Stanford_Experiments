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
import os
from contextlib import nullcontext

# Reduce CUDA memory fragmentation. Memory-only: does not change results. Must be
# set before torch initializes CUDA, and setdefault leaves any user value intact.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_SYSTEM = (
    "You solve arithmetic expressions (operators + - * / // % ** and parentheses). "
    "Use each number and operator exactly as given -- never invent an operation. "
    "Evaluate parentheses first, then powers (**), then * / // %, then + and - left "
    "to right. Show each calculation explicitly as `a op b = result`, one step at a "
    "time -- do not describe the rules in words. Put the steps inside <think></think>, "
    "then give ONLY the final integer inside <answer></answer>."
)
# Few-shot examples covering the patterns (add/subtract chain; powers + division;
# nested parentheses). Each shows ONLY explicit calculations, no prose, so the model
# imitates that and the thought reward (counting "=") agrees.
_EXAMPLES = [
    ("What is 8 + 2 - 5?",
     "<think>8 + 2 = 10\n10 - 5 = 5</think><answer>5</answer>"),
    ("What is 2 ** 3 + 12 / 4?",
     "<think>2 ** 3 = 8\n12 / 4 = 3\n8 + 3 = 11</think><answer>11</answer>"),
    ("What is 3 * (4 + 2 * (5 - 1))?",
     "<think>5 - 1 = 4\n2 * 4 = 8\n4 + 8 = 12\n3 * 12 = 36</think><answer>36</answer>"),
]


class RealPolicy:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None,
                 learning_rate=3e-6, kl_coef=0.2,
                 max_new_tokens=48, temperature=1.0, top_k=50, micro_batch=4):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading real model '{model_name}' on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Mixed precision when CUDA supports bf16: compute runs in bf16 (faster on
        # tensor cores, ~half the activation memory) while the trained weights stay
        # fp32 so small-LR Adam updates don't underflow.
        self.use_amp = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self.amp_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Frozen reference = the initial policy; the KL leash pulls back toward it.
        # It is inference-only, so we keep it in bf16 to save ~half its memory.
        ref_dtype = self.amp_dtype if self.use_amp else torch.float32
        self.reference = copy.deepcopy(self.model).to(device=self.device, dtype=ref_dtype)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        # The KV cache is unused in the training forward; turning it off saves memory.
        self.model.config.use_cache = False
        self.reference.config.use_cache = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.micro_batch = micro_batch  # cap GPU peak: process this many sequences at a time
        self._cache = None  # (sequences, attention_mask, prompt_len) from last generate()

    def _build_prompt(self, task_prompt):
        # Format as a chat with the few-shot examples; ask for an assistant turn.
        messages = [{"role": "system", "content": _SYSTEM}]
        for example_user, example_assistant in _EXAMPLES:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": example_assistant})
        messages.append({"role": "user", "content": task_prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt, num_samples, correct_answer=None, steps=None):
        """Sample `num_samples` completions for one prompt; return the texts.

        The full token sequences are cached so train_step() can reuse them. The
        `correct_answer` and `steps` arguments are accepted for interface parity but
        ignored -- a real model has to produce the answer and its working itself.
        """
        self.model.eval()
        enc = self.tokenizer(self._build_prompt(prompt), return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        pad_id = self.tokenizer.pad_token_id

        # Sample in micro-batches so the generation KV cache never holds all M at once.
        chunks, remaining = [], num_samples
        with torch.no_grad(), self._autocast():
            while remaining > 0:
                n = min(self.micro_batch, remaining)
                chunks.append(self.model.generate(
                    **enc,
                    do_sample=True,
                    num_return_sequences=n,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    pad_token_id=pad_id,
                    use_cache=True,  # keep decoding fast (config.use_cache is off for training)
                ))
                remaining -= n

        # Pad each chunk to a common length, then stack into [num_samples, L].
        max_len = max(c.size(1) for c in chunks)
        sequences = torch.cat([F.pad(c, (0, max_len - c.size(1)), value=pad_id) for c in chunks], dim=0)
        attention_mask = (sequences != pad_id).long()
        self._cache = (sequences, attention_mask, prompt_len)

        # Decode only the completion part (after the prompt) for scoring.
        return [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                for seq in sequences]

    def _autocast(self):
        """bf16 autocast context when supported, else a no-op."""
        return torch.autocast(self.device.type, dtype=self.amp_dtype) if self.use_amp else nullcontext()

    def _token_logp(self, model, sequences, attention_mask):
        """Log-prob each model assigns to the actually-sampled next token. [M, L-1].

        Uses logsumexp + gather instead of a full log_softmax, so the [M, L, vocab]
        probability tensor is never materialized (a large memory saving with a
        150k-token vocab); mathematically identical to log_softmax then gather.
        """
        logits = model(input_ids=sequences, attention_mask=attention_mask).logits[:, :-1, :]
        targets = sequences[:, 1:]
        chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return chosen - torch.logsumexp(logits, dim=-1)

    def train_step(self, advantages):
        """One regularized GRPO update from the last group's advantages.

        loss = -(advantage * mean_logp) + kl_coef * KL(policy || reference)
        * mean_logp is the length-normalized completion log-prob (mean per token).
        * KL uses the k3 estimator on the sampled tokens (>= 0), pulling the policy
          back toward the frozen reference so it cannot drift into incoherence.
        """
        sequences, attention_mask, prompt_len = self._cache
        adv = torch.tensor(advantages, dtype=torch.float, device=self.device)
        group = sequences.size(0)

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        # Process the group in micro-batches and accumulate gradients. Dividing each
        # chunk's summed loss by the full group size makes the accumulated gradient
        # identical to a single full-batch step -- only the peak memory is lower.
        for start in range(0, group, self.micro_batch):
            sl = slice(start, start + self.micro_batch)
            seq, mask_ids, adv_c = sequences[sl], attention_mask[sl], adv[sl]

            with self._autocast():
                policy_logp = self._token_logp(self.model, seq, mask_ids)            # grad
            policy_logp = policy_logp.float()  # do the loss math in fp32 for stability
            with torch.no_grad():
                ref_logp = self._token_logp(self.reference, seq, mask_ids).float()

            # Average over completion tokens only (skip prompt + padding).
            positions = torch.arange(seq.size(1) - 1, device=self.device)
            mask = ((positions >= prompt_len - 1).unsqueeze(0) & (mask_ids[:, 1:] == 1)).float()
            tokens = mask.sum(dim=1).clamp(min=1.0)
            mean_logp = (policy_logp * mask).sum(dim=1) / tokens

            # k3 KL estimator (Schulman): exp(r) - r - 1 with r = ref_logp - policy_logp.
            log_ratio = ref_logp - policy_logp
            kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
            mean_kl = (kl_per_token * mask).sum(dim=1) / tokens

            loss = (-(adv_c * mean_logp) + self.kl_coef * mean_kl).sum() / group
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        return total_loss
