"""
Autoregressive GPT-style Transformer over discrete image tokens.
Conditions on worker class token (compliant, no_helmet, no_vest)
and autoregressively generates a 16x16 grid of visual tokens (256 tokens total).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=384, n_head=8, max_len=257, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (tril)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fast FlashAttention / SDPA (PyTorch native)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd=384, n_head=8, max_len=257, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_len, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ImageTokenTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=516,       # 512 visual tokens + 4 class tokens
        seq_len=257,          # 1 class token + 256 image tokens
        n_layer=6,
        n_head=8,
        n_embd=384,
        dropout=0.10,
        num_classes=4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.codebook_offset = 512 # Class token IDs are 512, 513, 514, 515

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, seq_len, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # Head predicts vocab tokens
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: (B, T) where T <= seq_len
        device = idx.device
        b, t = idx.size()
        assert t <= self.seq_len, f"Sequence length {t} exceeds max {self.seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # targets: (B, T) with -100 for ignored prefix
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        class_id,
        num_samples=16,
        temperature=1.0,
        top_k=50,
        device="cuda"
    ):
        """
        Autoregressively generate 256 image tokens conditioned on class_id.
        """
        self.eval()
        # Initial token is class token: 512 + class_id
        prefix = torch.full((num_samples, 1), self.codebook_offset + class_id, dtype=torch.long, device=device)

        idx = prefix
        for step in range(256): # generate 256 visual tokens (16x16)
            logits, _ = self.forward(idx)
            # Focus on visual token logits only (0 to 511)
            logits = logits[:, -1, :self.codebook_offset] / max(1e-5, temperature)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        # Return only the generated 256 visual tokens (exclude prefix class token)
        visual_tokens = idx[:, 1:]
        return visual_tokens
