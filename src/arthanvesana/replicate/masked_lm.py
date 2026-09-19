from __future__ import annotations

import random


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "masked_lm requires torch (CPU): python -m pip install "
            "--index-url https://download.pytorch.org/whl/cpu torch"
        ) from exc
    return torch


def build_vocab(seqs: list[list[str]]) -> dict:
    vocab = sorted({sign for seq in seqs for sign in seq})
    if not vocab:
        raise ValueError("masked_lm requires nonempty training sequences")
    table = {"<PAD>": 0, "<MASK>": 1}
    for sign in vocab:
        table[sign] = len(table)
    return table


def _encode(seqs, table):
    return [[table[s] for s in seq] for seq in seqs]


def _masked_batch(torch, batch, table, mask_id, rng):
    inputs = [list(seq) for seq in batch]
    targets = []
    for seq in inputs:
        positions = [i for i in range(len(seq))]
        pos = positions[rng.randrange(len(positions))]
        targets.append((pos, seq[pos]))
        seq[pos] = mask_id
    width = max(len(seq) for seq in inputs)
    padded = [seq + [0] * (width - len(seq)) for seq in inputs]
    return (
        torch.tensor(padded, dtype=torch.long),
        targets,
    )


class TinyEncoder:
    def __init__(self, torch, vocab_size, dim=32, layers=1, heads=2, dropout=0.1, max_len=32):
        self.torch = torch
        self.sign_emb = torch.nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(max_len, dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(dim, vocab_size)
        self.max_len = max_len
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        self.encoder.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self):
        return (
            list(self.sign_emb.parameters()) + list(self.pos_emb.parameters())
            + list(self.encoder.parameters()) + list(self.head.parameters())
        )

    def forward(self, ids):
        torch = self.torch
        pos = torch.arange(ids.size(1)).unsqueeze(0).expand(ids.size(0), -1)
        pos = pos.clamp(max=self.max_len - 1)
        hidden = self.sign_emb(ids) + self.pos_emb(pos)
        hidden = self.dropout(hidden) if self.training else hidden
        key_mask = ids == 0
        return self.head(self.encoder(hidden, src_key_padding_mask=key_mask))

    def __call__(self, ids):
        return self.forward(ids)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def train_masked_lm(
    train_seqs: list[list[str]], valid_seqs: list[list[str]] | None = None,
    dim: int = 32, layers: int = 1, heads: int = 2, dropout: float = 0.1,
    lr: float = 0.003, batch_size: int = 64, max_epochs: int = 60,
    patience: int = 8, seed: int = 0,
) -> dict:
    torch = _torch()
    if dim % heads:
        raise ValueError("dim must be divisible by heads")
    table = build_vocab(train_seqs + list(valid_seqs or []))
    mask_id = table["<MASK>"]
    encoded = _encode(train_seqs, table)
    valid = _encode(valid_seqs, table) if valid_seqs else []
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyEncoder(torch, len(table), dim, layers, heads, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_state = None
    best_valid = float("inf")
    stale = 0
    epochs_run = 0
    for _ in range(max_epochs):
        epochs_run += 1
        order = list(range(len(encoded)))
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            batch = [encoded[i] for i in order[start:start + batch_size]]
            ids, targets = _masked_batch(torch, batch, table, mask_id, rng)
            logits = model(ids)
            rows = torch.arange(len(targets))
            cols = torch.tensor([pos for pos, _ in targets])
            truth = torch.tensor([truth for _, truth in targets])
            loss = loss_fn(logits[rows, cols], truth)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if not valid:
            continue
        model.eval()
        with torch.no_grad():
            ids, targets = _masked_batch(
                torch, valid, table, mask_id, random.Random(seed + 1)
            )
            logits = model(ids)
            rows = torch.arange(len(targets))
            cols = torch.tensor([pos for pos, _ in targets])
            truth = torch.tensor([truth for _, truth in targets])
            valid_loss = float(loss_fn(logits[rows, cols], truth))
        if valid_loss < best_valid - 1e-4:
            best_valid = valid_loss
            best_state = [p.detach().clone() for p in model.parameters()]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        for param, saved in zip(model.parameters(), best_state):
            param.data.copy_(saved)
    model.eval()
    return {
        "model": model, "table": table, "dim": dim, "layers": layers,
        "heads": heads, "dropout": dropout, "lr": lr,
        "epochs_run": epochs_run, "best_valid_loss": best_valid,
        "n_parameters": count_parameters(model),
    }


def masked_lm_ranks(bundle: dict, test_seqs: list[list[str]]) -> list[int | None]:
    torch = _torch()
    model = bundle["model"]
    table = bundle["table"]
    order = sorted(s for s in table if s not in ("<PAD>", "<MASK>"))
    ranks = []
    with torch.no_grad():
        for seq in test_seqs:
            for pos, target in enumerate(seq):
                if target not in table:
                    ranks.append(None)
                    continue
                ids = torch.tensor(
                    [[table[s] if s in table else table["<MASK>"] for s in seq]],
                    dtype=torch.long,
                )
                ids[0, pos] = table["<MASK>"]
                logits = model(ids)[0, pos]
                scored = sorted(
                    order, key=lambda s: (-float(logits[table[s]]), s)
                )
                ranks.append(scored.index(target) + 1)
    return ranks
