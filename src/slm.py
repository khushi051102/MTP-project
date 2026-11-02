"""Phase 5: sensor windows -> patch embedding -> (mostly frozen) GPT-2 blocks with LoRA -> small head.

The language model never sees words. We feed it our own vectors through `inputs_embeds`, so
GPT-2's stack of attention blocks is used as a pretrained sequence processor. What is trained:
the patch embedding, LoRA adapters on the attention projections, the LayerNorm gains and the head.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import GPT2Config, GPT2Model

from .config import SEED
from .dataset import split_by_load
from .dl import predict, train_classifier
from .utils import set_seed

D_MODEL = 768


class PatchGPT(nn.Module):
    """Input (batch, length) with `patch_len` set (CWRU windows cut into patches), or
    (batch, tokens, in_dim) (FEMTO: one token per snapshot).

    n_layers      keep only the first n GPT-2 blocks (12 = full GPT-2 small)
    pretrained    False = same architecture with random weights (ablation: does pretraining matter?)
    use_lora      False = frozen GPT-2, only embedding + head (and LayerNorms) train
    task          "cls" -> logits (batch, n_out); "reg" -> one number in (0, 1) per sample
    """

    def __init__(self, in_dim, n_out, n_layers=12, pretrained=True, use_lora=True, lora_r=8,
                 lora_alpha=16, train_ln=True, max_tokens=64, patch_len=None, task="cls"):
        super().__init__()
        gpt = GPT2Model.from_pretrained("gpt2") if pretrained else GPT2Model(GPT2Config())
        gpt.h = nn.ModuleList(list(gpt.h)[:n_layers])
        gpt.config.n_layer = n_layers
        # The word table (50,257 x 768 = 38.6M parameters) is never used when we pass inputs_embeds,
        # and only `max_tokens` positions are needed, so neither is counted in the deployed size.
        gpt.wte = nn.Embedding(1, D_MODEL)
        wpe = nn.Embedding(max_tokens, D_MODEL)
        wpe.weight.data.copy_(gpt.wpe.weight.data[:max_tokens])
        gpt.wpe = wpe
        for p in gpt.parameters():
            p.requires_grad = False
        if use_lora:
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["c_attn"],
                             lora_dropout=0.05, fan_in_fan_out=True, bias="none")
            gpt = get_peft_model(gpt, cfg)      # freezes the base weights, adds trainable low-rank updates
        if train_ln:
            for name, p in gpt.named_parameters():
                if "ln_" in name:                # ln_1, ln_2 in every block and the final ln_f
                    p.requires_grad = True
        self.patch_len = patch_len
        self.task = task
        self.embed = nn.Linear(in_dim, D_MODEL)
        self.lm = gpt
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, n_out))

    def forward(self, x):
        if self.patch_len is not None and x.dim() == 2:
            x = x.reshape(x.shape[0], -1, self.patch_len)        # (batch, tokens, patch_len)
        z = self.embed(x)                                        # (batch, tokens, 768)
        h = self.lm(inputs_embeds=z, use_cache=False).last_hidden_state
        out = self.head(h.mean(dim=1))                           # average over tokens
        return out if self.task == "cls" else torch.sigmoid(out).squeeze(-1)


def count_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_cwru_slm(X, info, n_layers=12, pretrained=True, use_lora=True, epochs=10, lr=5e-4, bs=128,
                 seed=SEED, patch_len=64, verbose=True):
    """Leave-one-load-out, same protocol, scaling and fixed-epoch training as the CNN."""
    y = pd.factorize(info["label"], sort=True)[0]
    rows, model = [], None
    for test_load in range(4):
        tr, _, te = split_by_load(info, test_load, None)
        scale = float(X[tr].std())
        set_seed(seed)                                           # initial weights of embed, head, LoRA
        model = PatchGPT(in_dim=patch_len, n_out=int(y.max()) + 1, n_layers=n_layers,
                         pretrained=pretrained, use_lora=use_lora, patch_len=patch_len)
        if verbose:
            print(f"fold: test load {test_load} HP", flush=True)
        train_classifier(model, X[tr] / scale, y[tr], epochs=epochs, lr=lr, bs=bs, seed=seed, verbose=verbose)
        pred = predict(model, X[te] / scale).argmax(axis=1)
        rows.append({"test_load": test_load, "accuracy": accuracy_score(y[te], pred),
                     "macro_F1": f1_score(y[te], pred, average="macro")})
    return pd.DataFrame(rows), model
