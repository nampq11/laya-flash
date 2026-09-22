"""Build the laya-flash LFM2.5 baseline checkpoint: pretrained encoder, untrained head.

The baseline exists to publish the LFM2.5-Encoder-230M backbone in the exact
checkpoint format Agent() loads, before any fine-tuning has happened. The encoder
weights are LiquidAI's pretrained ones; every head parameter (type_emb, transformer
head, scorer, act_head) is freshly initialized from --seed, so outputs are near-uniform
by construction. The first real fine-tune overwrites this checkpoint in place.

Config keys mirror the shipped ModernBERT checkpoints (act_costs, cost_wrong_act,
max_prefixes, amp_dtype) so head shapes match what the training pipeline expects;
temperatures stay at 1.0 because nothing was calibrated.

Usage:
    python scripts/build_lfm2_baseline.py [--out out/lfm2-baseline] [--seed 42]

Publish (after `hf auth login`), renaming the model card and license to repo-root names:
    hf upload nampham1106/laya-flash out/lfm2-baseline .
    hf upload nampham1106/laya-flash scripts/hf_README.md README.md
    hf upload nampham1106/laya-flash scripts/hf_LICENSE LICENSE
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_flash.backbones import lfm2_backbone
from laya_flash.common import DecisionModel

MODEL_ID = "LiquidAI/LFM2.5-Encoder-230M"

# Mirrored from the shipped checkpoints (convaiinnovations/laya rl_agent_config.json):
# the head shapes a fine-tune will continue from, so the baseline is drop-in for it.
ACT_COSTS = {"escalate": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/lfm2-baseline", help="output checkpoint directory")
    ap.add_argument("--seed", type=int, default=42, help="seed for the head initialization")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    enc = lfm2_backbone(MODEL_ID, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=2, n_act=len(ACT_COSTS) + 1)

    cfg = {
        "encoder": MODEL_ID,
        "head_layers": 2,
        "max_len": 512,
        "head_max_len": 192,
        "max_prefixes": 6,
        "act_costs": ACT_COSTS,
        "cost_wrong_act": 3.0,
        "amp_dtype": "bf16",
        "model_name": "laya-flash-lfm2-baseline",
        # Nothing was calibrated, so temperatures stay neutral; copying fitted ones
        # from a trained checkpoint would misreport confidence for random logits.
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
        "training": {"fine_tuned_from_checkpoint": False, "seed": args.seed},
    }
    with open(out / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    from safetensors.torch import save_model

    save_model(model, str(out / "model.safetensors"))

    from transformers import AutoTokenizer

    # Saved unmutated: Agent re-runs prepare_tokenizer on load, so the checkpoint
    # keeps the tokenizer exactly as LiquidAI ships it.
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(str(out / "tokenizer"))
    # build_model dispatches on the encoder config's model_type; shipping it keeps
    # that resolution offline and independent of the encoder's hub id in cfg.
    model.encoder.config.save_pretrained(str(out / "encoder"))

    params = sum(p.numel() for p in model.parameters())
    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"[lfm2-baseline] {params / 1e6:.1f}M params, model.safetensors {size_mb:.0f} MB")
    print(f"[lfm2-baseline] checkpoint written to {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(out))


if __name__ == "__main__":
    main()
