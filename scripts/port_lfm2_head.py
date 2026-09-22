"""Port a trained head onto the LFM2.5 backbone: encoder swapped, head weights kept.

Cross-backbone experiment, not a shippable model: take every non-encoder parameter
from a shipped ModernBERT checkpoint (head, scorer, type_emb, act_head, temperatures)
and load it unchanged onto the pretrained LFM2.5-Encoder-230M. ModernBERT-large and
LFM2.5 are both 1024-wide, so no projection is involved and the load is strict.

The expectation stated up front: the two encoders were pretrained independently, so
the head reads an unrelated representation space and accuracy should collapse toward
chance. The checkpoint exists to measure exactly that (see
research/scripts/bench_ported_head.py) and is the "before" number for the LFM2
fine-tune that follows.

Usage:
    python scripts/port_lfm2_head.py [--src convaiinnovations/laya-typed-decisions]
                                      [--out out/lfm2-ported-td]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_flash.backbones import lfm2_backbone
from laya_flash.common import DecisionModel

ENCODER_ID = "LiquidAI/LFM2.5-Encoder-230M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="convaiinnovations/laya-typed-decisions",
                    help="shipped ModernBERT checkpoint to take the head from")
    ap.add_argument("--out", default="out/lfm2-ported-td", help="output checkpoint directory")
    args = ap.parse_args()

    src_cfg = json.load(open(hf_hub_download(args.src, "rl_agent_config.json")))
    src_sd = load_file(hf_hub_download(args.src, "model.safetensors"))

    enc = lfm2_backbone(ENCODER_ID, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=src_cfg.get("head_layers", 2),
                          n_act=len(src_cfg.get("act_costs", {})) + 1)

    # Everything that is not the encoder is the head system; port it verbatim.
    head_sd = {k: v for k, v in src_sd.items() if not k.startswith("encoder.")}
    own_head_keys = {k for k in model.state_dict() if not k.startswith("encoder.")}
    missing = own_head_keys - head_sd.keys()
    if missing:
        sys.exit("head keys the destination expects but the source lacks: %s" % sorted(missing))
    for k, v in head_sd.items():
        if tuple(v.shape) != tuple(model.state_dict()[k].shape):
            sys.exit("shape mismatch on %s: src %s vs dst %s"
                     % (k, tuple(v.shape), tuple(model.state_dict()[k].shape)))
    model.load_state_dict(head_sd, strict=False)
    print("[port] %d head tensors ported, %d encoder tensors left pretrained-LFM2"
          % (len(head_sd), len(src_sd) - len(head_sd)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "encoder": ENCODER_ID,
        "head_layers": src_cfg.get("head_layers", 2),
        "max_len": src_cfg.get("max_len", 512),
        "head_max_len": src_cfg.get("head_max_len", 192),
        "max_prefixes": src_cfg.get("max_prefixes", 6),
        "act_costs": src_cfg.get("act_costs", {}),
        "cost_wrong_act": src_cfg.get("cost_wrong_act", 3.0),
        "amp_dtype": src_cfg.get("amp_dtype", "bf16"),
        "model_name": "laya-flash-lfm2-ported-%s" % src_cfg.get("model_name", "head"),
        # Temps are part of the head's readout, so they travel with it; argmax is
        # unaffected either way and the experiment is "the head exactly as shipped".
        "temperature": src_cfg.get("temperature", [1.0, 1.0, 1.0]),
        "temperature_by_options": src_cfg.get("temperature_by_options", {}),
        "training": {"fine_tuned_from_checkpoint": False, "ported_head_from": args.src},
    }
    with open(out / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    from safetensors.torch import save_model

    save_model(model, str(out / "model.safetensors"))

    from transformers import AutoTokenizer

    # Saved unmutated: Agent re-runs prepare_tokenizer on load, so the checkpoint
    # keeps the tokenizer exactly as LiquidAI ships it.
    AutoTokenizer.from_pretrained(ENCODER_ID).save_pretrained(str(out / "tokenizer"))
    # build_model dispatches on the encoder config's model_type; shipping it keeps
    # that resolution offline and independent of the encoder's hub id in cfg.
    model.encoder.config.save_pretrained(str(out / "encoder"))

    params = sum(p.numel() for p in model.parameters())
    print("[port] %s written: %.1fM params" % (out, params / 1e6))
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(out))


if __name__ == "__main__":
    main()
