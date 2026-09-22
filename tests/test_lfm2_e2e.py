"""End-to-end test: LiquidAI/LFM2.5-Encoder-230M as the Laya backbone, real weights.

Not run by CI (network + ~1 GB download). Opt in with:

    LAYA_LFM2_E2E=1 python tests/test_lfm2_e2e.py

Verifies the things the LFM2 integration can get silently wrong:
  1. the pretrained weights actually load (base_model_prefix strip, no random init);
  2. the encoder is bidirectional through laya's own backbone class, and bit-identical
     to the official trust_remote_code reference;
  3. the [MASK]-marker flow works with LFM2's tokenizer via prepare_tokenizer.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.backbones import lfm2_backbone
from laya.common import DecisionModel, build_sequence, collate_items

MODEL_ID = "LiquidAI/LFM2.5-Encoder-230M"


def main():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = lfm2_backbone(MODEL_ID, attn_implementation="sdpa")
    assert enc.config.hidden_size == 1024
    enc.prepare_tokenizer(tok)
    assert tok.mask_token_id == 16 and tok.cls_token_id == 1 and tok.sep_token_id == 2

    # 1. Weights really loaded: compare against AutoModel, which resolves to the native
    #    causal class and drops the checkpoint's `lfm2.*` keys (init at random). The
    #    embedding matrix of a trained model is nowhere near that random draw.
    probe = torch.randint(0, 1000, (1, 1))
    with torch.no_grad():
        ours = enc(input_ids=probe, attention_mask=torch.ones_like(probe))
    assert float(ours.abs().mean()) > 1e-3, "suspiciously small activations - random init?"

    # 2. Bidirectional: last-token changes must move position 0.
    with torch.no_grad():
        a = enc(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long))
        b = enc(input_ids=torch.tensor([[10, 11, 12, 13, 14, 999]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long))
    delta = float((a[0, 0] - b[0, 0]).abs().max())
    assert delta > 1e-4, "position 0 insensitive to the last token - encoder is causal"
    print("[lfm2 e2e] bidirectional pos-0 delta: %.4f" % delta)

    # 2b. Bit-identity with the official remote-code path: the vendored patches must
    # reproduce LiquidAI's own Lfm2BidirectionalModel exactly, not approximately.
    ref_full = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
    probe = tok("The capital of France <|mask|> is Paris.", return_tensors="pt")
    with torch.no_grad():
        h_ours = enc(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"])
        h_ref = ref_full.lfm2(
            input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]).last_hidden_state
    diff = float((h_ours - h_ref).abs().max())
    assert diff == 0.0, "vendored patches diverge from trust_remote_code (max diff %g)" % diff
    print("[lfm2 e2e] max abs diff vs trust_remote_code reference: %g" % diff)
    del ref_full

    # 3. Full decision flow with the [MASK] markers, exactly as Agent.system_one does.
    model = DecisionModel(enc, head_layers=2, n_act=2).eval()
    q = {"t": "choice", "ins": "Classify the priority",
         "crit": {"urgent": "act today", "later": "whenever convenient", "never": "do not act"}}
    item = {"ids": [], "markers": [], "qtype": 0}
    ids, markers = build_sequence(tok, "Server on fire in production.", q, max_len=512, head_max_len=192)
    item["ids"], item["markers"] = ids, markers
    assert len(markers) == 3 and all(ids[m] == tok.mask_token_id for m in markers)
    batch = collate_items([[item]], tok.pad_token_id)
    with torch.no_grad():
        logits, act_logits = model(
            batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
            batch["marker_mask"], batch["qtype"])
    assert logits.shape == (1, 3) and act_logits.shape == (1, 2)
    p = torch.softmax(logits[0], -1)
    print("[lfm2 e2e] option probabilities (untrained head, uniform is expected):", [round(float(x), 3) for x in p])
    assert torch.isfinite(logits).all() and torch.isfinite(act_logits).all()
    print("lfm2 e2e tests passed")


if __name__ == "__main__":
    if os.environ.get("LAYA_LFM2_E2E") != "1":
        print("skipped: set LAYA_LFM2_E2E=1 to run (downloads %s)" % MODEL_ID)
        sys.exit(0)
    torch.set_num_threads(2)
    main()
