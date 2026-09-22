"""Typed-decisions eval of a head-ported LFM2.5 checkpoint -- the encoder-swap control.

Same questions and scoring as bench_local.py part B (LocalLLaMA/typed-decisions test
split), run on one checkpoint: a trained ModernBERT head loaded verbatim onto the
pretrained LFM2.5-Encoder-230M. The point is the comparison -- if accuracy collapses
to chance while the same head scores 0.766 on its own encoder, the capability was
never in the head alone:

  0.766   the same head on its own ModernBERT encoder (measured, research/results)
  0.461   per-question majority class
  0.318   random guess

  USE_TF=0 python3 research/scripts/bench_ported_head.py [--model out/lfm2-ported-td] [--limit N]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "laya_flash"))
sys.path.insert(0, os.path.join(REPO, "research", "scripts"))

import laya_flash  # noqa: E402
from bench_local import parse_typed_decisions_rows, typed_decisions_metrics  # noqa: E402

OUT = os.path.join(REPO, "research", "results", "lfm2_ported_head_typed_decisions.json")
REFERENCE = {
    "same_head_own_encoder": {"accuracy": 0.766,
                              "source": "measured, research/results (t4 colab + app benchmarks)"},
    "per_question_majority_class": {"accuracy": 0.4610},
    "random_guess": {"accuracy": 0.3175},
    "note": "reference points from bench_local.py part B, identical questions and scoring",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(REPO, "out", "lfm2-ported-td"))
    ap.add_argument("--limit", type=int, default=0, help="cap cases (0 = all 400)")
    a = ap.parse_args()

    from datasets import load_dataset

    d = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    cases, gold, wfs = parse_typed_decisions_rows(d)
    if a.limit:
        cases, gold, wfs = cases[:a.limit], gold[:a.limit], wfs[:a.limit]
    print("loaded %d cases from LocalLLaMA/typed-decisions test" % len(cases), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ag = laya_flash.load(a.model, device=device)
    ag.model.eval()
    print("scoring %s on %s ..." % (a.model, device), flush=True)
    m = typed_decisions_metrics(ag, cases, gold, wfs, tag="ported-head")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device,
                        "torch": torch.__version__, "laya": laya_flash.__version__,
                        "model": a.model, "n_cases": len(cases)},
               "reference_points": REFERENCE, "metrics": m}
    json.dump(results, open(OUT, "w"), indent=2)

    print("\nported head on LFM2.5 encoder:")
    print("   acc %.4f | soft %.4f | brier(soft) %s | ECE %.4f | MAE %s | %.0f ms/case"
          % (m["accuracy"], m["soft_accuracy"] or 0, m["brier_vs_soft"], m["ece"],
             m["score_mae"], m["ms_per_case"]), flush=True)
    print("references: same head + own encoder 0.766 | majority 0.461 | random 0.318")
    for wf, v in m["by_workflow"].items():
        print("      %-28s acc %.3f (n=%d)" % (wf, v["accuracy"], v["n"]), flush=True)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
