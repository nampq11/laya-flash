---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-Encoder-230M
tags:
- laya-flash
- lfm2
- decision-model
- baseline
---

# laya-flash (LFM2.5 baseline)

**This is a format baseline, not a usable model yet.** It publishes the
[LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M)
backbone inside a [laya-flash](https://github.com/nampq11/laya-flash) checkpoint, with
the decision head **freshly initialized and untrained**. Expect near-uniform
probabilities and near-zero confidence on every question — that is the honest behaviour
of this checkpoint, not a bug. Fine-tuned versions will replace these files in place.

## What is in the checkpoint

| Part | State |
| --- | --- |
| Encoder (LFM2.5-Encoder-230M, bidirectional patches) | LiquidAI pretrained weights, unchanged |
| Decision head (transformer layers, scorer, act head) | Random initialization (seed 42) |
| Temperatures | Neutral (1.0 — nothing was calibrated) |

The config (`rl_agent_config.json`) mirrors the shipped laya-flash checkpoints
(`act_costs`, `cost_wrong_act`, `max_prefixes`, `amp_dtype`), so a fine-tune can start
from this checkpoint with the head shapes it expects.

## Usage

```python
pip install 'laya-flash[lfm2]'   # LFM2 needs transformers >= 4.55

import laya_flash
agent = laya_flash.Agent("nampham1106/laya-flash")
res = agent.system_one("Customer was charged twice and wants money back.", {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "money back", "tech": "a bug", "other": "anything else"}},
})
```

The checkpoint layout is the standard laya-flash one: `rl_agent_config.json`,
`model.safetensors` (full state dict, fp32), `tokenizer/`, `encoder/config.json`.

## License and attribution

The encoder weights are Liquid AI, Inc.'s, distributed under the **LFM Open License
v1.0** (see `LICENSE`). This checkpoint is a Derivative Work and ships under the same
license. In short: commercial use is licensed below $10M annual revenue; redistribution
must include the license and attribution. laya-flash's bidirectional LFM2 patches are
adapted from LiquidAI's Apache-2.0 `modeling_lfm2_bidirectional.py`.
