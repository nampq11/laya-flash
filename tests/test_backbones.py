"""Backbone abstraction tests (laya_flash/backbones.py): offline, tiny random-weight models.

Run: python tests/test_backbones.py

The LFM2 cases need transformers >= 4.55 (the `laya-flash[lfm2]` extra); on older installs
they are skipped so the file stays green at the package's base 4.48 floor.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModel, BertConfig, BertModel, PreTrainedTokenizerFast

from laya_flash.backbones import LayaFlashBackbone, as_backbone, backbone_hidden_size, hidden_states_of, lfm2_backbone
from laya_flash.common import DecisionModel, build_model, build_sequence


def _lfm2_available() -> bool:
    try:
        import transformers.models.lfm2  # noqa: F401

        return True
    except ImportError:
        return False


def _tiny_bert_config() -> BertConfig:
    return BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=1, intermediate_size=32)


def _tiny_bert() -> BertModel:
    return AutoModel.from_config(_tiny_bert_config())


def _tiny_lfm2_config():
    from transformers.models.lfm2 import Lfm2Config

    # num_key_value_heads must match num_attention_heads or GQA shapes break at this width.
    return Lfm2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=256,
        layer_types=["conv", "full_attention"],
    )


def _lfm2_style_tokenizer():
    """WordLevel tokenizer with LFM2's conventions: mask/pad present, no CLS/SEP."""
    vocab = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|startoftext|>": 2,
        "<|endoftext|>": 3,
        "<|mask|>": 4,
        "option": 5,
        "urgent": 6,
        "later": 7,
        "pick": 8,
        "one": 9,
    }
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<|unk|>")),
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        mask_token="<|mask|>",
    )
    return tok, vocab


def test_as_backbone_preserves_state_dict_keys():
    torch.manual_seed(0)
    enc = _tiny_bert().eval()
    keys_before = set(enc.state_dict())
    bb = as_backbone(enc)
    assert isinstance(bb, LayaFlashBackbone)
    assert set(bb.state_dict()) == keys_before  # no wrapper level, no projection
    ids = torch.randint(0, 50, (2, 6))
    am = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        want = hidden_states_of(BertModel.forward(enc, input_ids=ids, attention_mask=am))
        got = bb(input_ids=ids, attention_mask=am)
    assert torch.is_tensor(got)
    assert torch.allclose(got, want)
    assert bb.hidden_size == 16


def test_as_backbone_projection():
    torch.manual_seed(1)
    enc = _tiny_bert().eval()
    bb = as_backbone(enc, head_dim=8)
    assert "head_proj.weight" in bb.state_dict()
    assert bb.hidden_size == 8
    ids = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        out = bb(input_ids=ids, attention_mask=torch.ones(2, 6, dtype=torch.long))
    assert tuple(out.shape) == (2, 6, 8)

    # A head_dim equal to the native width must not attach a projection: published
    # checkpoints have no head_proj weights, so the parameter set has to stay identical.
    same = as_backbone(_tiny_bert(), head_dim=16)
    assert "head_proj.weight" not in same.state_dict() and same.hidden_size == 16


def test_lfm2_backbone_from_pretrained_path():
    # Loading by id/path is the hub branch of lfm2_backbone; head_dim is attached
    # after construction, so the projection must survive that branch too.
    if not _lfm2_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        # save_pretrained comes from the PretrainedModel base the dynamic class mixes in,
        # which the LayaFlashBackbone return type cannot express (nn.Module __getattr__).
        lfm2_backbone(_tiny_lfm2_config()).save_pretrained(tmp)  # pyright: ignore[reportCallIssue]
        bb = lfm2_backbone(tmp, head_dim=16)
        assert isinstance(bb, LayaFlashBackbone) and bb.hidden_size == 16


def test_lfm2_backbone_forward_shapes():
    torch.manual_seed(2)
    bb = lfm2_backbone(_tiny_lfm2_config())
    assert isinstance(bb, LayaFlashBackbone)
    ids = torch.randint(0, 64, (2, 7))
    am = torch.ones(2, 7, dtype=torch.long)
    with torch.no_grad():
        out = bb(input_ids=ids, attention_mask=am)
    assert torch.is_tensor(out) and tuple(out.shape) == (2, 7, 32)
    assert bb.hidden_size == 32 and "head_proj.weight" not in bb.state_dict()

    proj = lfm2_backbone(_tiny_lfm2_config(), head_dim=16)
    with torch.no_grad():
        out = proj(input_ids=ids, attention_mask=am)
    assert tuple(out.shape) == (2, 7, 16) and proj.hidden_size == 16


def test_lfm2_backbone_is_bidirectional():
    # Under causal attention a change to the LAST token cannot move position 0;
    # the conv and attention patches must both be live for it to move.
    torch.manual_seed(3)
    bb = lfm2_backbone(_tiny_lfm2_config()).eval()
    am = torch.ones(1, 7, dtype=torch.long)
    with torch.no_grad():
        h1 = bb(input_ids=torch.tensor([[5, 6, 7, 8, 9, 10, 11]]), attention_mask=am)
        h2 = bb(input_ids=torch.tensor([[5, 6, 7, 8, 9, 10, 63]]), attention_mask=am)
    assert float((h1[0, 0] - h2[0, 0]).abs().max()) > 1e-4


def test_lfm2_prepare_tokenizer_maps_cls_and_sep():
    tok, vocab = _lfm2_style_tokenizer()
    assert tok.cls_token_id is None and tok.sep_token_id is None  # the LFM2 situation
    size_before = len(tok)
    lfm2_backbone(_tiny_lfm2_config()).prepare_tokenizer(tok)
    assert tok.cls_token_id == vocab["<|startoftext|>"]
    assert tok.sep_token_id == vocab["<|endoftext|>"]
    assert tok.mask_token_id == vocab["<|mask|>"]
    assert len(tok) == size_before  # mapping must not grow the vocab (embedding matrix!)

    # The full build_sequence contract round-trips: one marker per option, in order.
    q = {"t": "choice", "ins": "Pick one", "crit": {"urgent": "act now", "later": "whenever"}}
    ids, markers = build_sequence(tok, "the state", q, max_len=64, head_max_len=32)
    assert ids[0] == tok.cls_token_id
    assert len(markers) == 2
    assert all(ids[m] == tok.mask_token_id for m in markers)
    assert tok.sep_token_id in ids


def test_decision_model_with_backbone():
    torch.manual_seed(4)
    for encoder, d in (
        (as_backbone(_tiny_bert(), head_dim=8), 8),
        (lfm2_backbone(_tiny_lfm2_config(), head_dim=16), 16),
    ):
        model = DecisionModel(encoder, head_layers=1, n_act=2).eval()
        assert model.scorer[1].out_features == d  # head sized from the post-projection width
        n, L, K = 2, 8, 3
        out = model(
            torch.randint(0, 50, (n, L)),
            torch.ones(n, L, dtype=torch.long),
            torch.arange(K).unsqueeze(0).expand(n, -1).contiguous(),
            torch.ones(n, K, dtype=torch.bool),
            torch.zeros(n, dtype=torch.long),
        )
        assert out[0].shape == (n, K) and out[1].shape == (n, 2)
        assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()


def test_build_model_dispatches_on_model_type():
    # LFM2 is detected from the saved encoder config's model_type, not its name:
    # AutoModel would build the causal native Lfm2Model and drop the pretrained weights.
    if not _lfm2_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        enc_dir = Path(tmp) / "encoder"
        enc_dir.mkdir()
        # Save the model (not just its config): the no-encoder_dir case below loads it
        # back through from_pretrained on the path.
        lfm2_backbone(_tiny_lfm2_config()).save_pretrained(enc_dir)  # pyright: ignore[reportCallIssue]
        cfg = {
            "encoder": str(enc_dir),
            "head_layers": 1,
            "act_costs": {"act": 0},
            "head_dim": 16,
            "max_len": 64,
            "head_max_len": 32,
        }
        model = build_model(cfg, encoder_dir=str(enc_dir))
        assert isinstance(model.encoder, LayaFlashBackbone)
        assert model.encoder.config.model_type == "lfm2"  # routed to the LFM2 class, not AutoModel
        assert model.encoder.hidden_size == 16

        # Same checkpoint addressed as the encoder id (no local encoder/ dir): the
        # config is resolved from the path itself, so dispatch still sees model_type.
        model_id = build_model(dict(cfg))
        assert model_id.encoder.config.model_type == "lfm2"
        assert model_id.encoder.hidden_size == 16

        bert_dir = Path(tmp) / "encoder-bert"
        bert_dir.mkdir()
        _tiny_bert_config().save_pretrained(bert_dir)
        cfg_bert = {"encoder": str(bert_dir), "head_layers": 1, "act_costs": {"act": 0}}
        model_bert = build_model(cfg_bert, encoder_dir=str(bert_dir))
        assert isinstance(model_bert.encoder, LayaFlashBackbone)
        assert model_bert.encoder.hidden_size == 16


def test_helpers():
    from transformers.modeling_outputs import BaseModelOutput

    enc = _tiny_bert()
    assert backbone_hidden_size(enc) == 16
    assert backbone_hidden_size(as_backbone(enc, head_dim=8)) == 8
    out = BaseModelOutput(last_hidden_state=torch.zeros(1, 2, 3))  # pyright: ignore[reportArgumentType]
    assert hidden_states_of(out) is out.last_hidden_state
    t = torch.zeros(1)
    assert hidden_states_of(t) is t


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_as_backbone_preserves_state_dict_keys()
    test_as_backbone_projection()
    if _lfm2_available():
        test_lfm2_backbone_from_pretrained_path()
        test_lfm2_backbone_forward_shapes()
        test_lfm2_backbone_is_bidirectional()
        test_lfm2_prepare_tokenizer_maps_cls_and_sep()
        test_decision_model_with_backbone()
        test_build_model_dispatches_on_model_type()
    else:
        print("skipped LFM2 backbone tests: transformers >= 4.55 not installed (pip install 'laya-flash[lfm2]')")
    test_helpers()
    print("all backbone tests passed")
