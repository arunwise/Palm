"""Unit tests for transformer implementation."""

import torch
import numpy as np

from hypothesis import given
from hypothesis.strategies import booleans, lists

from hypothesis.extra.numpy import arrays
import src.transformer as transformer


@given(
    x=arrays(np.float32, (5, 20)),
    w_q=arrays(np.float32, (20, 10)),
    w_k=arrays(np.float32, (20, 10)),
    w_v=arrays(np.float32, (20, 15)),
    backward=booleans(),
)
def test_self_attention(x, w_q, w_k, w_v, backward):
    """Sanity check implementation of self attention."""
    a = transformer.self_attention(
        torch.from_numpy(x),
        torch.from_numpy(w_q),
        torch.from_numpy(w_k),
        torch.from_numpy(w_v),
    )
    assert a.size() == (5, 15)


@given(
    x=arrays(np.float32, (5, 20)),
    w_qs=lists(arrays(np.float32, (20, 10)), min_size=5, max_size=5),
    w_ks=lists(arrays(np.float32, (20, 10)), min_size=5, max_size=5),
    w_vs=lists(arrays(np.float32, (20, 15)), min_size=5, max_size=5),
    w_o=arrays(np.float32, (75, 20)),
    backward=booleans(),
)
def test_multihead_attention(x, w_qs, w_ks, w_vs, w_o, backward):
    """Sanity check implementation of multiheaded attention."""
    x_t = torch.from_numpy(x)
    w_qs_ts = [torch.from_numpy(e) for e in w_qs]
    w_ks_ts = [torch.from_numpy(e) for e in w_ks]
    w_vs_ts = [torch.from_numpy(e) for e in w_vs]
    w_o_t = torch.from_numpy(w_o)
    assert x_t.size() == transformer.multihead_attention(
        x_t, w_qs_ts, w_ks_ts, w_vs_ts, w_o_t, backward
    ).size()
