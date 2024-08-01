"""Unit tests for transformer implementation."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans, integers

import src.transformer as transformer


@settings(max_examples=1)
@given(
    x=arrays(np.float32, (10, 16)),
    autoregressive=booleans(),
)
def test_self_attention(x, autoregressive):
    """Sanity check implementation of self attention."""
    x_t = torch.from_numpy(x)
    a = transformer.SelfAttention(input_dim=16, query_key_dim=4, value_dim=8)
    assert a(x_t, autoregressive).size() == (10, 8)


@settings(max_examples=1)
@given(
    x=arrays(np.float32, (10, 16)),
    autoregressive=booleans(),
)
def test_multihead_attention(x, autoregressive):
    """Sanity check implementation of multiheaded attention."""
    m = transformer.MultiHeadAttention(
        num_heads=5, input_dim=16, query_key_dim=4, value_dim=8
    )
    x_t = torch.from_numpy(x)
    assert m(x_t, autoregressive).size() == x_t.size()


@settings(max_examples=1)
@given(
    x=arrays(np.float32, (10, 16)),
    autoregressive=booleans(),
)
def test_transformer_block(x, autoregressive):
    """Sanity check implementation of transformer block."""
    b = transformer.TransformerBlock(
        num_heads=5,
        input_dim=16,
        query_key_dim=4,
        value_dim=8,
        feedforward_dim=32,
    )
    x_t = torch.from_numpy(x)
    assert b(x_t, autoregressive).size() == x_t.size()


@settings(max_examples=1)
@given(x=arrays(np.float32, (10,)), position=integers(0, 9999))
def test_positional_encoding(x, position):
    """Sanity check implementation of positional encoding."""
    x_t = torch.from_numpy(x)
    positional_embedding = transformer.positional_encoding(x_t, position)
    assert positional_embedding.size() == x_t.size()
