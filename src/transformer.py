"""Implementation of the transformer model from scratch."""

from typing import List

import torch
import math


def self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    backward: bool = True,
) -> torch.Tensor:
    """Compute the self-attention based representation/embeddings of input.

    :param x: input to the model. Shape is (N, D) where N is the
    number of tokens and D is the dimensionality of input embeddings.

    :param w_q: matrix that projects input into query embeddings.
    Shape is (D, D_q).

    :param w_k: matrix that projects input into key embeddings.
    Shape is (D, D_q).

    :param w_v: matrix that projects input into value embeddings.
    Shape is (D, D_v).

    :return: A matrix of self-attention based embeddings of the input having
    the shape (N, D_v).
    """
    query_embeddings = torch.matmul(x, w_q)
    key_embeddings = torch.matmul(x, w_k)
    value_embeddings = torch.matmul(x, w_v)
    d = query_embeddings.size(-1)
    attention_score = torch.matmul(
        query_embeddings, key_embeddings.t()
    ) / math.sqrt(d)

    if backward:
        # mask future tokens for autoregressive modeling
        mask = torch.ones_like(attention_score).tril().logical_not()
        attention_score[mask] = -float("inf")

    attention_probabilities = torch.softmax(attention_score, dim=-1)
    return torch.matmul(attention_probabilities, value_embeddings)


def multihead_attention(
    x: torch.Tensor,
    w_qs: List[torch.Tensor],
    w_ks: List[torch.Tensor],
    w_vs: List[torch.Tensor],
    w_o: torch.Tensor,
    backward: bool = True,
) -> torch.Tensor:
    """Compute multiheaded attention based representation of the input.

    :param x: input to the model. Shape is (N, D) where N is the
    number of tokens and D is the dimensionality of input embeddings.

    :param w_qs: list of query embedding projection matrices. Shape of
    each query embedding projection matrix is (N, D_q).

    :param w_ks: list of key embedding projection matrices. Shape of each
    key embedding projection matrix is (N, D_q).

    :param w_vs: list of value embedding projection matrices. Shape of each
    value embedding projection matrix is (N, D_v).

    :param w_o: matrix that projects the concatenated output of multiple heads
    to embeddings having the same dimensionality as input. Shape is
    (len(w_qs) * D_v, D).

    :return: A matrix of multiheaded attention based embeddings of the input
    having the shape (N, D).
    """
    attention_embeddings = [
        self_attention(x, w_q, w_k, w_v, backward=backward)
        for (w_q, w_k, w_v) in zip(w_qs, w_ks, w_vs)
    ]
    return torch.matmul(torch.cat(attention_embeddings, dim=1), w_o)
