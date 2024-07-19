"""Implementation of the transformer model from scratch."""


import torch
import math


def attention(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute the self-attention weights for the tokens.

    :param query: tensor of query embeddings shaped (N, D). N is
    the number of tokens and D is the length of each embedding
    vector.


    :param key: tensor of key embeddings shaped (N,D). N is the
    number of tokens and D is the length of each embedding vector.


    :return: Returns the tensor of self-attention weights shaped (N, N).
    """
    d = query.size(-1)
    scaled_weights = torch.matmul(query, key.t()) / math.sqrt(d)
    return torch.softmax(scaled_weights, dim=-1)
