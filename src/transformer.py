"""Implementation of the transformer model from scratch."""

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Implementation of scaled dot product self-attention."""

    def __init__(self, input_dim: int, query_key_dim: int, value_dim: int) -> None:
        """Initialize the self attention module.

        :param input_dim: dimensionality of input embeddings
        :param query_key_dim: dimensionality of query and key embeddings.
        :param value_dim: dimensionality of value embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.w_q = nn.Parameter(torch.randn(input_dim, query_key_dim))
        self.w_k = nn.Parameter(torch.randn(input_dim, query_key_dim))
        self.w_v = nn.Parameter(torch.randn(input_dim, value_dim))

    def forward(
        self,
        x: torch.Tensor,
        autoregressive: bool = True,
    ) -> torch.Tensor:
        """Run the forward computation to return self attention embeddings.

        :param x: tensor containing the embeddings of input tokens.
        Shape is (N, input_dim).
        :return: Self attention based contextual embeddings of the input.
        """
        query_embeddings = torch.matmul(x, self.w_q)
        key_embeddings = torch.matmul(x, self.w_k)
        value_embeddings = torch.matmul(x, self.w_v)
        attention_score = torch.matmul(query_embeddings, key_embeddings.t())
        attention_score = attention_score / math.sqrt(self.query_key_dim)

        if autoregressive:
            # mask future tokens for autoregressive modeling
            mask = torch.ones_like(attention_score).tril().logical_not()
            attention_score[mask] = -float("inf")
        attention_probabilities = torch.softmax(attention_score, dim=-1)
        return torch.matmul(attention_probabilities, value_embeddings)


class MultiHeadAttention(nn.Module):
    """Implementation of multi-headed attention."""

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        query_key_dim: int,
        value_dim: int,
    ) -> None:
        """Initialize the multi-headed attention module.

        :param num_heads: number of self attention heads.
        :param input_dim: dimensionality of input embeddings.
        :param query_key_dim: dimensionality of query and key embeddings.
        :param value_dim: dimensionality of value embeddings.
        """
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.heads = [
            SelfAttention(input_dim, query_key_dim, value_dim) for _ in range(num_heads)
        ]
        self.w_o = nn.Parameter(torch.randn(num_heads * value_dim, input_dim))

    def forward(
        self,
        x: torch.Tensor,
        autoregressive: bool = True,
    ) -> torch.Tensor:
        """Run the forward computation to return multiheaded attention embeddings.

        :param x: tensor containing the embeddings of input tokens.
        Shape is (N, input_dim).
        :return: Multiheaded attention based contextual embeddings of the input.
        """
        self_attention_embeddings = [head(x, autoregressive) for head in self.heads]
        return torch.matmul(torch.cat(self_attention_embeddings, dim=1), self.w_o)


class FeedForward(nn.Module):
    """Implementation of feedforward layer inside transformer block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the feedforward layer module.

        :param input_dim: dimension of the input to feedforward layer.
        :param hidden_dim: dimension of the hidden layer.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, input_dim)
        self.relu_2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward computation of the feedforward network.

        :param x: tensor containing the embeddings of input tokens.
        Shape is (N, input_dim).
        """
        out = self.linear_1(x)
        out = self.relu_1(out)
        out = self.linear_2(out)
        out = self.relu_2(out)
        return out


class TransformerBlock(nn.Module):
    """Implementation of the transformer block."""

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        query_key_dim: int,
        value_dim: int,
        feedforward_dim: int,
    ) -> None:
        """Initialize the transformer block module.

        :param num_heads: number of self attention heads.
        :param input_dim: dimensionality of input embeddings.
        :param query_key_dim: dimensionality of query and key embeddings.
        :param value_dim: dimensionality of value embeddings.
        :param feedforward_dim: dimensionality of the hidden layer of
        feedforward network.
        """
        super().__init__()
        self.multiheaded_attention = MultiHeadAttention(
            num_heads,
            input_dim,
            query_key_dim,
            value_dim,
        )
        self.layernorm_1 = nn.LayerNorm(input_dim)
        self.feed_forward = FeedForward(input_dim, feedforward_dim)
        self.layernorm_2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        autoregressive: bool = True,
    ) -> torch.Tensor:
        """Run the forward computation of the transformer block.

        :param x: tensor containing the embeddings of input tokens.
        Shape is (N, input_dim).
        :return: embeddings computed by the transformer block.
        """
        # multiheaded attention and residual connection followed by normalization
        out = self.layernorm_1(x + self.multiheaded_attention(x, autoregressive))

        # feedforward layer and residual connection followed by normalization
        out = self.layernorm_2(out + self.feed_forward(out))

        return out


def positional_encoding(token_embedding: torch.Tensor, position: int) -> torch.Tensor:
    """Compute the positional encoding as described in the Transformers paper.

    :param token_embedding: 1d tensor containing embedding of the token.
    :param position: zero based position of the token.
    :return: a positional encoding for the token
    """
    dimensions = token_embedding.size(0)
    idx_range = torch.arange(dimensions, device=token_embedding.device)
    halved_idxs = torch.div(idx_range, 2, rounding_mode="trunc")
    even_indices = torch.fmod(idx_range, 2).bool().logical_not()
    scaling_factor = torch.pow(10000, 2 * halved_idxs / dimensions)
    positional_embedding = torch.where(
        even_indices,
        torch.sin(position / scaling_factor),
        torch.cos(position / scaling_factor),
    )
    return positional_embedding
