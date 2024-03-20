import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, block_size : int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, x):
        x = self.embedding(x)
        return self.dropout(x)


class PTLayerNormalization(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x)


class FeedForwardBlock(nn.Module):
    """a simple linear layer followed by a non-linearity """
    def  __init__(self, d_model, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h : int, dropout: float, block_size: int):  # h = n_heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.block_size = block_size
        assert d_model % h == 0, "d_model is not divisable by n_heads"

        self.d_k = d_model // h # d_k = head size
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv
        self.w_o = nn.Linear(d_model, d_model) # wo
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, tril, block_size):
        d_k = query.shape[-1]

        # (Batch, h, block_size, d_k) --> (Batch, h, block_size, block_size)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(tril[:block_size, :block_size] == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1) # Batch, h, block_size, block_size
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (Batch, h, block_size, block_size) --> (Batch, h, block_size, d_k)
        return (attention_scores @ value), attention_scores



    def forward(self, x, mask):
        query = self.w_q(x) # q = (Batch, block_size, d_model), w_q = (Batch, d_model, d_model), query = (Batch, block_size, d_model)
        key = self.w_k(x) # k = (Batch, block_size, d_model), w_k = (Batch, d_model, d_model), key = (Batch, block_size, d_model)
        value = self.w_v(x) # v = (Batch, block_size, d_model), w_v = (Batch, d_model, d_model), value = (Batch, block_size, d_model)

        # (Batch, block_size, d_model) --> (Batch, block_size, h, d_k) --> (Batch, h, block_size, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout, self.tril, self.block_size)

        # (Batch, h, block_size, d_k) --> (Batch, block_size, h, d_k) --> (Batch, block_size, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, block_size, d_model) --> (Batch, block_size, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = PTLayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):

    def __init__(self, d_model, h, block_size, dropout: float):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, h, dropout, block_size)
        self.feed_forward_block = FeedForwardBlock(d_model, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: int, d_model: int, h: int, block_size: int, dropout: float):
        super().__init__()
        self.layers = layers
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, block_size, dropout) for _ in range(layers)])
        self.norm = PTLayerNormalization()

    def forward(self, x, mask):
        for layer in self.decoder_blocks:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch, block_size, d_model --> batch, block_size, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)



class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, src_embed: InputEmbeddings, src_pos: LearnedPositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def decode(self, x, mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.decoder(x, mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, src_block_size: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1):

    #create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)

    # create the prositional encoding layers

    src_pos = LearnedPositionalEncoding(d_model, src_block_size, dropout)


    # Create the decoder
    decoder = Decoder(N, d_model, h, src_block_size, dropout)

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, src_block_size)

    # Create the transformer
    transformer = Transformer(decoder, src_embed, src_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer