import torch
import torch.nn as nn
import math
from torch.nn import functional as F
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) #* math.sqrt(self.d_model)


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, block_size : int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        # self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, x, device):
        _, seq,_ = x.shape
        pos_emb = self.embedding(torch.arange(0, seq, dtype=torch.long, device=device))
        x = x + pos_emb
        return x


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
            nn.GELU(),
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(tril, query, key, value, dropout: nn.Dropout, block_size):
        d_k = query.shape[-1]

        # (Batch, h, block_size, d_k) --> (Batch, h, block_size, block_size)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # For encoder we can remove the below mask as we are having only decoder for this transformer we have only decoder in this model
        # tril = torch.tril(torch.ones(block_size, block_size, device='cuda' if torch.cuda.is_available() else 'mps', dtype=torch.bool))
        attention_scores = attention_scores.masked_fill(tril[:, :, :block_size, :block_size] == 0, float('-inf'))
        # Apply padding mask
        # if mask is not None:
        #     attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # Batch, h, block_size, block_size
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # print_msg(str(attention_scores.shape))
        # print_msg(f"value Shape : {value.shape}")
        # (Batch, h, block_size, block_size) --> (Batch, h, block_size, d_k)
        x = attention_scores @ value
        # print_msg(str(x.shape))
        return x, attention_scores



    def forward(self, q, k, v):
        # print_msg(f"step 4 : {str(x.shape)}")
        _, block_size, _ = q.shape
        query = self.w_q(q) # q = (Batch, block_size, d_model), w_q = (Batch, d_model, d_model), query = (Batch, block_size, d_model)
        key = self.w_k(k) # k = (Batch, block_size, d_model), w_k = (Batch, d_model, d_model), key = (Batch, block_size, d_model)
        value = self.w_v(v) # v = (Batch, block_size, d_model), w_v = (Batch, d_model, d_model), value = (Batch, block_size, d_model)
        # print_msg(f"step 5  value: {str(value.shape)}")
        # (Batch, block_size, d_model) --> (Batch, block_size, h, d_k) --> (Batch, h, block_size, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(self.tril, query, key, value, self.dropout, block_size)

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
        return x + sublayer(self.norm(x))


class DecoderBlock(nn.Module):

    def __init__(self, d_model, Decoder_self_attention, Feed_forward_block, dropout: float):
        super().__init__()
        self.self_attention_block = Decoder_self_attention
        self.feed_forward_block = Feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = PTLayerNormalization(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # print_msg(str(self.vocab_size))
        # batch, block_size, d_model --> batch, block_size, vocab_size
        # torch.log_softmax(self.proj(x), dim = -1)
        return self.proj(x)



class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, src_embed: InputEmbeddings, src_pos: LearnedPositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def decode(self, x, label = None):
        # print_msg(f"step 1 : {str(x.shape)}")
        device = x.device
        x = self.src_embed(x)
        # print_msg(f"step 2 : {str(x.shape)}")
        x = self.src_pos(x, device)
        # print_msg(f"step 3 : {str(x.shape)}")
        x = self.decoder(x)

        if label is not None:
            x = self.projection_layer(x)
            loss = F.cross_entropy(x.view(-1, x.size(-1)), label.view(-1), ignore_index=-1)
        else:
            x = self.projection_layer(x[:, [-1], :])
            loss = None

        return x, loss

    @torch.no_grad()
    def generate(self, config, input, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # print(f"Input in generater starting: {input}")
            # input = input[:, -config['seq_len']:]
            # if the sequence context is growing too long we must crop it at block_size
            input_cut = input if input.size(1) <= config['seq_len'] else input[:, -config['seq_len']:]
            # print(f"for loop input : {input}")
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.decode(input_cut)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.softmax(logits, dim=-1)
            # sample from the distribution
            input_next = torch.multinomial(probs, num_samples=1)
            # print(f"Input next generate next word: {input_next}")
            # append sampled index to the running sequence and continue
            input = torch.cat((input, input_next), dim=1)

        print(f"Input in generate after concatenate: {input}")

        return input

    # def project(self, x, label):
    #     # print_msg(f"step 6 : {str(x.shape)}")
    #
    #     if label is not None:
    #         x = self.projection_layer(x)
    #         loss = nn.CrossEntropyLoss(x.view(-1, x.size(-1)), label.view(-1), ignore_index=-1)
    #     else:
    #         x = self.projection_layer(x[:, [-1], :])
    #     return x, loss

def build_transformer(src_vocab_size: int, src_block_size: int, d_model: int = 384, N: int = 6, h: int = 6, dropout: float = 0.0):

    #create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)

    # create the prositional encoding layers

    src_pos = LearnedPositionalEncoding(d_model, src_block_size, dropout)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout, src_block_size)
        feed_forward_block = FeedForwardBlock(d_model, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the and decoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, src_vocab_size)

    # Create the transformer
    transformer = Transformer(decoder, src_embed, src_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer