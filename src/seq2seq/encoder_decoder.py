
# This tutorial is from: https://medium.com/correll-lab/building-an-encoder-decoder-for-text-summarization-c66ddf23f466
# It is an encoder-decoder for text summarization
# Kris Doing 8/7/2025

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """ Creates the positional embeddings for encoder-decoder model
    
        This class handles creating the positional embeddings for each element
        in the maximum sequence length sequence of inputs. It extends the nn.Module class.
        
        The class implements the Strategy of creating a vector of the numbers for each position,
        a vector of decreasing values the size of the embedding dimension, and sin and cos 
        vectors. Combined these create positional embeddings, where each position gets a unique 
        pattern of sine/cosine values. It puts the embeddings in a register buffer.
        The forward() method adjusts the size of the embeddings to match the input size.
        
        Attributes:
            register_buffer('pe', pe) (torch array [maximum sequence length x embedding dimension): Positional Embeddings
        
        Example:
            self.positional_embedding = PositionalEmbedding(emb_dim, max_seq_length)
    """
    
    def __init__(self, emb_dim, max_seq_length):
        """ Create fixed sinusoidal embeddings
        :param: emb_dim: embedding dimesion
        :param: max_seq_length: maximum sequence length
        """
        super().__init__()
        pe = torch.zeros(max_seq_length, emb_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # even-index columns
        pe[:, 1::2] = torch.cos(position * div_term)  # odd-index columns
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_val):
        """ Adjust positional embeddings to match the input size
        :param: input_val: input to the model
        :return: input_val: the input vector with the positional embedding added
        """
        input_val = input_val + self.pe[:, :input_val.shape[1], :]  

        return input_val


class AttentionHead(nn.Module):
    """ Creates an attention head module for an encoder-decoder model

        This class handles creating the attention head layers for the encoder-decoder network.
        It creates a 3 linear transform layers with input the size of the embedding dimension
        and output the number of attention heads (head_size). Extends nn.Module
        
        Attributes:
            head_size: the number of attention heads
            quiery: linear transform layer, without bias "what am I looking for?"
            key: linear transform layer, without bias "what do I contain?"
            value: linear transform layer, without bias "what information do I carry?"
            causal: boolean value for whether attention can "look ahead" (False) or use 
            current and previous positional embeddings (True).
        
        Example:
            self.heads = nn.ModuleList([AttentionHead(emb_dim, self.head_size, causal=self.causal) for _ in range(n_heads)])

    """
    def __init__(self, emb_dim, head_size, causal=False):
        """
        :param: emb_dim: embedding dimension
        :param: head_size: number of attention heads
        :param: causal: attention cannot be applied to later positions
        """
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        self.causal = causal

    def forward(self, input_val, kv=None, k_mask=None):
        """ Forward pass in the model
        Q vector has Shape: (B, query_len, head_size)
        K and V have Shape: (B, seq_len, head_size) where seq_len = k_len if cross attention, else q_len
        attention has Shape: (B, q_len, k_len)
        output has Shape: (B, seq_len, head_size)  where seq_len = k_len if cross attention, else q_len
        
        :param: input_val: the input vector coming in to the attention layer
        :param: kv: optional key vector
        :param: k_mask: optional padding mask
        :return: output: the vector of values for the attention heads
        """
        B, T, C = input_val.shape
        Q = self.query(input_val)
        K = self.key(
            kv if kv is not None else input_val)
        V = self.value(
            kv if kv is not None else input_val)  
        
        attention = Q @ K.transpose(-2, -1)

        # Scaling to prevent softmax saturation
        attention = attention / (self.head_size ** 0.5)

        # Applying k padding mask if provided (column-wise mask)
        if k_mask is not None:
            k_mask = k_mask.unsqueeze(1)  # Shape: (B, 1, k_len) can be broadcast with attention
            attention = attention.masked_fill(k_mask == 0, float("-inf"))

        # Applying causal mask for decoder's masked MHA
        if self.causal:
            c_mask = torch.tril(torch.ones(T, T, device=input_val.device))  # is broadcastable with attention (B, T, T)
            attention = attention.masked_fill(c_mask == 0, float("-inf"))

        attention = torch.softmax(attention, dim=-1)

        # Weighted sum of values
        output = attention @ V 

        return output


class MultiHeadAttention(nn.Module):
    """ Creates multiple attention mechanisms to run in parallel
    
        This class creates multiple attention mechanisms, each focusing on different types of 
        relationships or patterns in the data. Extends nn.Module.
        emd_dim % n_heads == 0 must be true. I catch that problem when the config file is read.
    
        Attributes
            head_size: the number of attention heads, must divid evenly into the embedding dimension
            W_o: Linear transform layer, without bias the output projection layer of concatenated attention heads
            causal: boolean value for whether attention can "look ahead" (False) or use 
            heads: the list of attention head modules created with the AttentionHead class
    
        Example:
            self.mha = MultiHeadAttention(emb_dim, n_heads)

    """
    def __init__(self, emb_dim, n_heads, causal=False):
        """
        :param: emb_dim: embedding dimension
        :param: n_heads: the number of attention heads
        :param: causal: attention cannot be applied to later positions
        """
        super().__init__()

        self.head_size = emb_dim // n_heads

        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)

        self.causal = causal

        self.heads = nn.ModuleList([AttentionHead(emb_dim, self.head_size, causal=self.causal) for _ in range(n_heads)])

    def forward(self, x, kv=None, k_mask=None):
        """ Forward pass in the model
        :param: input_val: input vector to the module
        :param: kv: optional key vector
        :param: k_mask: optional padding mask
        :return: out: Combined attention heads
        """
        out = torch.cat([head(x, kv, k_mask=k_mask) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out


class TransformerEncoder(nn.Module):
    """ Creates an Encoder module for an encoder-decoder transformer

        This class creates the encoder side. It puts together a layered structure, which
        Takes input, normalizes it, gets the attention head values, drops out some nodes (sets their 
        activation to zero, normalizes again, feeds that vector through a multi-layer perceptron, 
        and drops out again.

        Attributes
            emb_dim: embedding dimension
            n_heads: number of multi head attention layers
            ln1: first normalization layer the size of the embedding dimension
            mha: MultiHeadAttention module with n_heads
            dropout1: first dropout layer after attention module
            ln2: second normalization layer
            mlp: multi-layer perceptron (algorithm)
            dropout2: second dropout layer after mlp

        Example
            self.encoder = nn.ModuleList(
            [TransformerEncoder(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
            )
    """
    def __init__(self, emb_dim, n_heads, r_mlp=4, dropout=0.1):
        """
        :param: emb_dim: embedding dimension
        :param: n_heads: number of multi head attention layers
        :param: r_mlp: optional multiplier to expand the size of the mlp hidden layers
        :param: dropout: optional % of nodes to drop out
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(emb_dim)

        self.mha = MultiHeadAttention(emb_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * r_mlp),
            nn.GELU(),
            nn.Linear(self.emb_dim * r_mlp, self.emb_dim)
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input_val, src_mask=None):
        """ The forward pass in the model
        :param: input_val: the input to the encoder
        :param: src_mask: optional padding mask
        :return: input_val: the transformed input value
        """
        # Residual Connection After Sub-Layer 1 (MHA)
        input_val = input_val + self.dropout1(self.mha(self.ln1(input_val), k_mask=src_mask))

        # Residual Connection After Sub-Layer 2 (MLP)
        input_val = input_val + self.dropout2(self.mlp(self.ln2(input_val)))

        return input_val


class TransformerDecoder(nn.Module):
    """ Creates a Decoder module for an encoder-decoder transformer

        This class creates the decoder side. It puts together a layered structure, which
        Takes input, normalizes it, gets the attention head values, drops out some nodes (sets
        their activation to zero, normalizes again, gets cross_attention values that refer back
        to the encoder values, drops out again, feeds that vector through a multi-layer perceptron, 
        and drops out again.

        Attributes
            emb_dim: embedding dimension
            n_heads: number of multi head attention layers
            ln1: first normalization layer the size of the embedding dimension
            masked_mha: MultiHeadAttention module with n_heads, where causal=True, no look ahead
            dropout1: first dropout layer after attention module
            ln2: second normalization layer
            cross_attention: second multi-head attention module referring to enconder values
            dropout2: second dropout layer after cross_attention
            ln3: third normalization layer
            mlp: multi-layer perceptron (algorithm)
            dropout3: third dropout layer after mlp

        Example
        
    """
    def __init__(self, emb_dim, n_heads, r_mlp=4, dropout=0.1):
        """
        :param: emb_dim: embedding dimension
        :param: n_heads: number of multi head attention layers
        :param: r_mlp: optional multiplier to expand the size of the mlp hidden layers
        :param: dropout: optional % of nodes to drop out
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(emb_dim)
        self.masked_mha = MultiHeadAttention(emb_dim, n_heads, causal=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.cross_attention = MultiHeadAttention(emb_dim, n_heads)
        self.dropout2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * r_mlp),
            nn.GELU(),
            nn.Linear(self.emb_dim * r_mlp, self.emb_dim)
        )
        self.dropout3 = nn.Dropout(dropout)  # Dropout after MLP

    def forward(self, input_val, encoder_output, src_mask=None, tgt_mask=None):
        """ The forward pass in the model
        :param: input_val: the input to the encoder
        :param: src_mask: optional padding mask for source
        :param: tgt_mask: optional padding mask for target
        :return: input_val: the transformed input value
        """
        input_val = input_val + self.dropout1(self.masked_mha(self.ln1(input_val), k_mask=tgt_mask))
        input_val = input_val + self.dropout2(self.cross_attention(self.ln2(input_val), kv=encoder_output, k_mask=src_mask))
        input_val = input_val + self.dropout3(self.mlp(self.ln3(input_val)))
        return input_val


class TextEncoder(nn.Module):
    """ Creates the TextEncoder module of the encoder-decoder transformer

        This class creates the text encoder side. It puts together a layered structure, which
        takes input, adds the positional embeddings, drops out some nodes (sets their activation
        to zero), and creates an n_layers size TransformerEncoder module.

        Attributes
            encoder_embedding: the vector for the embedded input to the model
            positional_embeddings: positonal embeddings from the class PositionalEmbeddings
            dropout: run a dropout pass
            encoder: n_layers sized encoder module

        Example
            self.encoder = TextEncoder(vocab_size, emb_dim, max_text_length, n_heads, n_layers)
    """
    def __init__(self, vocab_size, emb_dim, max_seq_length, n_heads, n_layers, dropout=0.1):
        """
        :param: vocab_size: the size of the vocabulary
        :param: emb_dim: the embedding dimension
        :param: max_seq_length: the maximum sequence length
        :param: n_heads: number of attention layers in multi-head attention module
        :param: n_layers: number of encoder layers
        :param: dropout: optional % of nodes to drop out
        """
        super().__init__()

        self.encoder_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = PositionalEmbedding(emb_dim, max_seq_length)

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [TransformerEncoder(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, text, src_mask=None):
        """ Forward pass of the model
        Note: we apply dropout after the positional embeddings
        :param: text: text to input into the model
        :param: src_mask: optional padding mask
        :return: input_val: the transformed input value
        """
        input_val = self.encoder_embedding(text)
        input_val = self.positional_embedding(input_val)
        input_val = self.dropout(input_val)

        for encoder_layer in self.encoder:
            input_val = encoder_layer(input_val, src_mask=src_mask)

        return input_val


class TextDecoder(nn.Module):
    """ Creates the TextDecoder module of the encoder-decoder transformer

        This class creates the text decoder side. It puts together a layered structure, which
        takes input, adds the positional embeddings, drops out some nodes (sets their activation
        to zero), and creates an n_layers size TransformerDecoder module, creates an output
        projection that is a linear transform of the decoder output to the vocabulary.

        Attributes
            decoder_embedding: the vector for the embedded input to the model
            positional_embeddings: positonal embeddings from the class PositionalEmbeddings
            dropout: run a dropout pass
            decoder: n_layers sized decoder module
            output_projection: transform the output into vocabulary words

        Example
            self.decoder = TextDecoder(vocab_size, emb_dim, max_summary_length, n_heads, n_layers)
    """
    
    def __init__(self, vocab_size, emb_dim, max_seq_length, n_heads, n_layers, dropout=0.1):
        """
        :param: vocab_size: the size of the vocabulary
        :param: emb_dim: the embedding dimension
        :param: max_seq_length: the maximum sequence length
        :param: n_heads: number of attention layers in multi-head attention module
        :param: n_layers: number of encoder layers
        :param: dropout: optional % of nodes to drop out
        """
        super().__init__()

        self.decoder_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = PositionalEmbedding(emb_dim, max_seq_length)

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.output_projection = nn.Linear(emb_dim, vocab_size)

    def forward(self, tgt, encoder_outputs, src_mask=None, tgt_mask=None):
        """ Forward pass of the model
        Note: we apply dropout after the positional embeddings
        :param: tgt: target output for the model
        :param: encoder outputs: outputs from the TextEncoder class
        :param: src_mask: optional padding mask for the inputs
        :param: tgt_mask: optional padding mask for the targets
        :return: input_val: the transformed input value
        """
        q = self.decoder_embedding(tgt)
        q = self.positional_embedding(q)
        q = self.dropout(q)

        for decoder_layer in self.decoder:
            q = decoder_layer(q, encoder_outputs, src_mask=src_mask, tgt_mask=tgt_mask)

        q = self.output_projection(q)
        return q


class EncoderDecoder(nn.Module):
    """ Creates the EncoderDecoder transformer

        This class puts everything together creating the encoder-decoder transformer. 

        Attributes
            encoder: l_layers sized encoder module
            decoder: n_layers sized decoder module

        Example
            model = EncoderDecoder(
            vocab_size=tokenizer.vocab_size,
            emb_dim=config['emb_dim'],
            max_text_length=config['max_text_length'],
            max_summary_length=config['max_summary_length'],
            n_heads=config['text_heads'],
            n_layers=config['text_layers'],
            ).to(device)
    """
    def __init__(self, vocab_size, emb_dim, max_text_length, max_summary_length, n_heads, n_layers):
        """
        :param: vocab_size: the size of the vocabulary
        :param: emb_dim: the embedding dimension
        :param: max_text_length: the maximum input sequence length
        :param: max_summary_length: the maximum output sequence length
        :param: n_heads: number of attention layers in multi-head attention module
        :param: n_layers: number of encoder layers
        """
        super().__init__()
        self.encoder = TextEncoder(vocab_size, emb_dim, max_text_length, n_heads, n_layers)
        self.decoder = TextDecoder(vocab_size, emb_dim, max_summary_length, n_heads, n_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """ Forward pass of the model
        :param: src: the input text for the pass
        :param: tgt: the target output text for the pass
        :param: src_mask: optional padding mask for the inputs
        :param: tgt_mask: optional padding mask for the targets
        :return: decoder_outputs: the output of the TextDecoder module 
        """
        encoder_outputs = self.encoder(src, src_mask=src_mask)

        decoder_outputs = self.decoder(tgt, encoder_outputs, src_mask=src_mask, tgt_mask=tgt_mask)

        return decoder_outputs

    def generate(self, src, src_mask=None, max_length=128):
        """ Generate TextDecoder outputs to make a summary
        encoder_outputs Shape: (batch_size, seq_len, hidden_dim)
        predicted_text Shape: (B, 1)
        mask encoder outputs corresponding with padding in the input
        alternate line:
        next_token = torch.multinomial(torch.softmax(decoder_outputs[:, -1, :], dim=-1), num_samples=1)  # Sample from the output distribution
        Stop when all sequences predict <eos>
        :param: src: the input text for the pass
        :param: src_mask: optional padding mask for the inputs
        :param: max_length: optional maximum summary length
        """
        encoder_outputs = self.encoder(src, src_mask=src_mask)

        B = src.size(0)
        # Initialize predicted text with the start token {<s>: 0}
        predicted_text = torch.full((B, 1), fill_value=0, device=src.device)

        for _ in range(max_length):
            decoder_outputs = self.decoder(predicted_text, encoder_outputs,
                                           src_mask=src_mask)

            next_token = decoder_outputs[:, -1, :].argmax(dim=-1, keepdim=True)

            predicted_text = torch.cat((predicted_text, next_token), dim=1)

            if torch.all(next_token == 2):
                break

        return predicted_text



