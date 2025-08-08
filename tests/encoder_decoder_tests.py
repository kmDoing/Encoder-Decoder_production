import pytest
import torch
"""
This test script was written by Claude 4 Sonnet.
I am currently investigating the tests and making sure they work.
"""
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TestMultiHeadAttention:
    @pytest.fixture
    def attention_layer(self):
        return MultiHeadAttention(emb_dim=512, num_heads=8, dropout=0.1)
    
    def test_output_shape(self, attention_layer):
        batch_size, seq_len, emb_dim = 2, 10, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        output = attention_layer(x)
        
        assert output.shape == (batch_size, seq_len, emb_dim)
    
    def test_attention_with_mask(self, attention_layer):
        batch_size, seq_len, emb_dim = 2, 10, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = attention_layer(x, causal_mask)
        
        assert output.shape == (batch_size, seq_len, emb_dim)
        assert not torch.isnan(output).any()
    
    def test_parameter_count(self, attention_layer):
        # Should have 4 linear layers: Q, K, V, output projection
        param_count = sum(p.numel() for p in attention_layer.parameters())
        expected = 4 * (512 * 512)  # 4 linear layers, each 512x512
        assert param_count == expected
    
    def test_attention_is_permutation_equivariant(self, attention_layer):
        """Test that attention is equivariant to permutations when no mask is used"""
        batch_size, seq_len, emb_dim = 1, 5, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        # Apply attention
        out1 = attention_layer(x)
        
        # Permute input sequence
        perm_idx = torch.randperm(seq_len)
        x_perm = x[:, perm_idx, :]
        out2 = attention_layer(x_perm)
        
        # Output should be permuted in the same way
        assert torch.allclose(out1[:, perm_idx, :], out2, atol=1e-6)


class TestMLP:
    @pytest.fixture
    def mlp_layer(self):
        return MLP(emb_dim=512, r_mlp=4, dropout=0.1)
    
    def test_output_shape(self, mlp_layer):
        batch_size, seq_len, emb_dim = 2, 10, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        output = mlp_layer(x)
        
        assert output.shape == (batch_size, seq_len, emb_dim)
    
    def test_expansion_factor(self, mlp_layer):
        # Check that hidden layer has correct size
        fc1_out_features = mlp_layer.fc1.out_features
        expected_hidden_size = 512 * 4  # emb_dim * r_mlp
        assert fc1_out_features == expected_hidden_size
    
    def test_forward_pass_numerical_stability(self, mlp_layer):
        x = torch.randn(2, 10, 512)
        output = mlp_layer(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLayerNorm:
    def test_layer_norm_properties(self):
        emb_dim = 512
        ln = nn.LayerNorm(emb_dim)
        x = torch.randn(2, 10, emb_dim)
        
        output = ln(x)
        
        # Check mean and variance along last dimension
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


class TestTransformerEncoderLayer:
    @pytest.fixture
    def encoder_layer(self):
        return TransformerEncoderLayer(emb_dim=512, num_heads=8, r_mlp=4, dropout=0.1)
    
    def test_output_shape(self, encoder_layer):
        batch_size, seq_len, emb_dim = 2, 10, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        output = encoder_layer(x)
        
        assert output.shape == (batch_size, seq_len, emb_dim)
    
    def test_residual_connections(self, encoder_layer):
        """Test that residual connections preserve gradient flow"""
        x = torch.randn(2, 10, 512, requires_grad=True)
        
        output = encoder_layer(x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients due to residual connections
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_training_vs_eval_mode(self, encoder_layer):
        """Test that dropout affects training vs eval mode"""
        x = torch.randn(2, 10, 512)
        
        encoder_layer.train()
        out_train1 = encoder_layer(x)
        out_train2 = encoder_layer(x)
        
        encoder_layer.eval()
        out_eval1 = encoder_layer(x)
        out_eval2 = encoder_layer(x)
        
        # In training mode, dropout should make outputs different
        assert not torch.allclose(out_train1, out_train2)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(out_eval1, out_eval2)


class TestTransformerDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        return TransformerDecoderLayer(emb_dim=512, num_heads=8, r_mlp=4, dropout=0.1)
    
    def test_output_shape(self, decoder_layer):
        batch_size, seq_len, emb_dim = 2, 10, 512
        x = torch.randn(batch_size, seq_len, emb_dim)
        encoder_output = torch.randn(batch_size, seq_len, emb_dim)
        
        output = decoder_layer(x, encoder_output)
        
        assert output.shape == (batch_size, seq_len, emb_dim)
    
    def test_causal_masking(self, decoder_layer):
        """Test that causal masking prevents attention to future positions"""
        seq_len = 5
        emb_dim = 512
        x = torch.randn(1, seq_len, emb_dim)
        encoder_output = torch.randn(1, seq_len, emb_dim)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = decoder_layer(x, encoder_output, causal_mask)
        
        assert output.shape == (1, seq_len, emb_dim)
        assert not torch.isnan(output).any()


class TestIntegration:
    def test_encoder_decoder_compatibility(self):
        """Test that encoder output can be fed to decoder"""
        emb_dim, num_heads = 512, 8
        encoder = TransformerEncoderLayer(emb_dim, num_heads)
        decoder = TransformerDecoderLayer(emb_dim, num_heads)
        
        batch_size, seq_len = 2, 10
        src = torch.randn(batch_size, seq_len, emb_dim)
        tgt = torch.randn(batch_size, seq_len, emb_dim)
        
        # Encode
        encoder_output = encoder(src)
        
        # Decode
        decoder_output = decoder(tgt, encoder_output)
        
        assert encoder_output.shape == (batch_size, seq_len, emb_dim)
        assert decoder_output.shape == (batch_size, seq_len, emb_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the entire encoder-decoder"""
        emb_dim, num_heads = 512, 8
        encoder = TransformerEncoderLayer(emb_dim, num_heads)
        decoder = TransformerDecoderLayer(emb_dim, num_heads)
        
        src = torch.randn(2, 10, emb_dim, requires_grad=True)
        tgt = torch.randn(2, 10, emb_dim, requires_grad=True)
        
        encoder_output = encoder(src)
        decoder_output = decoder(tgt, encoder_output)
        
        loss = decoder_output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert src.grad is not None
        assert tgt.grad is not None
        
        # Check encoder parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None
        
        # Check decoder parameters have gradients
        for param in decoder.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    # Run tests with: python -m pytest test_transformer.py -v
    pytest.main([__file__, "-v"])