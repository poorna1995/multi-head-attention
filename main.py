import torch
import torch.nn as nn
import plots

# Import the MHA classes
from models.causal_attention import Ch03_MHA_Wrapper
from models.multi_head_attention import Ch03_MHA  # Existing MHA class
from models.multi_head_attention_combined import MultiHeadAttentionCombinedQKV  # New combined QKV class
from models.mha_einsum import MHAEinsum  # Importing the MHAEinsum class
from models.mha_pytorch_scaled import MHAPyTorchScaledDotProduct  # Importing the MHA PyTorch scaled dot product class

# Set random seed for reproducibility
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
batch_size = 8
context_len = 1024
embed_dim = 768

# Generate random embeddings
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# Initialize and test MHA Wrapper from Chapter 3
print("Testing MHA Wrapper from Chapter 3...")
mha_ch03_wrapper = Ch03_MHA_Wrapper(
    d_in=embed_dim,
    d_out=embed_dim // 12,  # Adjust according to your architecture
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the MHA wrapper
out_wrapper = mha_ch03_wrapper(embeddings)
print(f"MHA Wrapper Output Shape: {out_wrapper.shape}")  # Should be (batch_size, context_len, d_out * num_heads)

# Initialize and test Ch03_MHA
print("Testing Ch03_MHA...")
mha_ch03 = Ch03_MHA(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the Ch03_MHA
out_mha = mha_ch03(embeddings)
print(f"Ch03_MHA Output Shape: {out_mha.shape}")  # Should be (batch_size, context_len, d_out)

# Initialize and test the Combined QKV Multi-Head Attention
print("Testing Combined QKV MHA...")
mha_combined_qkv = MultiHeadAttentionCombinedQKV(
    d_in=embed_dim,
    d_out=embed_dim,
    num_heads=12,
    context_length=context_len,
    dropout=0.0,
    qkv_bias=False
).to(device)

# Forward pass through the combined QKV MHA
out_combined_qkv = mha_combined_qkv(embeddings)
print(f"Combined QKV MHA Output Shape: {out_combined_qkv.shape}")  # Should be (batch_size, context_len, embed_dim)

# Initialize and test MHA with Einsum
print("Testing MHA Einsum...")
mha_einsum = MHAEinsum(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the MHA with Einsum
out_einsum = mha_einsum(embeddings)
print(f"MHA Einsum Output Shape: {out_einsum.shape}")  # Should be (batch_size, context_len, embed_dim)

# Initialize and test the PyTorch Scaled Dot Product MHA
print("Testing MHA PyTorch Scaled Dot Product...")
mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the PyTorch Scaled Dot Product MHA
out_scaled_dot_product = mha_pytorch_scaled(embeddings)
print(f"MHA PyTorch Scaled Dot Product Output Shape: {out_scaled_dot_product.shape}")  # Should be (batch_size, context_len, embed_dim)

