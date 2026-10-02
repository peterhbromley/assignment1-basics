"""Resource accounting for LLM architecture."""


def resource_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
):
    trainable_params(vocab_size, num_layers, d_model, d_ff)
    matmul_flops(vocab_size, context_length, num_layers, d_model, d_ff)


def trainable_params(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
):
    embedding = vocab_size * d_model

    # Block contains:
    #   2 RMSNorms
    #   Q, K, V, O projection layers, each d_model * d_model in our architecture
    #   3 layers in FFN, each d_model * d_ff in our architecture
    rms_per_block = 2 * d_model
    attn_per_block = 4 * d_model * d_model
    ffn_per_block = 3 * d_model * d_ff
    block = rms_per_block + attn_per_block + ffn_per_block

    final_rms = d_model
    output_head = d_model * vocab_size

    total = embedding + (num_layers * block) + final_rms + output_head

    print(
        "Trainable parameters\n"
        f"  Token embedding: {embedding:,}\n"
        "  Per Transformer block:\n"
        f"    RMSNorms (2): {rms_per_block:,}\n"
        f"    Attention projections (Q, K, V, O): {attn_per_block:,}\n"
        f"    FFN projections (3): {ffn_per_block:,}\n"
        f"    Block subtotal: {block:,}\n"
        f"  All {num_layers} blocks: {num_layers * block:,}\n"
        f"  Final RMSNorm: {final_rms:,}\n"
        f"  Output head: {output_head:,}\n"
        f"  Total parameters: {total:,}"
    )

    return total


def matmul_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
):
    # Recall for MxN, NxP product, 2MNP FLOPs.

    # Embedding layer and norms don't have matmuls.

    # Attention projection matmuls: (seq_len, d_model) @ (d_model, d_model)
    # Attention scaled dot product matmuls (per head):
    #   Q @ K.T:       (seq_len, d_k) @ (d_k, seq_len)
    #   attention @ V: (seq_len, seq_len) @ (seq_len, d_v)
    # ^ Note: since we multiply each of the above by num_heads, and each of d_k
    # and d_v have num_heads in denominator, we can just use d_model and not
    # care about num_heads. Think of it as one big matmul across all heads.
    attn_projections_per_block = 4 * 2 * d_model * d_model * context_length
    attn_qk_per_block = 2 * context_length * context_length * d_model
    attn_v_per_block = 2 * context_length * context_length * d_model
    attn_per_block = attn_projections_per_block + attn_qk_per_block + attn_v_per_block

    # FFN matmuls: seq_len x d_model @ d_model x d_ff, and seq_len x d_ff @ d_ff x d_model
    # ^ Both have same FLOPs
    ffn_per_block = 3 * (2 * d_model * d_ff * context_length)

    block = attn_per_block + ffn_per_block

    output_head = 2 * context_length * d_model * vocab_size

    total = (num_layers * block) + output_head

    print(
        "Forward-pass matrix multiplication FLOPs (batch size 1)\n"
        "  Per Transformer block (all heads):\n"
        f"    Attention projections (Q, K, V, O): {attn_projections_per_block:,}\n"
        f"    Q @ K.T: {attn_qk_per_block:,}\n"
        f"    Attention weights @ V: {attn_v_per_block:,}\n"
        f"    Attention subtotal: {attn_per_block:,}\n"
        f"    FFN projections (3): {ffn_per_block:,}\n"
        f"    Block subtotal: {block:,}\n"
        f"  All {num_layers} blocks: {num_layers * block:,}\n"
        f"  Output head: {output_head:,}\n"
        f"  Total FLOPs: {total:,}"
    )

    return total


def adamw_memory(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    batch_size: int,
) -> dict[str, float]:
    """Print and return the simplified float32 memory estimate for AdamW"""
    d_ff = 8 * d_model / 3

    parameter_count = 2 * vocab_size * d_model + num_layers * (12 * d_model**2 + 2 * d_model) + d_model
    tokens = batch_size * context_length
    hidden = tokens * d_model
    expanded = tokens * d_ff
    attention_matrix = batch_size * num_heads * context_length**2

    per_block_activations = {
        "RMSNorm outputs (2)": 2 * hidden,
        "Q, K, V projections": 3 * hidden,
        "Q @ K.T scores": attention_matrix,
        "Attention softmax": attention_matrix,
        "Attention weights @ V": hidden,
        "Attention output projection": hidden,
        "FFN w1 projection": expanded,
        "FFN SiLU": expanded,
        "FFN w3 projection": expanded,
        "FFN elementwise product": expanded,
        "FFN w2 projection": hidden,
    }
    block_activations = sum(per_block_activations.values())
    logits = tokens * vocab_size
    activation_count = num_layers * block_activations + hidden + 2 * logits

    memory = {
        "parameters_bytes": 4 * parameter_count,
        "activations_bytes": 4 * activation_count,
        "gradients_bytes": 4 * parameter_count,
        "optimizer_state_bytes": 8 * parameter_count,
    }
    memory["total_bytes"] = sum(memory.values())

    print("AdamW float32 memory estimate (simplified activation accounting)")
    print(f"  Theoretical d_ff = 8/3 * d_model: {d_ff:,.3f}")
    print(f"  Per-block activation storage (batch size {batch_size}, all heads):")
    for name, count in per_block_activations.items():
        print(f"    {name}: {4 * count:,.2f} bytes")
    print(f"    Block subtotal: {4 * block_activations:,.2f} bytes")
    print(f"  All {num_layers} blocks' activations: {4 * num_layers * block_activations:,.2f} bytes")
    print(f"  Final RMSNorm activation: {4 * hidden:,.2f} bytes")
    print(f"  Output logits: {4 * logits:,.2f} bytes")
    print(f"  Cross-entropy intermediate: {4 * logits:,.2f} bytes")
    for name, byte_count in memory.items():
        label = name.removesuffix("_bytes").replace("_", " ").capitalize()
        print(f"  {label}: {byte_count:,.2f} bytes ({byte_count / 2**30:.3f} GiB)")

    return memory


def adamw_flops(num_parameters: int) -> int:
    """Estimate FLOPs for one AdamW update (not forward/backward).

    Per parameter: weight decay 2, first moment 3, second moment 4,
    and the final parameter update 5, giving 14 FLOPs total.
    Count sqrt and division as one FLOP each; omit scalar hyperparameter
    and bias-correction calculations. Assume every parameter has a gradient.
    """
    total = 14 * num_parameters
    print(f"AdamW update FLOPs: 14 x {num_parameters:,} = {total:,}")
    return total
