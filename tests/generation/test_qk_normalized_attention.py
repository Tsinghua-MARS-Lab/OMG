import torch

from omg.generation.denoisers.transformer import (
    MotionTransformerBlock,
    _qk_normalized_cross_attention,
)


def _attention_output(block: MotionTransformerBlock, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(query.shape[0], context.shape[1], dtype=torch.bool)
    return _qk_normalized_cross_attention(block.cross_attn, query, context, mask)


def test_qk_normalized_cross_attention_is_invariant_to_projection_scale():
    torch.manual_seed(7)
    block = MotionTransformerBlock(hidden_dim=32, num_heads=4, dropout=0.0).eval()
    query = torch.randn(2, 5, 32)
    context = torch.randn(2, 3, 32)
    expected = _attention_output(block, query, context)

    with torch.no_grad():
        block.cross_attn.in_proj_weight[:64].mul_(11.0)
        block.cross_attn.in_proj_bias[:64].mul_(11.0)
    actual = _attention_output(block, query, context)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_qk_normalized_cross_attention_input_gradient_is_scale_invariant():
    torch.manual_seed(11)
    block = MotionTransformerBlock(hidden_dim=32, num_heads=4, dropout=0.0).eval()
    context = torch.randn(2, 3, 32)
    query = torch.randn(2, 5, 32, requires_grad=True)
    _attention_output(block, query, context).square().mean().backward()
    expected = query.grad.clone()

    block.zero_grad(set_to_none=True)
    with torch.no_grad():
        block.cross_attn.in_proj_weight[:64].mul_(17.0)
        block.cross_attn.in_proj_bias[:64].mul_(17.0)
    scaled_query = query.detach().clone().requires_grad_(True)
    _attention_output(block, scaled_query, context).square().mean().backward()

    torch.testing.assert_close(scaled_query.grad, expected, atol=2e-7, rtol=2e-5)
