import torch

from edurec.recsys.architecture import FusionConfig, SelfAttentionFusion


def test_self_attention_fusion_returns_one_embedding_per_entity() -> None:
    fusion = SelfAttentionFusion(
        FusionConfig(emb_dim=8, n_heads=2, num_module_types=6, dropout=0.0)
    )
    modules = torch.randn(5, 3, 8)

    output = fusion(
        modules,
        module_types=torch.tensor([0, 1, 2]),
        active_modules=torch.tensor([True, True, True]),
    )
    output.sum().backward()

    assert output.shape == (5, 8)
    assert fusion.attention.in_proj_weight.grad is not None


def test_self_attention_fusion_ignores_inactive_modules() -> None:
    fusion = SelfAttentionFusion(
        FusionConfig(emb_dim=8, n_heads=2, num_module_types=6, dropout=0.0)
    ).eval()
    modules = torch.randn(2, 3, 8)
    changed_modules = modules.clone()
    changed_modules[:, 1] = 1_000

    args = {
        "module_types": torch.tensor([3, 4, 5]),
        "active_modules": torch.tensor([True, False, True]),
    }

    torch.testing.assert_close(
        fusion(modules, **args),
        fusion(changed_modules, **args),
    )
