"""HuggingFace ViT backbone for LeWM.

Matches ``stable_pretraining.backbone.utils.vit_hf`` (galilai-group/stable-pretraining),
which builds ``transformers.ViTModel`` with timm/DINO-style size presets.
See: https://github.com/galilai-group/stable-pretraining/blob/main/stable_pretraining/backbone/utils.py
"""

from __future__ import annotations

from torch import nn
from transformers import ViTConfig, ViTModel

# Size presets from stable-pretraining (timm / DINOv3-style ViT scales).
_SIZE_CONFIGS = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
    },
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "huge": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
    },
}


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = False,
    **kwargs,
) -> ViTModel:
    """Create a HuggingFace ``ViTModel`` with LeWM / stable-pretraining defaults.

    Args:
        size: ``tiny``, ``small``, ``base``, ``large``, or ``huge``.
        patch_size: ViT patch size (LeWM default: 14).
        image_size: Input resolution; must be divisible by ``patch_size``.
        pretrained: If True, load ``google/vit-{size}-patch{patch_size}-{image_size}``
            from the Hub (only when that checkpoint exists).
        use_mask_token: Passed to ``ViTModel`` (LeWM uses ``False``).
        **kwargs: Extra ``ViTConfig`` fields.

    Returns:
        ``ViTModel`` with ``add_pooling_layer=False`` and
        ``config.interpolate_pos_encoding=True`` for variable resolutions.
    """
    if size not in _SIZE_CONFIGS:
        raise ValueError(f"Invalid size {size!r}. Choose from {list(_SIZE_CONFIGS)}")

    config_params = dict(_SIZE_CONFIGS[size])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        model = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(
            config,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )

    model.config.interpolate_pos_encoding = True
    return model
