import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, in_channels=3):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(in_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, in_channels: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        if return_patches:
            # LayerNorm is per token, so ln_post(x)[:, 0] == ln_post(x[:, 0]): the CLS output is unchanged.
            x = self.ln_post(x)
            if self.proj is not None:
                x = x @ self.proj
            return x[:, 0], x[:, 1:]  # (B, D) CLS, (B, num_patches, D) patch tokens in the joint space

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, context_length: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=0.02)

    def build_attention_mask(self, seq_len: int, device: torch.device):
        mask = torch.empty(seq_len, seq_len, device=device)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text)
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len].to(x.dtype)

        attn_mask = self.build_attention_mask(seq_len, x.device)

        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)

        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class OpenAICLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        # vision
        image_resolution: int = 224,
        vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
        vision_width: int = 64,
        vision_patch_size: int = 32,
        in_channels: int = 3,
        # text
        vocab_size: int = 49408,
        context_length: int = 256,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        # logit scale / bias (SigLIP uses scale=log(10) and a learnable bias=-10)
        logit_scale_init: float = float(np.log(1 / 0.07)),
        use_logit_bias: bool = False,
        logit_bias_init: float = -10.0,
    ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                in_channels=in_channels,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                in_channels=in_channels,
            )

        self.transformer = TextTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        if use_logit_bias:
            self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_init)
        else:
            self.register_parameter("logit_bias", None)

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        if isinstance(self.visual, VisionTransformer):
            return self.visual.conv1.weight.dtype
        else:
            return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_patches: bool = False):
        if return_patches:
            assert isinstance(self.visual, VisionTransformer), "patch tokens require the ViT image encoder"
            return self.visual(image.type(self.dtype), return_patches=True)
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        return self.transformer(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = self.logits(image_features, text_features)
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def logits(self, image_features, text_features):
        """exp(scale) * <img, txt> (+ bias) for already L2-normalised features, shape (N_img, N_txt)."""
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits


def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.layer")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2])) for k in state_dict if k.startswith("visual.layer")]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2]) for k in state_dict if k.startswith("transformer.resblocks"))

    model = OpenAICLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


# =============================================================================
# Pretrained towers: Qwen3 text encoder + DINOv2 image encoder.
#
# Both backbones stay frozen (by default) and only a linear projection into the
# shared `embed_dim` space is trained, so the embeddings are "just projected"
# from each tower.
# =============================================================================

DINOV2_REPO = "facebookresearch/dinov2"

# hidden (embed) dim of the DINOv2 backbones, needed to build the projection when the backbone
# itself is not loaded (precomputed embeddings). Keys are the torch.hub entrypoints.
DINOV2_HIDDEN_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def pool_tokens(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                pooling_mode: str = "last_eos") -> torch.Tensor:
    """Pool token states the same way as the Qwen3 embedding notebooks."""
    mode = pooling_mode.lower()

    if mode in ("first_eos", "bos", "cls"):
        return last_hidden_states[:, 0]

    if mode in ("last_eos", "eos"):
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).to(last_hidden_states.dtype)
        summed = torch.sum(last_hidden_states * mask, dim=1)
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

    raise ValueError(f"Unsupported pooling mode '{pooling_mode}'.")


class DINOv2ImageTower(nn.Module):
    """DINOv2 ViT backbone (frozen by default) + a trainable linear projection.

    ``forward`` returns the projected CLS token; with ``return_patches=True`` it also
    returns the projected patch tokens (both in the shared embedding space).

    With ``precomputed=True`` the backbone is not loaded at all: ``forward`` instead expects
    the ``(B, 1 + P, hidden_dim)`` [CLS + patch] features produced by
    ``preprocess/generate_dino_emb.py`` and only projects them. This must go together with a
    frozen image tower (there is no backbone left to fine-tune).
    """

    def __init__(self, model_name: str = "dinov2_vits14", embed_dim: int = 512,
                 image_resolution: int = 224, freeze: bool = True, precomputed: bool = False):
        super().__init__()
        self.precomputed = precomputed
        self.model_name = model_name
        self.image_resolution = image_resolution
        self.patch_size = 14  # every DINOv2 entrypoint uses patch size 14

        if precomputed:
            if not freeze:
                raise ValueError("precomputed_image=True requires freeze_image_tower=True "
                                 "(there is no DINOv2 backbone to fine-tune)")
            key = model_name.replace("_reg", "")  # the register variants share the exact same width
            if key not in DINOV2_HIDDEN_DIM:
                raise ValueError(f"unknown DINOv2 model {model_name!r}; known: "
                                 f"{sorted(DINOV2_HIDDEN_DIM)} (+ their _reg variants)")
            self.hidden_dim = DINOV2_HIDDEN_DIM[key]
            self.backbone = None
        else:
            try:
                backbone = torch.hub.load(DINOV2_REPO, model_name, trust_repo=True)
            except TypeError:  # older torch without trust_repo
                backbone = torch.hub.load(DINOV2_REPO, model_name)
            self.backbone = backbone
            self.hidden_dim = int(self.backbone.embed_dim)
            self.patch_size = int(self.backbone.patch_size)

        self.freeze = freeze
        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        if freeze and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        dtype = self.proj.weight.dtype

        if self.precomputed:
            if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"precomputed DINOv2 features must have shape (B, 1 + P, {self.hidden_dim}), "
                    f"got {tuple(x.shape)}")
            cls = self.proj(x[:, 0].to(dtype))
            if return_patches:
                return cls, self.proj(x[:, 1:].to(dtype))
            return cls

        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        if self.freeze:
            cls = cls.detach()
        cls = self.proj(cls.to(dtype))

        if return_patches:
            patches = feats["x_norm_patchtokens"]
            if self.freeze:
                patches = patches.detach()
            return cls, self.proj(patches.to(dtype))
        return cls


class QwenTextTower(nn.Module):
    """Qwen3-Embedding backbone (frozen by default) + a trainable linear projection."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", embed_dim: int = 512,
                 context_length: int = 256, pooling: str = "last_eos", freeze: bool = True,
                 pad_token_id: Optional[int] = None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.context_length = context_length
        self.pooling = pooling
        self.pad_token_id = pad_token_id

        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            if self.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = (input_ids != self.pad_token_id).long()

        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.freeze:
            hidden = hidden.detach()
        pooled = pool_tokens(hidden, attention_mask, self.pooling)
        return self.proj(pooled.to(self.proj.weight.dtype))


class CLIP(nn.Module):
    """Image/text towers with a shared embedding space.

    The image tower is DINOv2, the text tower is Qwen3-Embedding; each backbone output is
    linearly projected into `embed_dim`. Backbones are frozen by default.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 224,
        text_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        image_model_name: str = "dinov2_vits14",
        text_pooling: str = "last_eos",
        context_length: int = 256,
        freeze_text_tower: bool = True,
        freeze_image_tower: bool = True,
        precomputed_image: bool = False,
        pad_token_id: Optional[int] = None,
        # logit scale / bias (SigLIP uses scale=log(10) and a learnable bias=-10)
        logit_scale_init: float = float(np.log(1 / 0.07)),
        use_logit_bias: bool = False,
        logit_bias_init: float = -10.0,
    ):
        super().__init__()

        self.visual = DINOv2ImageTower(
            model_name=image_model_name,
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            freeze=freeze_image_tower,
            precomputed=precomputed_image,
        )
        self.transformer = QwenTextTower(
            model_name=text_model_name,
            embed_dim=embed_dim,
            context_length=context_length,
            pooling=text_pooling,
            freeze=freeze_text_tower,
            pad_token_id=pad_token_id,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        if use_logit_bias:
            self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_init)
        else:
            self.register_parameter("logit_bias", None)

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image(self, image, return_patches: bool = False):
        return self.visual(image.type(self.dtype), return_patches=return_patches)

    def encode_text(self, tokens):
        if isinstance(tokens, dict):
            return self.transformer(tokens["input_ids"], tokens["attention_mask"])
        return self.transformer(tokens)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = self.logits(image_features, text_features)
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def logits(self, image_features, text_features):
        """exp(scale) * <img, txt> (+ bias) for already L2-normalised features, shape (N_img, N_txt)."""
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits
