from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

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

    def attention(self, x, attn_mask=None, key_padding_mask=None):
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, attn_weight = self.attention(self.ln_1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weight


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for block in self.resblocks:
            x, attn_weight = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return x, attn_weight


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_weight = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        if self.proj is not None:
            x = torch.bmm(x, self.proj.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = x.permute(1, 0, 2) 
        seq_tokens = x[1:] 
        cls_token = x[0]
        attn_weight = attn_weight[:, 0, 1:]

        return seq_tokens, attn_weight, cls_token  # LND', NS, ND'


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, context_length):
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask.to(self.device)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, key_padding_mask):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x.permute(1, 0, 2)

        attn_mask = self.build_attention_mask(x.shape[0])
        key_padding_mask = key_padding_mask

        x, attn_weight = self.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        EOS = text.argmax(dim=-1)  # N

        attn_weight = attn_weight[torch.arange(x.shape[1]), EOS]  
        attn_weight[torch.arange(x.shape[1]), EOS] = 0 

        new_key_padding_mask = key_padding_mask + (text == 49407)

        x = self.ln_final(x).type(self.dtype)
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.text_projection.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        x = x.permute(1, 0, 2)
        
        seq_tokens = x  
        EOS_token = x[EOS, torch.arange(x.shape[1])]

        return seq_tokens, attn_weight, new_key_padding_mask, EOS_token

    def forward(self, image, text, key_padding_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, key_padding_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

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
    CLIP_INFO = {}

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        CLIP_INFO['vision_width'] = vision_width
        CLIP_INFO['vision_layers'] = vision_layers
        CLIP_INFO['vision_patch_size'] = vision_patch_size
        CLIP_INFO['grid_size'] = grid_size
        CLIP_INFO['image_resolution'] = image_resolution

    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
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
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    CLIP_INFO['embed_dim'] = embed_dim
    CLIP_INFO['context_length'] = context_length
    CLIP_INFO['vocab_size'] = vocab_size
    CLIP_INFO['transformer_width'] = transformer_width
    CLIP_INFO['transformer_heads'] = transformer_heads
    CLIP_INFO['transformer_layers'] = transformer_layers

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
   
    return model, CLIP_INFO


def load_download_clip(clip_path: str) -> tuple:
    try:
        model = torch.jit.load(clip_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(clip_path, map_location="cpu")

    return build_model(state_dict)
