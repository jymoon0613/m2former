# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
import copy
from functools import partial
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn as nn
from models.attention import MultiScaleBlock
from models.common import round_width
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair
from einops import rearrange
import torch.nn.functional as F
# from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

############################################################################################################################################################

class PatchMerging(nn.Module):

    def __init__(self, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :].unsqueeze(1)  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :].unsqueeze(1)  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :].unsqueeze(1)  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :].unsqueeze(1)  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=1)  # B 4 H/2 W/2 C
        x = x.mean(dim=1)
        x = x.view(B, -1, C)        

        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_size = 96
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = Linear(96, self.all_head_size)
        self.query2 = Linear(192, self.all_head_size)
        self.query3 = Linear(384, self.all_head_size)
        self.query4 = Linear(768, self.all_head_size)
        
        self.key1 = Linear(96, self.all_head_size)
        self.key2 = Linear(192, self.all_head_size)
        self.key3 = Linear(384, self.all_head_size)
        self.key4 = Linear(768, self.all_head_size)
        
        self.value1 = Linear(96, self.all_head_size)
        self.value2 = Linear(192, self.all_head_size)
        self.value3 = Linear(384, self.all_head_size)
        self.value4 = Linear(768, self.all_head_size)

        self.out1 = Linear(768, 96)
        self.out2 = Linear(768, 192)
        self.out3 = Linear(768, 384)
        self.out4 = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states1, hidden_states2, hidden_states3, hidden_states4):
        
        mixed_query_layer1 = self.query1(hidden_states1)
        mixed_key_layer1 = self.key1(hidden_states1)
        mixed_value_layer1 = self.value1(hidden_states1)
        
        mixed_query_layer2 = self.query2(hidden_states2)
        mixed_key_layer2 = self.key2(hidden_states2)
        mixed_value_layer2 = self.value2(hidden_states2)
        
        mixed_query_layer3 = self.query3(hidden_states3)
        mixed_key_layer3 = self.key3(hidden_states3)
        mixed_value_layer3 = self.value3(hidden_states3)
        
        mixed_query_layer4 = self.query4(hidden_states4)
        mixed_key_layer4 = self.key4(hidden_states4)
        mixed_value_layer4 = self.value4(hidden_states4)
        
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        
        query_layer3 = self.transpose_for_scores(mixed_query_layer3)
        key_layer3 = self.transpose_for_scores(mixed_key_layer3)
        value_layer3 = self.transpose_for_scores(mixed_value_layer3)
        
        query_layer4 = self.transpose_for_scores(mixed_query_layer4)
        key_layer4 = self.transpose_for_scores(mixed_key_layer4)
        value_layer4 = self.transpose_for_scores(mixed_value_layer4)
                
        all_keys = torch.cat([key_layer1, key_layer2, key_layer3, key_layer4], axis=2)
        all_values = torch.cat([value_layer1, value_layer2, value_layer3, value_layer4], axis=2)
                
        attention_scores1 = torch.matmul(query_layer1, all_keys.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_probs1 = self.softmax(attention_scores1)
        attention_probs1 = self.attn_dropout(attention_probs1)
        
        attention_scores2 = torch.matmul(query_layer2, all_keys.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_probs2 = self.softmax(attention_scores2)
        attention_probs2 = self.attn_dropout(attention_probs2)
        
        attention_scores3 = torch.matmul(query_layer3, all_keys.transpose(-1, -2))
        attention_scores3 = attention_scores3 / math.sqrt(self.attention_head_size)
        attention_probs3 = self.softmax(attention_scores3)
        attention_probs3 = self.attn_dropout(attention_probs3)
        
        attention_scores4 = torch.matmul(query_layer4, all_keys.transpose(-1, -2))
        attention_scores4 = attention_scores4 / math.sqrt(self.attention_head_size)
        attention_probs4 = self.softmax(attention_scores4)
        attention_probs4 = self.attn_dropout(attention_probs4)
                
        context_layer1 = torch.matmul(attention_probs1, all_values)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        
        context_layer2 = torch.matmul(attention_probs2, all_values)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        
        context_layer3 = torch.matmul(attention_probs3, all_values)
        context_layer3 = context_layer3.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape3 = context_layer3.size()[:-2] + (self.all_head_size,)
        context_layer3 = context_layer3.view(*new_context_layer_shape3)
        
        context_layer4 = torch.matmul(attention_probs4, all_values)
        context_layer4 = context_layer4.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape4 = context_layer4.size()[:-2] + (self.all_head_size,)
        context_layer4 = context_layer4.view(*new_context_layer_shape4)
        
        attention_output1 = self.out1(context_layer1)
        attention_output1 = self.proj_dropout(attention_output1)
        
        attention_output2 = self.out2(context_layer2)
        attention_output2 = self.proj_dropout(attention_output2)
        
        attention_output3 = self.out3(context_layer3)
        attention_output3 = self.proj_dropout(attention_output3)
        
        attention_output4 = self.out4(context_layer4)
        attention_output4 = self.proj_dropout(attention_output4)
        
        return attention_output1, attention_output2, attention_output3, attention_output4
            
class SpatialCrossAttention(nn.Module):
    def __init__(self):
        super(SpatialCrossAttention, self).__init__()
        self.attention_norm1 = LayerNorm(96, eps=1e-6)
        self.attention_norm2 = LayerNorm(192, eps=1e-6)
        self.attention_norm3 = LayerNorm(384, eps=1e-6)
        self.attention_norm4 = LayerNorm(768, eps=1e-6)
        self.ffn_norm1 = LayerNorm(96, eps=1e-6)
        self.ffn_norm2 = LayerNorm(192, eps=1e-6)
        self.ffn_norm3 = LayerNorm(384, eps=1e-6)
        self.ffn_norm4 = LayerNorm(768, eps=1e-6)
        mlp_ratio = 4.
        dim_mlp_hidden1 = int(96 * mlp_ratio)
        dim_mlp_hidden2 = int(192 * mlp_ratio)
        dim_mlp_hidden3 = int(384 * mlp_ratio)
        dim_mlp_hidden4 = int(768 * mlp_ratio)
        self.ffn1 = MLP(in_features=96, hidden_features=dim_mlp_hidden1, act_layer=nn.GELU, drop=0.0)
        self.ffn2 = MLP(in_features=192, hidden_features=dim_mlp_hidden2, act_layer=nn.GELU, drop=0.0)
        self.ffn3 = MLP(in_features=384, hidden_features=dim_mlp_hidden3, act_layer=nn.GELU, drop=0.0)
        self.ffn4 = MLP(in_features=768, hidden_features=dim_mlp_hidden4, act_layer=nn.GELU, drop=0.0)
        self.attn = CrossAttention()

    def forward(self, x1, x2, x3, x4):
        h1 = x1
        h2 = x2
        h3 = x3
        h4 = x4
        x1 = self.attention_norm1(x1)
        x2 = self.attention_norm2(x2)
        x3 = self.attention_norm3(x3)
        x4 = self.attention_norm4(x4)
        x1, x2, x3, x4 = self.attn(x1, x2, x3, x4)
        x1 = x1 + h1
        x2 = x2 + h2
        x3 = x3 + h3
        x4 = x4 + h4

        h1 = x1
        h2 = x2
        h3 = x3
        h4 = x4
        x1 = self.ffn_norm1(x1)
        x2 = self.ffn_norm2(x2)
        x3 = self.ffn_norm3(x3)
        x4 = self.ffn_norm4(x4)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)
        x1 = x1 + h1
        x2 = x2 + h2
        x3 = x3 + h3
        x4 = x4 + h4
        return x1, x2, x3, x4
        
class ChannelCrossAttention(nn.Module):
    def __init__(self):
        super(ChannelCrossAttention, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(1440, 1440//2),
            nn.BatchNorm1d(1440//2),
            nn.ReLU(inplace=True),
            nn.Linear(1440//2, 1440))

    def forward(self, x1, x2, x3, x4):

        h1 = x1
        h2 = x2
        h3 = x3
        h4 = x4
        
        c1 = x1.mean(dim=1)
        c2 = x2.mean(dim=1)
        c3 = x3.mean(dim=1)
        c4 = x4.mean(dim=1)
                
        x_cat = torch.cat([c1, c2, c3, c4], dim=-1)

        x_cat = F.sigmoid(self.linear(x_cat))

        c1 = x_cat[:, :96]
        c2 = x_cat[:, 96:288]
        c3 = x_cat[:, 288:672]
        c4 = x_cat[:, 672:]

        x1 = x1 * c1.unsqueeze(1)
        x2 = x2 * c2.unsqueeze(1)
        x3 = x3 * c3.unsqueeze(1)
        x4 = x4 * c4.unsqueeze(1)
        
        o1 = x1 + h1
        o2 = x2 + h2
        o3 = x3 + h3
        o4 = x4 + h4

        return o1, o2, o3, o4

class PatchSelection(nn.Module):
    def __init__(self, stage=3):
        super(PatchSelection, self).__init__()
        
        if stage == 1:
            self.len_keep = 162
        elif stage == 2:
            self.len_keep = 54
        elif stage == 3:
            self.len_keep = 18
        else:
            self.len_keep = 6

    def forward(self, x, token):
                
        admat = x
        
        B, L, D = admat.shape
                                
        admat = admat.mean(dim=-1)
                
        inds = torch.argsort(admat, dim=1, descending=True)
        
        inx = inds[:,:self.len_keep]
        
        x_new = torch.gather(x, dim=1, index=inx.unsqueeze(-1).repeat(1, 1, D))
                
        x_new = torch.cat((token.unsqueeze(1), x_new), dim=1)
                
        return x_new, inx

############################################################################################################################################################
            
class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


# @MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        # Prepare input.
        in_chans = 3
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        self.num_classes = num_classes
        embed_dim = cfg.MVIT.EMBED_DIM
        # MViT params.
        num_heads = cfg.MVIT.NUM_HEADS
        depth = cfg.MVIT.DEPTH
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.zero_decay_pos_cls = cfg.MVIT.ZERO_DECAY_POS_CLS

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
        )
        if cfg.MODEL.ACT_CHECKPOINT:
            patch_embed = checkpoint_wrapper(patch_embed)
        self.patch_embed = patch_embed

        patch_dims = [
            spatial_size // cfg.MVIT.PATCH_STRIDE[0],
            spatial_size // cfg.MVIT.PATCH_STRIDE[1],
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, cfg.MVIT.DROPPATH_RATE, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs(
            cfg
        )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.MVIT.MLP_RATIO,
                qkv_bias=cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
            )

            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out
        
        self.norm1 = norm_layer(96)
        self.norm2 = norm_layer(192)
        self.norm3 = norm_layer(384)
        self.norm4 = norm_layer(embed_dim)
        
        self.head1 = TransformerBasicHead(
            96,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head2 = TransformerBasicHead(
            192,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head3 = TransformerBasicHead(
            384,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head4 = TransformerBasicHead(
            768,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head5 = TransformerBasicHead(
            96+192+384+768,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
        self.patch_merge1 = PatchMerging((112, 112))
        self.patch_merge2 = PatchMerging((56, 56))
        self.patch_merge3 = PatchMerging((28, 28))
        self.patch_merge4 = PatchMerging((14, 14))
        
        self.part_select1 = PatchSelection(stage=1)
        self.part_select2 = PatchSelection(stage=2)
        self.part_select3 = PatchSelection(stage=3)
        self.part_select4 = PatchSelection(stage=4)
        
        self.cca = ChannelCrossAttention()
        
        self.sca = SpatialCrossAttention()
        
        self.linear3 = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 384))
        self.linear2 = nn.Sequential(
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192))
        self.linear1 = nn.Sequential(
            nn.Linear(768, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 96))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward_features(self, x):
        
        x, bchw = self.patch_embed(x)

        H, W = bchw[-2], bchw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            x = x + self.pos_embed

        thw = [H, W]
        cnt = 0
        inter = []
        for blk in self.blocks:
            x, thw = blk(x, thw)
            
            if cnt in [1, 4, 20]:
                inter.append(x[:,1:,:])
                             
            cnt += 1
        
        inter.append(x[:,1:,:])
        
        token4 = x[:, 0]
        token3 = self.linear3(token4)
        token2 = self.linear2(token4)
        token1 = self.linear1(token4)
        
        x_mer4 = self.patch_merge4(inter[3])
        x_mer3 = self.patch_merge3(inter[2])
        x_mer2 = self.patch_merge2(inter[1])
        x_mer1 = self.patch_merge1(inter[0])
        
        x_masked1, inx1 = self.part_select1(x_mer1, token1)
        x_masked2, inx2 = self.part_select2(x_mer2, token2)
        x_masked3, inx3 = self.part_select3(x_mer3, token3)
        x_masked4, inx4 = self.part_select4(x_mer4, token4)
                        
        outs = []
        
        out1, out2, out3, out4 = self.cca(x_masked1, x_masked2, x_masked3, x_masked4)
        
        out1, out2, out3, out4 = self.sca(out1, out2, out3, out4)
        
        o1 = self.norm1(out1)
        o2 = self.norm2(out2)
        o3 = self.norm3(out3)
        o4 = self.norm4(out4)
        
        outs.append(o1[:,0])
        outs.append(o2[:,0])
        outs.append(o3[:,0])
        outs.append(o4[:,0])
        
        return outs
    
    def forward(self, x, labels=None):
        outs = self.forward_features(x)
        x1 = self.head1(outs[0])
        x2 = self.head2(outs[1])
        x3 = self.head3(outs[2])
        x4 = self.head4(outs[3])
        x5 = self.head5(torch.cat(outs, -1))
        
        if labels is not None:
                        
            loss1 = smooth_CE(x1.view(-1, self.num_classes), labels.view(-1), 0.6)
            loss2 = smooth_CE(x2.view(-1, self.num_classes), labels.view(-1), 0.7)
            loss3 = smooth_CE(x3.view(-1, self.num_classes), labels.view(-1), 0.8)
            loss4 = smooth_CE(x4.view(-1, self.num_classes), labels.view(-1), 0.9)
            loss5 = smooth_CE(x5.view(-1, self.num_classes), labels.view(-1), 1.0)
            
            return loss1, x1, loss2, x2, loss3, x3, loss4, x4, loss5, x5
        else:
            return x1, x2, x3, x4, x5

def _prepare_mvit_configs(cfg):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    depth = cfg.MVIT.DEPTH
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    for i in range(len(cfg.MVIT.DIM_MUL)):
        dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
    for i in range(len(cfg.MVIT.HEAD_MUL)):
        head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
        stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
        pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
        _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
        cfg.MVIT.POOL_KV_STRIDE = []
        for i in range(cfg.MVIT.DEPTH):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

    for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
        stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
        pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv

def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss

def distillation_loss(out_s, out_t, T):

    loss = F.kl_div(F.log_softmax(out_s/T, dim=1),
                    F.softmax(out_t/T, dim=1),
                    reduction='batchmean') * T * T

    return loss
