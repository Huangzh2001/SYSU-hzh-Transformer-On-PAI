import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 这个函数跟swin-transformer中的一样
# 很简单的一个线性层
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

# 这个函数跟swin-transformer中的一样

def window_partition(x, window_size):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # 1.得到x的shape
    B, H, W, C = x.shape
    # 2.使用view函数（Pytorch中的view函数主要用于Tensor维度的重构，即返回一个有相同数据但不同维度的Tensor。）、
    # 将[B, H, W, C]->[B, H // window_size, window_size, W // window_size, window_size, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    # 3.再使用permute调换2，3所在的位置（即[B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]）
    # 然后再通过view()方法将[B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 上面的处理过程可以自己想象一下理解。我反正是理解了的。
    # 4.将经过上述处理的东西返回
    return windows

# 这个函数跟swin-transformer中的一样
def window_reverse(windows, window_size, H, W):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # 上面的window_partition是将图片化成窗口。这里是将窗口还原为图片。这两个函数可以一起看。这个函数的过程其实就是上述函数的反过程。
    # 1.首先先计算一下batch的维度（其实就是window_partition中的反向计算）
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 2.通过view函数将[B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 3.再使用permute调换2，3所在的位置（即[B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]）
    # 然后再通过view()方法将[B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # 4.最后将恢复的图片返
    return x

# 这个函数跟swin-transformer中的一样
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        # 因为用的是多头注意力机制。所以用维度除以头数，得到每一个头的维数。
        # 关于多头注意力机制的过程可以看”transformer中的selfattention以及multihead attention“；里的10：00~15：00
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 下面的操作是建立位置参数表和相对位置索引（我们要根据相对位置索引在位置参数表里面找相对应的数，然后加上去）
        # 具体的原理可以看原理视频的38：00~48：00
        # 1）创建相对位置偏差的参数表（这是一个可以训练的东西）
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 2）生成相对位置索引（我们要根据这个索引在上面的参数表里取参数）
        # 具体tensor在下面操作中是如何变的看代码视频的56：00~1：03：00
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 关于这个qkv的原理，看视频：”transformer中的selfattention以及multihead attention“；里的2：00~5：00
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # 1.首先获得我们的x的shape：[batch_size*num_windows, Mh*Mw, total_embed_dim]
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # 2.将x输入到qkv得到：[batch_size*num_windows, Mh*Mw, total_embed_dim]-> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # 再对它进行reshape和permute操作得到[3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]，目的是为了将q、k、v分离出来
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # q、k、v的形状都为[batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 先把q除以根号d，然后再q乘上k的转置（转置的是最后两个维度）得到的形状为[batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 根据index去找位置参数table里的位置数，然后与attn相加
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        # 由于attn是[batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]，relative_position_bias是[nH, Mh*Mw, Mh*Mw]
        # 所以上面用.unsqueeze(0)增加一个维度，然后利用广播机制将二者attn和relative_position_bias相加

        # 如果mask是none的话，就直接做softmax。如果不是none的话，就
        # 将attn与mask相加，然后再做softmax
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # 最后再做一个dropout
        attn = self.attn_drop(attn)

        # 最后再乘以v，再reshape一下
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 再通过proj的线性层对多头进行融合
        x = self.proj(x)
        # 最后再dropout一下
        x = self.proj_drop(x)
        # 最后再返回x
        return x


# 这个函数跟swin-transformer中的一样
# 但少了pad的操作
# 还有将create_mask移到了这里
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # 下面四行与swin transformer有点不同
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 下面这一段代码（到forward之前）是创造一个掩码
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            # 1.创造大小为(1, H, W, 1)的掩码
            # 拥有和feature map一样的通道排列顺序，方便后续window_partition
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            # 关于下面这一段可以看视频31分钟。反正我是理解了。
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # 关于window_partition的看上面定义的函数
            # 2.这一步的作用是：将所有窗口“并列”
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            # 3.使用view函数将[nW, Mh, Mw, 1]->[nW, Mh*Mw]。即将“窗口图片”二维展开成一维。
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 4.利用unsqueeze函数先在中间加一个维度，然后再在最后边加一个维度；最后二者相减。这里涉及广播机制
            # 关于怎么理解这一步呢，可以去看视频40:00
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # [nW, Mh*Mw, Mh*Mw]
            # 5.然后将其中非0的元素替换为-100.为0的元素替换为0。得到attn_mask
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 首先，根据下文。传入的x的形状应该是[B, H0*W0/16, 96]=[B, L, C]，其中L=H*W。传入的attn_mask的形状应该是[nW, Mh*Mw, Mh*Mw]。
        # 1.先判断是否“合法”
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 2.先保存现在的x为shortcut，方便后面用resnet连接。然后x先通过一个norm层。
        shortcut = x
        x = self.norm1(x)
        # 再将x由[B, H*W, C]->[B, H, W, C]
        x = x.view(B, H, W, C)

        # 3.我们通过传入的shift_size是否为0来判断当前使用的是msa还是wmsa。如果是大于0的话就是swmsa，如果是为0就是wmsa。
        # cyclic shift
        if self.shift_size > 0:
            # 如果是大于0的话就是swmsa。这时就需要将x做相应的移动。（你懂的）然后记为shifted_x。
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            # 如果为0的话就是wmas。这时shifted_x就是x。然后attn_mask就为0。
            shifted_x = x

        # 4.先用window_partition将shifted_x分成一个个窗口。即[B, H, W, C]->[nW*B, Mh, Mw, C]
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # 然后再利用view函数将[nW*B, Mh, Mw, C]->[nW*B, Mh*Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # 5.然后将x_windows和attn_mask传入到attn函数中做attention机制
        # 具体怎么做的看上面
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # 6.这段是还原为原来的形状。即[nW*B, Mh*Mw, C]->[nW*B, Mh, Mw, C]
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # 然后利用window_reverse函数将其拼回去。即[nW*B, Mh, Mw, C]->[B, H', W', C]
        # 具体怎么拼的看上面
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 7.如果是使用swmsa的话（此时上面是做了移动了的）这时就要移动回去。如果用的是wmsa的话，则不需要移动回去。
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # 8.将x变成[B, H * W, C]的形状（因为输入这个函数的形状就是[B, H * W, C]，这时要还原回去，好返回
        x = x.view(B, H * W, C)

        # 9.这里做一个resnet连接。
        # FFN
        x = shortcut + self.drop_path(x)
        # 10.这里再做一个norm层和一个mlp层，还有一个dropout层。
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 11.最后将这个x返回
        return x


# 这个函数跟swin-transformer中的一样
# 但少了pad的操作
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    # 有关patchmerging的原理看视频13：00

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        我们输入的x的形状为: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 这段看视频16：00
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 再通过一个normal层
        x = self.norm(x)
        # 再经过全连接层
        x = self.reduction(x) # [B, H/2*W/2, 2*C]

        # 最后返回x
        return x

# 这个就是swin-transformer没有的了
# Patch expanding 层：
# 以第一个patch expanding层为例，在上采样之前，在输入特征（W/32×H/32×8C）上应用线性层，
# 以将特征维度增加到原始维度（W/32 x H/32×16C）的2倍。然后，使用rearrange操作
# 将输入特征的分辨率扩展到输入分辨率的2倍，并将特征维数减小到输入维数的四分之一（W/32×H/32×16C→ W/16×H/16×4C）。
# rearrange是einops中的一个函数
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 下面的(p1 p2 c)就是将原来的C=p1*p2*c拆开，然后将p1、p2给h和w
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

# 这个就是swin-transformer没有的了
# 这跟上面的patchexpand的不同：
# 就在于上面是C拆分为C=2*2*(C/4)，这里是C=dim_scale*dim_scale*(C/dim_scale的平方)
# 这里的dim_scale默认是4，即这里的图片默认扩大四倍
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

# 这个函数跟swin-transformer中的一样
# 但少了create mask！！
# 移到了SwinTransformerBlock
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        # 这个blocks里面有depth个SwinTransformerBlock
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 2.将x弄入一个blocks层。blocks层是由depth个SwinTransformerBlock组成的
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # 2.进行一次降采样。
        if self.downsample is not None:
            x = self.downsample(x)
        # 3.最后返回处理后的x
        return x


# 这个就是swin-transformer没有的了
# 实现的是unet中往上那一部分
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

# 这个函数跟swin-transformer中的一样
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    功能就是将原来的二位图片拉长成一个一维向量。即[B, C, H, W] -> [B, C, HW]-> [B, HW, C]
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # 把img_size变成(img_size, img_size)
        patch_size = to_2tuple(patch_size) # 把patch_size变成(patch_size, patch_size)
        # patches_resolution是图片经过patch后的形状
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # num_patches是patch后的“像素”数
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # patch是用一个卷积层来完成的
        # in_c：[N, C, H, W]中的C了，即输入张量的channels数，RGB为3
        # embed_dim：即期望的四维输出张量的channels数
        # kernel_size：卷积核的大小
        # stride：步长
        # 所以下面用的是一个将[N, 3, H, W] -> [N, 96, H/4, W/4]的卷积核
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 1.首先获得图片的长和宽
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # 2.如果输入图片的H，W不是patch_size的整数倍，
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 3.下采样patch_size倍
        # （1）这里下采样用的是一个卷积层。一个将[N, 1, H, W] -> [N, 96, H/4, W/4]的卷积层
        # （2）下采样完了之后再将宽和高展开为一个一维向量，然后再调整一下位置。
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.to(torch.float32)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # （3）然后再归一化一下
        if self.norm is not None:
            x = self.norm(x)

        # 4.输入[N, 3, H, W]，再返回[N, 3, H, W] -> [N, 96, H/4, W/4]-> [B, 96, HW/16]-> [B, HW/16, 96]的x
        return x


# 这个就是swin-transformer没有的了
# 这个是总的神经网络结构
class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        # 打印出神经网络的向下、向上各层transformer的深度，以及随机深度率（是啥不知道）
        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{}".format(depths,
        depths_decoder,drop_path_rate))

        self.num_layers = len(depths) # 层数，4
        self.embed_dim = embed_dim # 顾名思义，默认96
        self.ape = ape # 是否加绝对位置编码。默认否
        self.patch_norm = patch_norm # 是否在patch embedding后加正则化。默认true
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 96*2的三次方？这个是什么？
        self.num_features_up = int(embed_dim * 2) # 96*2？这个是什么？
        self.mlp_ratio = mlp_ratio # mlp隐藏dim与嵌入dim的比率。
        self.final_upsample = final_upsample # 这个是什么？最后一层做的上采样？

        # 将图像分割成不重叠的patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches # num_patches是patch后的“像素”数
        patches_resolution = self.patch_embed.patches_resolution # patches_resolution是图片经过patch后的形状
        self.patches_resolution = patches_resolution # patches_resolution是图片经过patch后的形状

        # 绝对位置编码
        # 这里默认是false，我们就不看了
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # 按drop_rate的dropout层
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度（不知道是啥）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规律

        # 构造向下的神经网络和“瓶底”的神经网络
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # 构建向上的神经网络
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # 构建正则化层
        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        # 如果final_upsample == "expand_first"。那么最后一个上采样层就是FinalPatchExpand_X4，即图片变四倍。最后再接一个卷积层
        # 至于为什么要x4呢？因为第一层的patch partition就把h、w变成四分之一了
        # 即[w/4,h/4,c]->[w,h,16c]->[w,h,class]
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=in_chans,kernel_size=1,bias=False)

        # 这是啥不知道
        self.apply(self._init_weights)

    #定义网络中的每个层进行权重初始化函数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    # 这里包括向下和底层的神经网络
    def forward_features(self, x):
        # 1.先做一个patch_embed
        # 在这里：[N, 3, H, W] -> [N, 96, H/4, W/4]-> [B, 96, HW/16]-> [B, HW/16, 96]
        x = self.patch_embed(x)
        # 2.如果要加绝对位置编码，则加。这里默认是不加
        if self.ape:
            x = x + self.absolute_pos_embed
        # 3.一个dropout层
        x = self.pos_drop(x)
        # x_downsample这个是用来保存各个中间tensor的，用来unet连接
        x_downsample = []

        # 4.将x输入向下和“瓶底”的神经网络，并且将各输出存放在x_downsample（设计的巧妙，经过“瓶底”后的输出不存在x_downsample中）
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        # 5.最后经过一个norm层
        x = self.norm(x)  # B L C

        # 6.返回x和x_downsample
        return x, x_downsample

    #Dencoder and Skip connection
    # 这段是向上的
    def forward_up_features(self, x, x_downsample):
        # 1.将x输入到向上的神经网络中
        for inx, layer_up in enumerate(self.layers_up):
            # 第一次不需要跳层连接
            if inx == 0:
                x = layer_up(x)
            else:
                # 2.跳层连接
                x = torch.cat([x,x_downsample[3-inx]],-1)
                # concat_back_dim里面是一个nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer))
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        # 3.最后再经过一个norm层
        x = self.norm_up(x)  # B L C

        # 4.返回x
        return x

    # 这是最后一层
    # 功能是先通过一个上采样层，扩大四倍然后，再通过一个卷积层
    # 即[w/4,h/4,c=96]->[w,h,c=96]->[w,h,class=1]，这里的class我们要设计为1
    # 这三个要结合在一起看才清楚：
    # 第一部分：
    # 注：其中patch_size=4，embed_dim=96，img_size就是原图尺寸
        # self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
        #                               dim_scale=4, dim=embed_dim)
        # self.output = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, bias=False)
    # 第二部分：
        # class FinalPatchExpand_X4(nn.Module):
        #     def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        #         super().__init__()
        #         self.input_resolution = input_resolution
        #         self.dim = dim
        #         self.dim_scale = dim_scale
        #         self.expand = nn.Linear(dim, 16 * dim, bias=False)
        #         self.output_dim = dim
        #         self.norm = norm_layer(self.output_dim)
        #
        #     def forward(self, x):
        #         """
        #         x: B, H*W, C
        #         """
        #         H, W = self.input_resolution
        #         # 先将c变成16*c
        #         x = self.expand(x)
        #         B, L, C = x.shape
        #         assert L == H * W, "input feature has wrong size"
        #
        #         x = x.view(B, H, W, C)
        #         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
        #                       c=C // (self.dim_scale ** 2))
        #         x = x.view(B, -1, self.output_dim)
        #         x = self.norm(x)
        #
        #         return x
    # 第三部分：
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    # 总的前向传播函数
    def forward(self, x):
        # 依次是上面定义的几个模型
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

# 下面来定义模型
def hzh_swin_unet_window7_448_448(**kwargs):
    model = SwinTransformerSys(
        img_size=448, # 输入图片的尺寸
        patch_size=4, # 每个patch的窗口看作一个元素
        in_chans=1, # 灰度图像的channel为1
        embed_dim=96,
        depths=[2, 2, 2, 2], # 左半边每个Swin Transformer layer的深度
        depths_decoder=[1, 2, 2, 2], # 右半边每个Swin Transformer layer的深度
        num_heads=[3, 6, 12, 24], #每层注意力头的数量
        window_size=7, # windowSize就是一个窗口内含有的patch个数
                       # 如果patch_size=4则原来的图片为[448, 448]，则patch之后可以看作是[112,112]，然后windowSize要能整除这个
                       # 所以我选择windowSize = 7
        mlp_ratio=4., # mlp隐藏dim与嵌入dim的比率。
        qkv_bias=True, # true，则给qkv一个可学习的偏置
        qk_scale=None, # 覆盖head_dim的默认qk比例。这个不知道是什么
        drop_rate=0., # 这个是dropout的概率
        attn_drop_rate=0., # attention里面的dropout的概率
        drop_path_rate=0.1, # 随机深度率。这个不知道是什么
        norm_layer=nn.LayerNorm, # 标准化层
        ape=False, # 如果为True，则将绝对位置嵌入添加到面片嵌入patch embeding中
        patch_norm=True, # 如果为True，则在patch embeding后添加规范化
        use_checkpoint=False, # 是否使用检查点来节省内存
    )
    return model