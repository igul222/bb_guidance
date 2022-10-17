# TODO: clean up redundancies between here and unet.py
import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


PRETRAINED_DICT = {
  'cifar10': '/atlas2/u/kechoi/improved-diffusion/checkpoints/cifar_time_model_best.ckpt',
  'imagenet64': '/atlas2/u/kechoi/guided-diffusion/results/openai-2022-10-09-16-28-02-934280/model299999.pt',
  'imagenet128': '/atlas2/u/kechoi/guided-diffusion/results/openai-2022-10-09-16-22-04-044408/model299999.pt',
  'imagenet256': '/atlas2/u/kechoi/guided-diffusion/results/openai-2022-10-09-21-20-30-564839/model310000.pt', 
}


def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module


class QKVAttention(nn.Module):
  """
  A module which performs QKV attention and splits in a different order.
  """

  def __init__(self, n_heads):
    super().__init__()
    self.n_heads = n_heads

  def forward(self, qkv):
    """
    Apply QKV attention.
    :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
    :return: an [N x (H * C) x T] tensor after attention.
    """
    bs, width, length = qkv.shape
    assert width % (3 * self.n_heads) == 0
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.chunk(3, dim=1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    weight = th.einsum(
      "bct,bcs->bts",
      (q * scale).view(bs * self.n_heads, ch, length),
      (k * scale).view(bs * self.n_heads, ch, length),
    )  # More stable with f16 than dividing afterwards
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
    return a.reshape(bs, -1, length)


def conv_nd(dims, *args, **kwargs):
  """
  Create a 1D, 2D, or 3D convolution module.
  """
  if dims == 1:
    return nn.Conv1d(*args, **kwargs)
  elif dims == 2:
    return nn.Conv2d(*args, **kwargs)
  elif dims == 3:
    return nn.Conv3d(*args, **kwargs)
  raise ValueError(f"unsupported dimensions: {dims}")


class AttentionPool2d(nn.Module):
  """
  Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
  """

  def __init__(
    self,
    spacial_dim: int,
    embed_dim: int,
    num_heads_channels: int,
    output_dim: int = None,
  ):
    super().__init__()
    self.positional_embedding = nn.Parameter(
      torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
    )
    self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
    self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
    self.num_heads = embed_dim // num_heads_channels
    self.attention = QKVAttention(self.num_heads)

  def forward(self, x):
    b, c, *_spatial = x.shape
    x = x.reshape(b, c, -1)  # NC(HW)
    x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
    x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
    x = self.qkv_proj(x)
    x = self.attention(x)
    x = self.c_proj(x)
    return x[:, :, 0]


def get_timestep_embedding(timesteps, embedding_dim):
  """
  This matches the implementation in Denoising Diffusion Probabilistic Models:
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
  emb = emb.to(device=timesteps.device)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
  return emb


def nonlinearity(x):
  # swish
  return x*torch.sigmoid(x)


def Normalize(in_channels):
  return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = torch.nn.Conv2d(in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)

  def forward(self, x):
    x = torch.nn.functional.interpolate(
      x, scale_factor=2.0, mode="nearest")
    if self.with_conv:
      x = self.conv(x)
    return x


class Downsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      # no asymmetric padding in torch conv, must do it ourselves
      self.conv = torch.nn.Conv2d(in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0)

  def forward(self, x):
    if self.with_conv:
      pad = (0, 1, 0, 1)
      x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
      x = self.conv(x)
    else:
      x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    return x


class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
         dropout, temb_channels=512):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels
    self.use_conv_shortcut = conv_shortcut

    self.norm1 = Normalize(in_channels)
    self.conv1 = torch.nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   padding=1)
    self.temb_proj = torch.nn.Linear(temb_channels,
                     out_channels)
    self.norm2 = Normalize(out_channels)
    self.dropout = torch.nn.Dropout(dropout)
    self.conv2 = torch.nn.Conv2d(out_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   padding=1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = torch.nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1)
      else:
        self.nin_shortcut = torch.nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)

  def forward(self, x, temb):
    h = x
    h = self.norm1(h)
    h = nonlinearity(h)
    h = self.conv1(h)

    h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    return x+h


class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = Normalize(in_channels)
    self.q = torch.nn.Conv2d(in_channels,
                 in_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0)
    self.k = torch.nn.Conv2d(in_channels,
                 in_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0)
    self.v = torch.nn.Conv2d(in_channels,
                 in_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0)
    self.proj_out = torch.nn.Conv2d(in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)

  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return x+h_


class Model(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    in_channels = config.model.in_channels
    resolution = config.data.image_size
    resamp_with_conv = config.model.resamp_with_conv
    num_timesteps = config.diffusion.num_diffusion_timesteps
    
    if config.model.type == 'bayesian':
      self.logvar = nn.Parameter(torch.zeros(num_timesteps))
    
    self.ch = ch
    self.temb_ch = self.ch*4
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels

    # timestep embedding
    self.temb = nn.Module()
    self.temb.dense = nn.ModuleList([
      torch.nn.Linear(self.ch,
              self.temb_ch),
      torch.nn.Linear(self.temb_ch,
              self.temb_ch),
    ])

    # downsampling
    self.conv_in = torch.nn.Conv2d(in_channels,
                     self.ch,
                     kernel_size=3,
                     stride=1,
                     padding=1)

    curr_res = resolution
    in_ch_mult = (1,)+ch_mult
    self.down = nn.ModuleList()
    block_in = None
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = ch*in_ch_mult[i_level]
      block_out = ch*ch_mult[i_level]
      for i_block in range(self.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in,
                     out_channels=block_out,
                     temb_channels=self.temb_ch,
                     dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.num_resolutions-1:
        down.downsample = Downsample(block_in, resamp_with_conv)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in,
                     out_channels=block_in,
                     temb_channels=self.temb_ch,
                     dropout=dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in,
                     out_channels=block_in,
                     temb_channels=self.temb_ch,
                     dropout=dropout)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = ch*ch_mult[i_level]
      skip_in = ch*ch_mult[i_level]
      for i_block in range(self.num_res_blocks+1):
        if i_block == self.num_res_blocks:
          skip_in = ch*in_ch_mult[i_level]
        block.append(ResnetBlock(in_channels=block_in+skip_in,
                     out_channels=block_out,
                     temb_channels=self.temb_ch,
                     dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in, resamp_with_conv)
        curr_res = curr_res * 2
      self.up.insert(0, up)  # prepend to get consistent order

    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1)

  def forward(self, x, t):
    assert x.shape[2] == x.shape[3] == self.resolution

    # timestep embedding
    temb = get_timestep_embedding(t, self.ch)
    temb = self.temb.dense[0](temb)
    temb = nonlinearity(temb)
    temb = self.temb.dense[1](temb)

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1], temb)
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions-1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks+1):
        h = self.up[i_level].block[i_block](
          torch.cat([h, hs.pop()], dim=1), temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h


class Classifier(nn.Module):
  """
  NOTE: same as the diffusion model, but only the downsampling block
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    ch, out_ch, ch_mult = config.model.ch, config.model.cls_out_ch, tuple(config.model.ch_mult)
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    in_channels = config.model.in_channels
    resolution = config.data.image_size
    resamp_with_conv = config.model.resamp_with_conv
    num_timesteps = config.diffusion.num_diffusion_timesteps
    
    if config.model.type == 'bayesian':
      self.logvar = nn.Parameter(torch.zeros(num_timesteps))
    
    self.ch = ch
    self.temb_ch = self.ch*4
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels

    # timestep embedding
    self.temb = nn.Module()
    self.temb.dense = nn.ModuleList([
      torch.nn.Linear(self.ch,
              self.temb_ch),
      torch.nn.Linear(self.temb_ch,
              self.temb_ch),
    ])

    # downsampling
    self.conv_in = torch.nn.Conv2d(in_channels,
                     self.ch,
                     kernel_size=3,
                     stride=1,
                     padding=1)

    curr_res = resolution
    in_ch_mult = (1,)+ch_mult
    self.down = nn.ModuleList()
    block_in = None
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = ch*in_ch_mult[i_level]
      block_out = ch*ch_mult[i_level]
      for i_block in range(self.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in,
                     out_channels=block_out,
                     temb_channels=self.temb_ch,
                     dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.num_resolutions-1:
        down.downsample = Downsample(block_in, resamp_with_conv)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in,
                     out_channels=block_in,
                     temb_channels=self.temb_ch,
                     dropout=dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in,
                     out_channels=block_in,
                     temb_channels=self.temb_ch,
                     dropout=dropout)

    # attention pooling
    # num_head_channels = 64
    # print(resolution, curr_res, ch, num_head_channels, out_ch)
    # self.out = nn.Sequential(
    #     Normalize(block_in),
    #     nn.SiLU(),
    #     AttentionPool2d(
    #         (resolution // curr_res), ch, num_head_channels, out_ch
    #     ),
    # )
    # adaptive pooling
    dims = 2
    out_channels = out_ch
    self.out = nn.Sequential(
      Normalize(block_in),
      nn.SiLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
      # zero_module(conv_nd(dims, ch, out_channels, 1)),
      zero_module(conv_nd(dims, block_in, out_channels, 1)),
      nn.Flatten(),
    )

  def forward(self, x, t):
    assert x.shape[2] == x.shape[3] == self.resolution

    # timestep embedding
    temb = get_timestep_embedding(t, self.ch)
    temb = self.temb.dense[0](temb)
    temb = nonlinearity(temb)
    temb = self.temb.dense[1](temb)

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1], temb)
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions-1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # attention pool
    # h is (bs, 512, 8, 8)
    # TODO: using adaptive pooling for now!
    h = self.out(h)

    return h


def dict2namespace(config):
  namespace = argparse.Namespace()
  for key, value in config.items():
    if isinstance(value, dict):
      new_value = dict2namespace(value)
    else:
      new_value = value
    setattr(namespace, key, new_value)
  return namespace


def load_black_box_classifier(dataset, device):
  if dataset == 'cifar10':
    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    preprocess = None
  elif dataset == 'imagenet':
    # initialize model with best available weights
    # ResNet50_Weights.IMAGENET1K_V2 is 80.858 accuracy
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)

    # data preprocessing for resizing input
    preprocess = weights.transforms()
  else:
    raise NotImplementedError
  
  # prep classifier
  # classifier = classifier.float().to(device).eval()

  return classifier, preprocess


def load_time_dependent_classifier(dataset, device):
  classifier_path = PRETRAINED_DICT[dataset]
  if dataset == 'cifar10':
    with open(os.path.join('./configs/cifar10.yml'), "r") as f:
      config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    classifier = Classifier(new_config)
    classifier.load_state_dict(torch.load(classifier_path))
    preprocess = None
  elif dataset == 'imagenet64':
    raise NotImplementedError
  else:
    raise NotImplementedError
  # prep classifier
  classifier = classifier.float().to(device).eval()

  return classifier, preprocess