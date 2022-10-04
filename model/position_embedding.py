import math
import torch
from torch import nn

def get_angles(pos, i, dim):
    angle_rates = 1 / torch.pow(10000., 2 * (i//2) / dim)
    # angle_rates = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
    #                      -(math.log(10000.0) / d_model)))

    return pos * angle_rates
    
def positional_encoding(coords, dim):
    """coords in (bsz, size), return (bsz, size, dim)."""
    angle_rads = get_angles(coords[:,None],
                          torch.arange(dim)[None, None, :],
                          dim)
    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[:, :, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[:, :, 1::2])
    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1).float()
    return pos_encoding
    
def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.
    Args:
        seqlen: a `int` specifying the length of the sequence.
        out_dim: a `int` specifying the output dimension of the encoding.
        normalization_max: normalize coordinates between [0, normalization_max].
        If None, raw coordinates from 0 to seqlen will be used.
    Returns:
        positional code of shape (1, seqlen, out_dim)
    """
    # coords = tf.cast(tf.range(seqlen), tf.float32)
    position = torch.arange(0, seqlen).float()
    if normalization_max is not None:
        position = position / (seqlen - 1) * normalization_max
    position = positional_encoding(position, out_dim)
    return position

def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.
    Args:
        height: a `int` specifying the height of the 2d image / feature map.
        width: a `int` specifying the width of the 2d image / feature map.
        out_dim: a `int` specifying the output dimension of the encoding.
        Must be divisible by 2.
        normalization_max: normalize coordinates between [0, normalization_max].
        If None, raw coordinates from 0 to height/width will be used.
    Returns:
        positional code of shape (1, height, width, out_dim)
    """
    y_coords = torch.arange(height).float()
    if normalization_max is not None:
        y_coords = y_coords / (height - 1) * normalization_max
    y_coords = positional_encoding(y_coords, out_dim//2)
    y_coords = torch.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = torch.arange(width).float()
    if normalization_max is not None:
        x_coords = x_coords / (width - 1) * normalization_max
    x_coords = positional_encoding(x_coords, out_dim//2)
    x_coords = torch.unsqueeze(2)
    x_coords = torch.concat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords

def add_seq_pos_emb(length,dim,encoding_type="learned"):
    if encoding_type == "learned":
        weight = nn.Parameter(torch.Tensor(length,dim))
        seq_pos_emb = nn.init.xavier_normal_(weight)
    elif encoding_type == "sin_cos":
        seq_pos_emb = get_1d_position_codes(length,dim)
    else:
        raise ValueError('Unknown pos encoding %s' % encoding_type)
    return seq_pos_emb

def add_vis_pos_emb(n_rows, n_cols, dim,encoding_type="learned"):
    """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""

    if encoding_type == 'learned':
        weight = nn.Parameter(torch.Tensor(n_rows*n_cols,dim))
        vis_pos_emb = nn.init.xavier_normal_(weight)

    elif encoding_type == 'sin_cos':
        sin_cos = get_2d_position_codes(
            n_rows, n_cols, dim, normalization_max=6.2831852)
        vis_pos_emb = sin_cos.reshape(n_rows * n_cols, dim)
    else:
        raise ValueError('Unknown pos encoding %s' % encoding_type)
    return vis_pos_emb

def add_cls_token_emb(dim):
    
    weight = nn.Parameter(torch.Tensor(1,dim))
    cls_token_emb = nn.init.xavier_normal_(weight)
  
    return cls_token_emb

def add_bias_emb(dim):
    
    weight = nn.Parameter(torch.Tensor(1,dim))
    bias_emb = nn.init.xavier_normal_(weight)
  
    return bias_emb

def add_vocab_token_emb(vocab_size,dim):

    weight = nn.Parameter(torch.Tensor(vocab_size,dim))
    token_emd = nn.init.xavier_normal_(weight)
    return token_emd
