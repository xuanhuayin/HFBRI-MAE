import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN

import torch
import torch.nn as nn

class ChamferDistanceL2(nn.Module):
    """
    Chamfer Distance L2 between two point clouds xyz1 and xyz2.
    """
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: (B, N, 3)
            xyz2: (B, M, 3)
        Returns:
            chamfer distance: scalar
        """
        if self.ignore_zeros:
            mask1 = torch.sum(xyz1, dim=2).ne(0)  # (B, N)
            mask2 = torch.sum(xyz2, dim=2).ne(0)  # (B, M)
            xyz1 = [x[m] for x, m in zip(xyz1, mask1)]
            xyz2 = [x[m] for x, m in zip(xyz2, mask2)]
            xyz1 = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, xyz1[0].size(0) - x.size(0))) for x in xyz1])
            xyz2 = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, xyz2[0].size(0) - x.size(0))) for x in xyz2])

        B, N, _ = xyz1.size()
        _, M, _ = xyz2.size()

        # Compute pairwise distance
        dist_matrix = torch.cdist(xyz1, xyz2, p=2)  # (B, N, M)

        # For each point in xyz1, find closest point in xyz2
        dist1 = torch.min(dist_matrix, dim=2)[0]  # (B, N)
        # For each point in xyz2, find closest point in xyz1
        dist2 = torch.min(dist_matrix, dim=1)[0]  # (B, M)

        return (torch.mean(dist1 ** 2) + torch.mean(dist2 ** 2))

class RI_encoder(nn.Module):
    def __init__(self, encoder_channel):
        super(RI_encoder, self).__init__()

        # self.nsample = nsample
        self.prev_mlp_convs = nn.ModuleList()
        self.prev_mlp_bns = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        in_channel_0 = 8
        mlp_0 = [64, 128]
        last_channel = in_channel_0
        for out_channel in mlp_0:
            self.prev_mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.prev_mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        last_channel = mlp_0[-1]
        for out_channel in encoder_channel:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel


    def forward(self, xyz, centers, center_norms, idx, pts, norm_ori):
        xyz = xyz.contiguous()
        ri_feat, idx_ordered = RI_features(xyz, centers, center_norms, idx, pts, norm_ori)

        # lift
        ri_feat = ri_feat.permute(0, 3, 2, 1)  # B, 8, K, N
        for i, conv in enumerate(self.prev_mlp_convs):
            bn = self.prev_mlp_bns[i]
            ri_feat = F.relu(bn(conv(ri_feat)))

        # concat previous layer features

        new_points = ri_feat

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        ri_feat = torch.max(new_points, 2)[0]  # maxpooling  B, 384, N

        return ri_feat.permute(0, 2, 1)





def compute_LRA(xyz, weighting=False, nsample=64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)
    # print('xyz.shape', xyz.shape)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    # eigen_values, vec = M.symeig(eigenvectors=True)

    LRA = vec[:, :, :, 0]

    B, N, _ = xyz.shape
    inner_product = (LRA * xyz).sum(-1)
    mask = torch.ones((B, N), device=LRA.device)
    mask[inner_product < 0] = -1
    LRA = LRA * mask.unsqueeze(-1)
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA  # B N 3



#     return xyz, basis


def RI_features(xyz, centers, center_norms, idx, pts, norm_ori):
    B, S, C = centers.shape

    new_norm = center_norms.unsqueeze(-1)
    dots_sorted, idx_ordered = order_index(xyz, centers, new_norm, idx)

    epsilon = 1e-7
    grouped_xyz = index_points(pts, idx_ordered)  # [B, npoint, nsample, C]
    # print('neighborhood.shape', xyz.shape)
    # print('norm.shape', norm.shape)
    # print('idx', idx.shape)
    grouped_xyz_distance = torch.norm(grouped_xyz, dim=-1, keepdim=True)
    grouped_xyz_local = grouped_xyz - centers.view(B, S, 1, C)  # treat orgin as center
    grouped_xyz_length = torch.norm(grouped_xyz_local, dim=-1, keepdim=True)  # nn lengths
    grouped_xyz_unit = grouped_xyz_local / grouped_xyz_length
    grouped_xyz_unit[grouped_xyz_unit != grouped_xyz_unit] = 0  # set nan to zero
    grouped_xyz_norm = index_points(norm_ori, idx_ordered)  # nn neighbor normal vectors

    grouped_xyz_angle_0 = torch.matmul(grouped_xyz_unit, new_norm)
    grouped_xyz_angle_1 = (grouped_xyz_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_angle_norm = torch.matmul(grouped_xyz_norm, new_norm)
    grouped_xyz_angle_norm = torch.acos(torch.clamp(grouped_xyz_angle_norm, -1 + epsilon, 1 - epsilon))  #
    D_0 = (grouped_xyz_angle_0 < grouped_xyz_angle_1)
    D_0[D_0 == 0] = -1
    grouped_xyz_angle_norm = D_0.float() * grouped_xyz_angle_norm

    grouped_xyz_inner_vec = grouped_xyz_local - torch.roll(grouped_xyz_local, 1, 2)
    grouped_xyz_inner_length = torch.norm(grouped_xyz_inner_vec, dim=-1, keepdim=True)  # nn lengths
    grouped_xyz_inner_unit = grouped_xyz_inner_vec / grouped_xyz_inner_length
    grouped_xyz_inner_unit[grouped_xyz_inner_unit != grouped_xyz_inner_unit] = 0  # set nan to zero
    grouped_xyz_inner_angle_0 = (grouped_xyz_inner_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_1 = (grouped_xyz_inner_unit * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = (grouped_xyz_norm * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = torch.acos(torch.clamp(grouped_xyz_inner_angle_2, -1 + epsilon, 1 - epsilon))
    D_1 = (grouped_xyz_inner_angle_0 < grouped_xyz_inner_angle_1)
    D_1[D_1 == 0] = -1
    grouped_xyz_inner_angle_2 = D_1.float() * grouped_xyz_inner_angle_2

    proj_inner_angle_feat = dots_sorted - torch.roll(dots_sorted, 1, 2)
    proj_inner_angle_feat[:, :, 0, 0] = (-3) - dots_sorted[:, :, -1, 0]

    ri_feat = torch.cat([grouped_xyz_length,
                         proj_inner_angle_feat,
                         grouped_xyz_angle_0,
                         grouped_xyz_angle_1,
                         grouped_xyz_angle_norm,
                         grouped_xyz_inner_angle_0,
                         grouped_xyz_inner_angle_1,
                         grouped_xyz_inner_angle_2], dim=-1)


    return ri_feat, idx_ordered


def index_points_2d(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, norm=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    center = index_points_2d(xyz, centroids)

    norm_center = index_points_2d(norm, centroids)
    return center, norm_center, centroids


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]
    return new_points


def order_index(xyz, new_xyz, new_norm, idx):
    B, S, C = new_xyz.shape
    nsample = xyz.shape[2]
    grouped_xyz = xyz
    grouped_xyz_local = grouped_xyz  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane * new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / proj_xyz_length
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1, 1, 1,
                                                                               3))  # corresponds to the largest length

    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the center point itself, just set sign as 1 to differ from ref_vec
    dots = sign * dots - (1 - sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered



def knn(xyz, centroids, k):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        centroids: sampled pointcloud index, [B, npoint]
        k: number of nearest neighbors
    Return:
        knn_indices: indices of k nearest neighbors for each centroid, [B, npoint, k]
    """
    B, N, C = xyz.shape
    npoint = centroids.shape[1]
    centroids_xyz = xyz[torch.arange(B, dtype=torch.long)[:, None], centroids]  # [B, npoint, 3]
    dists = torch.cdist(centroids_xyz, xyz)  # [B, npoint, N]
    knn_indices = torch.topk(dists, k, dim=-1, largest=False)[1]  # [B, npoint, k]
    return knn_indices


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, norm=None, radius=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out

        center, norm_center, idx_new = farthest_point_sample(xyz, self.num_group, norm)  # B G 3
        idx = knn(xyz, idx_new, self.group_size)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        neighborhood = index_points(xyz, idx)
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        norm = index_points(norm, idx)
        norm = norm.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        return neighborhood, center, norm, norm_center, idx, idx_new


## Transformers
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


def get_robust_centroid(input):
    # input: [B, N, K, 3]
    batch_size, point_num, neighbor_size, _ = input.shape
    subset_size = int(round(neighbor_size * 0.9))  # select 90% of K

    centroid_list = []
    sample_num = 10  # 10  # 5
    sample_num_selected = int(round(sample_num * 0.7))

    for i in range(sample_num):
        input_transpose = input.permute(2, 1, 0, 3)  # [K, N, B, 3]
        input_transpose_shuffle = input_transpose[torch.randperm(neighbor_size)]  # shuffle along the K dimension
        subset_transpose = input_transpose_shuffle[:subset_size, ...]
        subset = subset_transpose.permute(2, 1, 0, 3)  # [B, N, K*0.9, 3]

        centroid = torch.mean(subset, dim=2, keepdim=True)  # [B, N, 1, 3]
        centroid_list.append(centroid)

    centroid_list = torch.cat(centroid_list, dim=2)  # [B, N, sample_num, 3]

    centroid_list_transpose = centroid_list.permute(0, 1, 3, 2)  # [B, N, 3, sample_num]
    inner = torch.matmul(centroid_list, centroid_list_transpose)  # [B, N, sample_num, sample_num]

    centroid_list_square = torch.sum(centroid_list ** 2, dim=-1, keepdim=True)  # [B, N, sample_num, 1]
    centroid_list_square_transpose = centroid_list_square.permute(0, 1, 3, 2)  # [B, N, 1, sample_num]

    pairwise_dist = centroid_list_square + centroid_list_square_transpose - 2 * inner  # [B, N, sample_num, sample_num]
    pairwise_dist_sum = torch.sum(pairwise_dist, dim=-1)  # [B, N, sample_num]

    sorted_indices = torch.topk(pairwise_dist_sum, sample_num, largest=False).indices  # sorted from smallest to largest
    sorted_indices_selected = sorted_indices[:, :, :sample_num_selected]  # [B, N, P]

    # group
    flattened_centroid = centroid_list.view(-1, 3)  # [B*N*sample_num, 3]
    indices_reshape = sorted_indices_selected.view(batch_size * point_num, -1)  # [B*N, P]
    offset = torch.arange(0, batch_size * point_num, device=input.device).unsqueeze(1) * sample_num
    flattened_indices = (indices_reshape + offset).view(-1)  # [B*N*P]
    selected_rows = flattened_centroid[flattened_indices]
    centroid_selected = selected_rows.view(batch_size, point_num, sample_num_selected, 3)  # [B, N, P, 3]

    centroid_avg = torch.mean(centroid_selected, dim=2)  # [B, N, 3]

    return centroid_avg


def calculate_radius(grouped_xyz):
    """
    Calculate the average radius for the grouped points.
    Input:
        grouped_xyz: grouped point cloud data, [B, N, k, 3]
    Return:
        radius: average radius, scalar
    # """
    dists = torch.norm(grouped_xyz, dim=-1)  # [B, npoint, nsample]

    # Find the maximum distance in each group
    max_dists = dists.max(dim=2)[0]  # [B, npoint]

    # Calculate the average of the maximum distances
    radius = max_dists.mean()
    return radius


def Global_feature(new_xyz, radius, grouped_xyz):
    centroid_xyz = get_robust_centroid(grouped_xyz)  # [B, N, 3]

    # calculate intersection point
    reference_vector_norm = torch.norm(new_xyz, dim=-1, keepdim=True)  # [B, N, 1]
    reference_vector_unit = new_xyz / (reference_vector_norm + 1e-4)  # [B, N, 3]
    inter_xyz = radius * reference_vector_unit + new_xyz

    # prepare features of center point
    centroid_reference_vector = new_xyz - centroid_xyz
    centroid_reference_dist = torch.norm(centroid_reference_vector, dim=-1, keepdim=True)  # [B, N, 1]

    centroid_inter_vector = inter_xyz - centroid_xyz
    centroid_inter_dist = torch.norm(centroid_inter_vector, dim=-1, keepdim=True)  # [B, N, 1]

    dot_product = torch.sum(centroid_reference_vector * centroid_inter_vector, dim=-1, keepdim=True)
    reference_centroid_inter_angle = dot_product / (centroid_reference_dist * centroid_inter_dist + 1e-6)

    inter_reference_vector = new_xyz - inter_xyz
    inter_centroid_vector = centroid_xyz - inter_xyz
    dot_product = torch.sum(inter_reference_vector * inter_centroid_vector, dim=-1, keepdim=True)
    reference_inter_centroid_angle = dot_product / (radius * centroid_inter_dist + 1e-6)

    center_point_features = torch.cat([
        reference_vector_norm,
        centroid_reference_dist,
        centroid_inter_dist,
        reference_centroid_inter_angle,
        reference_inter_centroid_angle
    ], dim=-1)  # [B, N, 5]

    return center_point_features


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.radius = config.transformer_config.radius
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.RI_encoder = RI_encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, center_norms, idx, pts, norm_ori, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.RI_encoder(neighborhood, center, center_norms, idx, pts, norm_ori)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        _, N, _ = x_vis.shape
        neighborhood_vis = neighborhood[~bool_masked_pos].reshape(batch_size, N, -1, 3)
        grouped_xyz = index_points(pts, idx)
        # add pos embedding
        # mask pos center
        radius = calculate_radius(neighborhood_vis)
        RI_global_feat = Global_feature(center, radius, grouped_xyz)
        pos = self.pos_embed(RI_global_feat)

        pos = pos[~bool_masked_pos].reshape(batch_size, -1, self.trans_dim)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis_norm = self.norm(x_vis)

        return x_vis_norm, bool_masked_pos


@MODELS.register_module()
class HFBRI_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[HFBRI_MAE] ', logger='HFBRI_MAE')
        self.config = config
        self.radius = config.transformer_config.radius
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[HFBRI_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='HFBRI_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def forward(self, pts, pts_ori=None, vis=False, eval=False, **kwargs):
        norm_ori = compute_LRA(pts)
        neighborhood, center, norm, center_norms, idx, idx_center = self.group_divider(pts, norm_ori)
        if pts_ori is not None:
            center_ori = index_points_2d(pts_ori, idx_center)
            neighborhood_ori = index_points(pts_ori, idx)
            neighborhood_ori = neighborhood_ori - center_ori.unsqueeze(2)

        if eval:
            x_vis, mask = self.MAE_encoder(neighborhood, center, center_norms, idx, pts, norm_ori, noaug=True)
            return x_vis.mean(1) + x_vis.max(1)[0]
        else:
            x_vis, mask = self.MAE_encoder(neighborhood, center, center_norms, idx, pts, norm_ori)
        B, _, C = x_vis.shape  # B VIS C

        center_vis = center[~mask].reshape(B, -1, 3)
        center_mask = center[mask].reshape(B, -1, 3)

        _, N_vis, _ = center_vis.shape
        _, N_mask, _ = center_mask.shape

        group_xyz = index_points(pts, idx)
        radius = calculate_radius(neighborhood)

        RI_global_feat = Global_feature(center, radius, group_xyz)

        RI_global_feat_vis = RI_global_feat[~mask].reshape(B, N_vis, -1)
        RI_global_feat_mask = RI_global_feat[mask].reshape(B, N_mask, -1)


        pos_emd_vis = self.decoder_pos_embed(RI_global_feat_vis).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(RI_global_feat_mask).reshape(B, -1, C)

        RI_global_feat_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        mask_token = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)

        x_rec = self.MAE_decoder(x_full, RI_global_feat_full, N_mask)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        try:
            gt_points = neighborhood_ori[mask].reshape(B * M, -1, 3)
            # print(gt_points[0][0])
            loss1 = self.loss_func(rebuild_points, gt_points)
            # print(loss1)
            return loss1

        except:
            print('error')



# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.radius = config.radius

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.RI_encoder = RI_encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim*2, 512),
            # nn.Linear(self.trans_dim * self.depth, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        # cross-entropy with label smoothing
        eps = 0.2
        gt = gt.to(torch.int64)

        one_hot = torch.zeros_like(ret).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (self.cls_dim - 1)
        log_prb = F.log_softmax(ret, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()

        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            print('base_ckpt.keys()', base_ckpt.keys())

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        norm_ori = compute_LRA(pts)
        neighborhood, center, norm, center_norms, idx, idx_center = self.group_divider(pts, norm_ori)

        group_input_tokens = self.RI_encoder(neighborhood, center, center_norms, idx, pts, norm_ori)

        grouped_xyz = index_points(pts, idx)
        radius = calculate_radius(neighborhood)
        RI_global_feat = Global_feature(center, radius, grouped_xyz)
        pos = self.pos_embed(RI_global_feat)

        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        ret = self.cls_head_finetune(concat_f)

        return ret
