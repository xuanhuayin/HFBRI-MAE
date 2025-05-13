import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation
from pytorch3d.common.workaround import symeig3x3

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

def RI_features(xyz, centers, center_norms, idx, pts, norm_ori):
    B, S, C = centers.shape

    new_norm = center_norms.unsqueeze(-1)
    dots_sorted, idx_ordered = order_index(xyz, centers, new_norm, idx)

    epsilon = 1e-7
    grouped_xyz = index_points(pts, idx_ordered)  # [B, npoint, nsample, C]
    # print('neighborhood.shape', xyz.shape)
    # print('norm.shape', norm.shape)
    # print('idx', idx.shape)

    grouped_xyz_local = grouped_xyz - centers.view(B, S, 1, C) # treat orgin as center
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

def calculate_radius(grouped_xyz):
    """
    Calculate the average radius for the grouped points.
    Input:
        grouped_xyz: grouped point cloud data, [B, N, k, 3]
    Return:
        radius: average radius, scalar
    # """
    # distances = torch.norm(grouped_xyz, dim=-1)  # [B, N, k]
    # radius = torch.mean(distances)
    dists = torch.norm(grouped_xyz, dim=-1)  # [B, npoint, nsample]

    # Find the maximum distance in each group
    max_dists = dists.max(dim=2)[0]  # [B, npoint]

    # Calculate the average of the maximum distances
    radius = max_dists.mean()
    return radius


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


def Global_feature(new_xyz, radius, grouped_xyz):
    # grouped_xyz = index_points(pts, idx)
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


def sample_and_group(xyz, centers, norm, center_norms, idx, pts, norm_ori):
    """
    Input:
        npoint: number of new points
        radius: radius for each new points
        nsample: number of samples for each new point
        xyz: input points position data, [B, M, K, 3]
        centers: input center points position data, [B, M, 3]
        norm: input normal vector, [B, M，K, 3]
        center_norms: input centers normal vector, [B, M, 3]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        ri_feat: sampled ri attributes, [B, npoint, nsample, 8]
        new_norm: sampled norm data, [B, npoint, 3]
        idx_ordered: ordered index of the sample position data, [B, npoint, nsample]
    """
    xyz = xyz.contiguous()
    norm = norm.contiguous()

    # new_xyz, new_norm = sample(npoint, xyz, norm=norm, sampling='fps')
    # idx = group_index(nsample, radius, xyz, new_xyz, group='knn')

    ri_feat, idx_ordered = RI_features(xyz, centers, center_norms, idx, pts, norm_ori)

    return ri_feat, idx_ordered

class RIConv2SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, encoder_channel):
        super(RIConv2SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.prev_mlp_convs = nn.ModuleList()
        self.prev_mlp_bns = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.encoder_channel = encoder_channel

        in_channel_0 = 8
        mlp_0 = [64, 128]
        last_channel = in_channel_0
        for out_channel in mlp_0:
            self.prev_mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.prev_mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        last_channel = mlp_0[-1]
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel


        self.group_all = group_all

    def forward(self, xyz, centers, norm, center_norms, idx, pts, norm_ori):
        """
        Input:
            xyz: input points position data, [B, M, K, 3]
            centers: input center points position data, [B, M, 3]
            norm: input normal vector, [B, M，K, 3]
            center_norms: input centers normal vector, [B, M, 3]
            points: input points (feature) data, [B, C, N]
            idx: input index data, [B, M, K]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_norm: sample points normal data, [B, S, 3]
            ri_feat: created ri features, [B, C, S]
        """



        # B, N, _, C = xyz.shape


        ri_feat, idx_ordered = sample_and_group(xyz, centers, norm, center_norms, idx, pts, norm_ori)

        # bs, g, n, _ = ri_feat.shape
        # point_groups = ri_feat.reshape(bs * g, n, 8)
        # # encoder
        # feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        # feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        # feature = self.second_conv(feature)  # BG 1024 n
        # feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024

        # lift
        ri_feat = ri_feat.permute(0, 3, 2, 1)  # B, 8, K, N
        for i, conv in enumerate(self.prev_mlp_convs):
            bn = self.prev_mlp_bns[i]
            ri_feat = F.relu(bn(conv(ri_feat)))

        # print('ri_feat.shape', ri_feat.shape)

        # concat previous layer features

        new_points = ri_feat

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        ri_feat = torch.max(new_points, 2)[0]  # maxpooling  B, 384, N

        return ri_feat.permute(0, 2, 1)

class RI_encoder(nn.Module):
    def __init__(self, encoder_channel):
        super(RI_encoder, self).__init__()
        in_channel = 64
        self.encoder_channel = encoder_channel
        # self.normal_channel = normal_channel

        self.sa0 = RIConv2SetAbstraction(npoint=1024, radius=0.12, nsample=16, in_channel=0 + in_channel, mlp=[self.encoder_channel],
                                         group_all=False, encoder_channel=self.encoder_channel)

    def forward(self, xyz, centers, norm, center_norms, idx, pts, norm_ori):
        B, _, _ ,_ = xyz.shape
        RI_feat = self.sa0(xyz, centers, norm, center_norms, idx, pts, norm_ori)

        return RI_feat

def compute_LRA(xyz, weighting=False, nsample = 64):
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
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    # eigen_values, vec = M.symeig(eigenvectors=True)

    LRA = vec[:,:,:,0]

    B, N, _ = xyz.shape
    inner_product = (LRA*xyz).sum(-1)
    mask = torch.ones((B, N), device=LRA.device)
    mask[inner_product < 0] = -1
    LRA = LRA * mask.unsqueeze(-1)
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

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

def farthest_point_sample(xyz, npoint, norm = None):
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

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, norm = None, radius = None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out

        center, norm_center, idx_new = farthest_point_sample(xyz, self.num_group, norm)  # B G 3

            # center, idx_new = farthest_point_sample(xyz, self.group_size)
        # knn to get the neighborhood
        # idx, _ = query_ball_point(radius, self.group_size, xyz, center)  # B G M
        idx = knn(xyz, idx_new, self.group_size)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        # print('idx.shape', idx.shape)
        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx1 = idx + idx_base
        # idx1 = idx1.view(-1)
        # neighborhood = xyz.view(batch_size * num_points, -1)[idx1, :]
        # neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = index_points(xyz, idx)
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # norm = norm.view(batch_size * num_points, -1)[idx1, :]

        # # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        norm = index_points(norm, idx)
        norm = norm.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        return neighborhood, center, norm, norm_center, idx, idx_new


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


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

        attn = (q * self.scale) @ k.transpose(-2, -1)
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
    """ Transformer Encoder without hierarchical structure
    """

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
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class LRF(nn.Module): # Local Reference Frames
    def __init__(self, axis1="pca", axis2="pca"):
        super().__init__()
        self.axis1 = axis1
        self.axis2 = axis2

    def disambiguate_vector_directions(self, lps, vecs) :
        # disambiguate sign of normals in the SHOT manner
        # the codes below are borrowed from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_normals.html
        # lps: local point sets [B,G,M,3]
        # vecs: reference vectors [B,G,3]
        knn = lps.shape[2]
        proj = ( vecs[:, :, None] * lps ).sum(3) # projection of the difference on the principal direction
        n_pos = (proj > 0).to(torch.float32).sum(2, keepdim=True) # check how many projections are positive
        # flip the principal directions where number of positive correlations
        flip = (n_pos < (0.5 * knn)).to(torch.float32) # 0 or 1
        vecs = (1.0 - 2.0 * flip) * vecs # convert to -1 or 1 before multiplication with vecs
        return vecs

    def forward(self, neighbor, center):
        '''
            input:
            neighbor: B G S 3or6 (Local point sets, whose coordinates are normalized for each local region.)
            center: B G 3or6
            ---------------------------
            outputs
            rot_neighbor: B G S 3or6 (Rotation-normalized local point sets.)
            lrf : B G 3 3 (Local reference frames)
        '''
        B, G, S, C = neighbor.shape # B: batch_size, G: num_group, S: group_size
        pos = neighbor[ :, :, :, 0:3 ]

        if( C == 3 or self.axis1 == "pca" or self.axis2 == "pca" ): # in the case that PCA is necessary
            # generate covariance matrices
            norms = torch.linalg.norm( pos, dim=3, keepdims=True )
            max_norms, _ = torch.max( norms, dim=2, keepdims=True )
            w = max_norms - norms
            w = w / ( torch.sum( w, dim=2, keepdims=True ) + 1e-6 )
            scaled_pos = 100.0 * pos # for numerical stability
            covs = torch.einsum( "bijk,bijl->bikl", w * scaled_pos, scaled_pos )

            # There are multiple functions for eigen value decomposition
            # Option 1
            # _, _, eigvec = torch.linalg.svd( covs, full_matrices=False )
            # eigvec = torch.flip( eigvec, dims=[2]).permute(0,1,3,2) # create same format as torch.linalg.eigh
            # Option 2
            # _, eigvec = torch.linalg.eigh( covs )
            # Option 3
            _, eigvec = symeig3x3( covs, eigenvectors=True )

            # eigvec: [B, , 3, 3], where [:, i, :, 0] corresponds to the normal vector for the local point set i

        # Compute the first axis (z_axis)
        if( C == 3 or self.axis1 == "pca" ):
            # z_axis is a surface normal estimated by PCA
            z_axis = self.disambiguate_vector_directions( pos, eigvec[ :, :, :, 0 ] )
            axis1_pca = True
        elif( self.axis1 == "normal" ):
            # z_axis is a true surface normal computed from polygonal 3D shape
            z_axis = neighbor[ :, :, 0, 3:6 ] # In the "neighbor" tensor, center point always locates at the 0-th in the third axis
            axis1_pca = False

        # Compute the second axis (x_axis)
        if( self.axis2 == "pca" ):
            x_axis = eigvec[ :, :, :, 2 ] # eigen vectors associated with the largest eigen values
            if( not axis1_pca ): # need to orthogonalize
                # each principal axis is projected onto the tangent plane of a z-axis
                dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
                x_axis = x_axis - dot * z_axis
                x_axis = F.normalize( x_axis, dim=2 )
            x_axis = self.disambiguate_vector_directions( pos, x_axis )
        elif( self.axis2 == "mean" ):
            x_axis = torch.mean( pos, axis=2 ) # subtraction by center is not necessary since the center coordinates are always (0,0,0).
            norm = torch.linalg.norm( x_axis, axis=2, keepdim=True )
            x_axis = x_axis / ( norm + 1e-6 )
            # each mean vector is projected onto the tangent plane of a z-axis
            dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
            x_axis = x_axis - dot * z_axis
            x_axis = F.normalize( x_axis, dim=2 )

        # Compute the third axis (y-axis), which is just a cross between z and x
        y_axis = torch.cross( z_axis, x_axis, dim=2 )

        # cat to form the set of principal directions
        lrfs = torch.stack( ( z_axis, y_axis, x_axis ), dim=3 )

        # normalize orientation of local point sets
        pos = torch.reshape( pos, [ B * G, S, 3 ] )
        pos = torch.bmm( pos, torch.reshape( lrfs, [ B * G, 3, 3 ] ) )
        pos = torch.reshape( pos, [ B, G, S, 3 ] )

        if( C == 3 ):
            rot_neighbor = pos
        elif( C == 6 ):
            ori = neighbor[ :, :, :, 3:6 ]
            ori = torch.reshape( ori, [ B * G, S, 3 ] )
            ori = torch.bmm( ori, torch.reshape( lrfs, [ B * G, 3, 3 ] ) )
            ori = torch.reshape( ori, [ B, G, S, 3 ] )
            rot_neighbor = torch.cat( [ pos, ori ], dim=3 )

        return rot_neighbor, lrfs


def reverse_ordered_features(pts, idx_ordered, ri_features, pts_global_feats):
    """
    通过 idx_ordered 将 ri_features 和 pts_global_feats 反向映射到与原始点云 pts 对应的特征

    参数:
    pts: 原始点云数据, [B, N, 3]
    idx_ordered: 索引数据, [B, G, K]
    ri_features: 局部特征数据, [B, G, K, 8]
    pts_global_feats: 全局特征数据, [B, G, 5]

    返回:
    pts_features: 反向映射后的特征数据, [B, N, 13]
    """
    B, N, _ = pts.shape
    _, G, K, C_ri = ri_features.shape  # C_ri 应该是 8
    _, _, C_global = pts_global_feats.shape  # C_global 应该是 5

    # 初始化空的 pts_features 和 pts_counts
    pts_features = torch.zeros((B, N, C_ri + C_global), device=pts.device)  # [B, N, 13]
    pts_counts = torch.zeros((B, N), device=pts.device)  # [B, N]

    # 复制 pts_global_feats，并与 ri_features 在最后一个维度上拼接
    pts_global_feats_expanded = pts_global_feats.unsqueeze(2).repeat(1, 1, K, 1)  # [B, G, K, 5]
    combined_features = torch.cat([ri_features, pts_global_feats_expanded], dim=-1)  # [B, G, K, 13]

    # 反向映射并累加特征
    for i in range(G):
        # 根据 idx_ordered 将 combined_features 累加到对应的 pts_features 中
        pts_features.scatter_add_(
            1,
            idx_ordered[:, i, :].unsqueeze(-1).expand(-1, -1, C_ri + C_global),
            combined_features[:, i, :, :]
        )
        # 记录每个点被访问的次数
        pts_counts.scatter_add_(
            1,
            idx_ordered[:, i, :],
            torch.ones_like(idx_ordered[:, i, :], device=pts.device, dtype=torch.float32)
        )

    # 归一化特征，将累加的特征除以访问次数
    pts_features = pts_features / pts_counts.unsqueeze(-1).clamp(min=1)

    return pts_features


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def get_k_nn(xyz1, xyz2, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)

    dists, inds = dists[:, :, :k], inds[:, :, :k]
    return dists, inds


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]



def interpolate(xyz1, xyz2, feature, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)   N1>N2
    :param feature: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = feature.shape
    dists, inds = get_k_nn(xyz1, xyz2, k)

    # inversed_dists = 1.0 / (dists + 1e-8)
    #
    # weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    #
    # weight = torch.unsqueeze(weight, -1)

    interpolated_feature = gather_points(feature, inds)  # shape=(B, N1, 3, C2)

    # return interpolated_feature, inds, weight
    return interpolated_feature, inds

# ref: PaRot: Patch-Wise Rotation-Invariant Network via Feature Disentanglement and Pose Restoration
# based on: https://github.com/dingxin-zhang/PaRot
class FP_Module_angle(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_angle, self).__init__()

        dim_posembed = 32
        self.posembed = nn.Sequential(
            nn.Conv2d( 3+1, dim_posembed, kernel_size=1, bias=False),
            nn.BatchNorm2d( dim_posembed ),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.backbone = nn.Sequential()
        bias = False if bn else True

        in_channels = in_channels + dim_posembed
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz1, xyz2, feat2, lrf2, k=3):

        B, N1, _ = xyz1.shape
        _, N2, C2 = feat2.shape

        interpolated_feature, inds = interpolate(xyz1, xyz2, feat2, k) # get features of neighboring points

        lrf2 = lrf2.reshape( B, N2, 9 )
        close_lrf = gather_points( xyz2, inds )
        lrf2 = gather_points( lrf2, inds ).view(-1, 3, 3)

        relate_position = xyz1.unsqueeze(2).repeat(1, 1, k, 1) - close_lrf

        for_dot = F.normalize(relate_position.view(-1, 3), dim=-1).unsqueeze(2)
        angle = lrf2.matmul(for_dot)
        angle = angle.view(B, N1, k, -1)

        relative_pos = torch.cat((torch.norm(relate_position, dim=-1, keepdim=True), angle), dim=3)
        pos = self.posembed(relative_pos.permute(0, 3, 2, 1))
        interpolated_feature = interpolated_feature.permute(0, 3, 2, 1)
        cat_interpolated_points = torch.cat((interpolated_feature, pos), dim=1)

        new_points = self.backbone(cat_interpolated_points)
        new_points = torch.sum(new_points, dim=2)

        return new_points



class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 8

        self.group_size = 64
        self.num_group = 256
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        # self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.RI_encoder = RI_encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

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
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 13,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(3392,  512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.lrf_estimator = LRF(axis1 = 'pca', axis2 = 'mean')
        self.partseg_propagation = FP_Module_angle(in_channels=1152,
                                                   mlp=[256 * 2, 256])

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        norm_ori = compute_LRA(pts)
        neighborhood, center, norm, center_norms, idx, idx_center = self.group_divider(pts, norm_ori)
        grouped_xyz = index_points(pts, idx)
        radius = calculate_radius(neighborhood)
        RI_global_feat = Global_feature(center, radius, grouped_xyz)
        local_features, idx_ordered = RI_features(neighborhood, center, center_norms, idx, pts, norm_ori)

        pts_feats = reverse_ordered_features(pts, idx_ordered, local_features, RI_global_feat)
        # print('sssssssssssssssssssssssssssslrfshape', lrf.shape)
        group_input_tokens = self.RI_encoder(neighborhood, center, norm, center_norms, idx, pts, norm_ori)
        # divide the point clo  ud in the same form. This is important
        # neighborhood, center = self.group_divider(pts)

        # group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(RI_global_feat)

        # pos = self.pos_embed(center)
        # final input
        x = group_input_tokens
        # transformer
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]

        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) #1152*2 + 64
        # f_level_0 = self.partseg_propagation(pts[:, :, 0:3], center[:, :, 0:3],
        #                                      x.transpose(-1, -2), lrf)
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts_feats.transpose(-1, -2), x)



        x = torch.cat((f_level_0,x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss