import chainer
from chainer import functions as F
from chainer.functions.math import minmax
from chainer import links
import numpy as np
import cupy as cp
import copy 

def edge_conv(x,k,gpu=False):
    point_cloud=F.transpose(x,(0,2,1,3))
    adj_matrix = pairwise_distance(point_cloud)
    nn_idx = knn(adj_matrix, k=k, gpu=gpu)
    edge_feature = get_edge_feature(point_cloud, nn_idx=nn_idx, k=k,gpu=gpu)
    edge_feature = F.transpose(edge_feature,(0,3,1,2))
    return edge_feature

def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_dims, num_points)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0]
    point_cloud = F.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = F.expand_dims(point_cloud, 0)
        
    point_cloud_transpose = F.transpose(point_cloud,axes=[0,2,1])
    point_cloud_inner = F.matmul(point_cloud,point_cloud_transpose)
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = F.sum(F.square(point_cloud), axis=-1, keepdims=True)
    point_cloud_square_tranpose = F.transpose(point_cloud_square, axes=[0,2,1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(x, k=20, axis=None, gpu=False):
    x_arr = x.array
    """
    if x.data.dtype == cp.float64 or x.data.dtype == cp.float32:
        res = cp.argpartition(x_arr,kth=k)
    elif x.data.dtype == np.float64 or x.data.dtype == np.float32:
        res = np.argpartition(x_arr,kth=k)
    """
    if gpu:
        res = cp.argpartition(x_arr,kth=k-1)
    else:
        res = np.argpartition(x_arr,kth=k-1)

    return res[:,:,0:k]

def get_edge_feature(point_cloud, nn_idx, k=20, gpu=False):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.shape[0]
    point_cloud = F.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = F.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    num_dims = point_cloud.shape[2]

    if gpu:
        idx_ = cp.arange(batch_size) * num_points
    else:
        idx_ = np.arange(batch_size) * num_points
    
    idx_ = F.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = F.reshape(point_cloud, [-1, num_dims])
    pair_arr = nn_idx+idx_
    point_cloud_neighbors = point_cloud_flat[pair_arr.data]
    point_cloud_neighbors = F.reshape(point_cloud_neighbors,(batch_size,num_points,k,num_dims))
    point_cloud_central = F.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = F.tile(point_cloud_central, (1, 1, k, 1))

    edge_feature = F.concat(
        [point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature
  
if __name__ == '__main__':
    np.random.seed(0)
    point_cloud = np.random.rand(4,80,3)
    point_cloud = np.array([
        [
            [0,4,6],
            [1,1,2],
            [2,4,4],
            [3,6,7],
            [0,9,0],
        ],
        [
            [0,9,0],
            [0,4,6],
            [2,4,4],
            [1,1,2],
            [3,6,7],
        ]
    ],dtype=float)
    point_cloud=F.transpose(point_cloud,(0,2,1))
    chainer_res = edge_conv(point_cloud,3)
    print(chainer_res)
