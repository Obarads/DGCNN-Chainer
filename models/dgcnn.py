import numpy as np
import math

import chainer
import chainer.functions as F
from chainer import links
from chainer import reporter
from chainer import backends

from .conv_block import ConvBlock
from .linear_block import LinearBlock
from .transform_net import TransformNet
from . import edge_conv_block as ec

def calc_trans_loss(t):
    # Loss to enforce the transformation as orthogonal matrix
    # t (batchsize, K, K) - transform matrix
    xp = backends.cuda.get_array_module(t)
    bs, k1, k2 = t.shape
    assert k1 == k2
    mat_diff = F.matmul(t, F.transpose(t, (0, 2, 1)))
    mat_diff = mat_diff - xp.identity(k1, dtype=xp.float32)
    # divide by 2. is to make the behavior same with tf.
    # https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/nn/l2_loss
    return F.sum(F.batch_l2_norm_squared(mat_diff)) / 2.

class DGCNN(chainer.Chain):
    def __init__(self, out_dim, in_dim=3, middle_dim=64, dropout_ratio=0.3,
                 use_bn=True, trans_lam1=0.001, compute_accuracy=True, 
                 residual=False, k=20):
        super(DGCNN, self).__init__()
        with self.init_scope():

            self.input_transform_net = TransformNet(
                k=in_dim, use_bn=use_bn, residual=residual)

            self.conv_block1 = ConvBlock(
                in_dim, 64, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block2 = ConvBlock(
                64, middle_dim, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block3 = ConvBlock(
                middle_dim, 64, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block4 = ConvBlock(
                64, 128, ksize=1, use_bn=use_bn, residual=residual)
            self.conv_block5 = ConvBlock(
                128, 1024, ksize=1, use_bn=use_bn, residual=residual)

            # original impl. uses `keep_prob=0.7`.
            self.fc_block6 = LinearBlock(
                1024, 512, use_bn=use_bn, dropout_ratio=dropout_ratio,)
            self.fc_block7 = LinearBlock(
                512, 256, use_bn=use_bn, dropout_ratio=dropout_ratio,)
            self.fc8 = links.Linear(256, out_dim)

        self.in_dim = in_dim
        self.trans_lam1 = trans_lam1
        self.compute_accuracy = compute_accuracy
        self.k = k

    def __call__(self, x, t):
        h, t1 = self.calc(x)
        cls_loss = F.softmax_cross_entropy(h,t)
        reporter.report({'cls_loss': cls_loss}, self)

        loss = cls_loss
        # Enforce the transformation as orthogonal matrix
        trans_loss1 = self.trans_lam1 * calc_trans_loss(t1)
        reporter.report({'trans_loss1': trans_loss1}, self)
        loss = loss + trans_loss1

        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
            acc = F.accuracy(h, t)
            reporter.report({'accuracy': acc}, self)
        return loss

    def calc(self, x):
        # --- input transform ---
        k = self.k
        edge_feature = ec.edge_conv(x,k)
        h, t1 = self.input_transform_net(edge_feature)

        h = ec.edge_conv(h,k)
        h = self.conv_block1(h)
        h = F.max(h, axis=2, keepdims=True)
        h1 = h

        h = ec.edge_conv(h,k)
        h = self.conv_block2(h)
        h = F.max(h, axis=2, keepdims=True)
        h2 = h

        h = ec.edge_conv(h,k)
        h = self.conv_block3(h)
        h = F.max(h, axis=2, keepdims=True)
        h3 = h

        h = ec.edge_conv(h,k)
        h = self.conv_block4(h)
        h = F.max(h, axis=2, keepdims=True)
        h4 = h

        h = ec.edge_conv(h,k)
        h = self.conv_block4(h)
        h = F.max(h, axis=2, keepdims=True)
        h4 = h

        h = self.conv_block5(F.concat((h1,h2,h3,h4)))
        h = F.max(h, axis=1, keepdims=True)
        
        h = self.fc_block6(h)
        h = self.fc_block7(h)
        h = self.fc8(h)

        return h, t1
