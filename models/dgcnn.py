import chainer
import chainer.functions as F
import numpy as np
import math
from conv_block import ConvBlock
from linear_block import LinearBlock
from transform_net import TransformNet
import edge_conv as ec

class DGCNN(chainer.Chain):
    def __init__(self, out_dim, in_dim=3, middle_dim=64, dropout_ratio=0.3,
                 use_bn=True, trans_lam1=0.001, trans_lam2=0.001,
                 compute_accuracy=True, residual=False, k=20):
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

        self.in_dim = in_dim
        self.trans_lam1 = trans_lam1
        self.k = k

    def __call__(self, x, y):
        h, t1 = self.calc(x)

    def calc(self, x):
        # --- input transform ---
        k = self.k
        edge_feature = ec.edge_conv(x,k)
        h, t1 = self.input_transform_net(edge_feature)

        h = ec.edge_conv(h,k)
        h = self.conv_block1(h)

        h = ec.edge_conv(h,k)
        h = self.conv_block2(h)

        h = ec.edge_conv(h,k)
        h = self.conv_block3(h)

        h = ec.edge_conv(h,k)
        h = self.conv_block4(h)

        h = ec.edge_conv(h,k)
        h = self.conv_block5(h)

        # Symmetric function: max pooling
        bs, k, n, tmp = h.shape
        assert tmp == 1
        h = F.max_pooling_2d(h, ksize=h.shape[2:])
        # h: (minibatch, K, 1, 1)

        return h, t1
