from models.dgcnn import DGCNN
import numpy as np


if __name__=='__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    np.random.seed(0)
    input_feed = np.random.rand(batch_size, pos_dim, num_pt, 1)
    input_feed = np.around(input_feed).astype("float32")
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed>=0.5] = 1
    label_feed[label_feed<0.5] = 0
    label_feed = label_feed.astype(np.int32)

    model = DGCNN(40,pos_dim)

    print(model(input_feed,label_feed))
