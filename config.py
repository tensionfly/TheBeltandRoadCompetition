import mxnet as mx
import numpy as np
class _Config:
    # Dataset config
    img_dir='data/train/'
    train_index='data/train_txt/train_index.txt'
    val_index='data/train_txt/val_index.txt'
    #device
    ctx=mx.cpu()
    #data mean st
    train_rgb_mean=np.array([0.46855228, 0.53813771, 0.62178215])
    train_rgb_st=np.array([0.18125395, 0.16496751, 0.1475986])

    test_rgb_mean=np.array([0.46610361, 0.53610539, 0.54132975])
    test_rgb_st=np.array([0.18099032, 0.16456958, 0.16787457])

    weight4cls=[1,1.3,2.7,7,2.8,1.7,2.7,3.7,3.4]
    #model attributions
    num_classes=9
    batch_size=16
    model_path='models/valiacc-{:.5}.gluonmodel'

cfg=_Config()