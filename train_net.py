from config import cfg
from dataset import DataSet
from model import models
import mxnet as mx
from mxnet import nd
import math
import numpy as np
import time
from tqdm import tqdm
import os

from gluoncompetition.data.data_augmentation import ImageRandomFlipRotate,ColorNormalize
from gluoncompetition.loss.loss import *
from gluoncompetition.metrics.metric_methods import metric_confusion_matrix
# from gluoncompetition.learning_rate.find_lr import find_lr

def loss_operation(pred,gt):
    loss_ce=softmax_celoss_with_weight(pred,gt,cfg.weight4cls)
    return loss_ce

def test_transformation(data,label):
    data=ColorNormalize(data/255.0,cfg.train_rgb_mean,cfg.train_rgb_st)
    data=np.transpose(data,(2,0,1))
    return nd.array(data).astype('float32'),nd.array([label]).asscalar().astype('float32')

def train_transformation(data, label):
    data=ColorNormalize(data/255.0,cfg.train_rgb_mean,cfg.train_rgb_st)
    data=ImageRandomFlipRotate(data)
    data = nd.array(data).astype('float32')
    if np.random.uniform() > 0.5:
        aug1 = mx.image.RandomCropAug([90,90])
        data = aug1(data)
        data = mx.image.imresize(data, 100, 100)
    data=data.transpose((2,0,1))

    return data,nd.array([label]).asscalar().astype('float32')


val_dataset = DataSet(img_dir=cfg.img_dir,
                        dataset_index=cfg.val_index,
                        transform=test_transformation)
val_datait = mx.gluon.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)


train_dataset = DataSet(img_dir=cfg.img_dir,
                        dataset_index=cfg.train_index,
                        transform=train_transformation)
train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

ctx=cfg.ctx

net=models('Inception3',cfg.num_classes)
net.init_params(ctx)
net.hybridize()

trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate' :math.pow(10,-3) ,'momentum': 0.9})
# find_lr(net,trainer,loss_operation,train_datait,1e-10,0.1,100)

lr_scheduler=mx.lr_scheduler.MultiFactorScheduler(step=[120,140], factor=0.1, base_lr=0.001, warmup_steps=100, 
                                            warmup_begin_lr=1e-6, warmup_mode='linear')
warmup_steps=lr_scheduler.warmup_steps
    
for e in range(50):
    assert warmup_steps<=len(train_datait),\
    'please set warmup_steps that is not bigger than len(train_data)'

    acc = mx.metric.Accuracy()
    train_loss=0.
    confusion_matrix=nd.zeros((cfg.num_classes,cfg.num_classes),ctx=ctx)

    if e>0:
        trainer.set_learning_rate(lr_scheduler(e+warmup_steps-1))

    for i,(data,label) in enumerate(train_datait):

        if e==0:
            if i>=warmup_steps-1:
                trainer.set_learning_rate(lr_scheduler(warmup_steps-1))
            else:
                trainer.set_learning_rate(lr_scheduler(i))
        
        data = data.as_in_context(ctx) 
        label = label.as_in_context(ctx)
        
        with mx.autograd.record():
            output=net(data)
            loss=loss_operation(output,label)
            
        loss.backward()
        trainer.step(data.shape[0],ignore_stale_grad=True)

        train_loss+=loss.mean().asscalar()
        acc.update(preds=output, labels=label)
        print('epoch: %d, iter: %d, acc: %.4f'%(e,i,acc.get()[1]))

        out_argmax=output.argmax(axis=1)
        label=label.reshape((-1,)).asnumpy().astype('int32')
        out_argmax=out_argmax.reshape((-1,)).asnumpy().astype('int32')
        for i,j in zip(label,out_argmax):
            confusion_matrix[i,j]+=1
        

    sum_confusion_matrix=confusion_matrix.sum(axis=1).reshape((-1,1))
    confusion_matrix_ratio=confusion_matrix*(1.0/sum_confusion_matrix)

    evalu_confusion_matrix_ratio,evalu_confusion_matrix=metric_confusion_matrix(val_datait,net4img,cfg.num_classes,ctx)
    eye_nd=nd.eye(cfg.num_classes,ctx=ctx)

    train_acc=(confusion_matrix_ratio*eye_nd).sum().asscalar()
    val_acc=(evalu_confusion_matrix_ratio*eye_nd).sum().asscalar()
    train_loss_scalar=train_loss/len(train_datait)

    print('epoch: %d , loss: %.4f , train_acc: %.4f , val_acc: %.4f'%(e,train_loss_scalar,train_acc,val_acc))

    with open('evaluate_result.txt','a+') as f:
        f.write('epoch: %d , loss: %.4f , train_acc: %.4f , val_acc: %.4f\n'
                %(e,train_loss_scalar,train_acc,val_acc))

        f.write('train:\n')
        f.write(str(confusion_matrix_ratio)+'\n')
        f.write(str(confusion_matrix)+'\n')
        
        f.write('val:\n')
        f.write(str(evalu_confusion_matrix_ratio)+'\n')
        f.write(str(evalu_confusion_matrix)+'\n')
