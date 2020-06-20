import gluoncv
import mxnet as mx

class models(mx.gluon.Block):

    modle_dic={'AlexNet':gluoncv.model_zoo.AlexNet,
                'Inception3':gluoncv.model_zoo.Inception3,
                'MobileNetV2':gluoncv.model_zoo.MobileNetV2,
                'SqueezeNet':mx.gluon.model_zoo.vision.SqueezeNet,
                'densenet121':mx.gluon.model_zoo.vision.densenet121
                }
    
    def __init__(self,model, num_classes, **kwargs):
        super(models, self).__init__(**kwargs)

        with self.name_scope():
            #image
            self.feature_extractor = self.modle_dic[model]().features
            self.img_dense=mx.gluon.nn.Dense(64,activation='relu')
            #txt
            self.txt_dense0=mx.gluon.nn.Dense(256,activation='relu')
            self.txt_dense1=mx.gluon.nn.Dense(128,activation='relu')
            self.txt_dense2=mx.gluon.nn.Dense(64,activation='relu')
            #mix
            self.mix_dense=mx.gluon.nn.Dense(num_classes)

    def forward(self,data,*args):
        data0=data[0]
        data1=data[1]
        # img
        img=self.feature_extractor(data0)
        img=self.img_dense(img)
        #txt
        txt1=self.txt_dense0(data1)
        txt2=self.txt_dense1(txt1)
        txt3=self.txt_dense2(txt2)
        #mix
        img_txt=mx.nd.concat(img,txt3,dim=1)
        mix=self.mix_dense(img_txt)
    
        return mix
        
    def init_params(self, ctx):
        self.collect_params().initialize(mx.init.Normal(), ctx=ctx)
