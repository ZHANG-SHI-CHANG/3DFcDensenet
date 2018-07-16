import tensorflow as tf
from tensorflow.python.ops import array_ops

class FCDenseNet():
    
    def __init__(self,num_classes=22,input_size=32,learning_rate=0.001):
        self.num_classes = num_classes
        self.input_size = input_size
        self.learning_rate = learning_rate
        
        self.__build()
    
    def __build(self):
        self.norm = 'batch_norm'#group_norm,batch_norm
        self.activate = 'relu'#selu,leaky,swish,relu,relu6,prelu
        self.DownDepth = [4,5,7,10,12]
        self.TransitionDepth = 15
        self.UpDepth = [12,10,7,5,4]
        self.growk = 16
    
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        
        with tf.variable_scope('zsc_feature'):
            x = PrimaryConv('PrimaryConv',self.input_image,48,self.norm,self.activate,self.is_training)
            
            DownList = []
            for i,depth in enumerate(self.DownDepth):
                _x = FCDenseBlock('DownFCDenseBlock_{}'.format(i),x,depth,self.growk,self.norm,self.activate,self.dropout,self.is_training)
                x = tf.concat([x,_x],axis=-1)
                DownList.append(x)
                x = DownTransition('DownTransition_{}'.format(i),x,self.norm,self.activate,self.dropout,self.is_training)
            
            x = FCDenseBlock('TransitionFCDenseBlock',x,self.TransitionDepth,self.growk,self.norm,self.activate,self.dropout,self.is_training)
            
            for i,UpDepth in enumerate(self.UpDepth):
                x = UpTransition('UpTransition_{}'.format(i),x)
                x = tf.concat([x,DownList[-(i+1)]],axis=-1)
                x = FCDenseBlock('UpFCDenseBlock_{}'.format(i),x,UpDepth,self.growk,self.norm,self.activate,self.dropout,self.is_training)
            
            self.classifier_logits = _conv_block('Conv',x,self.num_classes,1,1,'SAME',self.norm,self.activate,self.is_training)
            
            self.x = x
        self.__init__output()

    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = sum(tf.get_collection("regularzation_loss"))
            
            self.all_loss = tf.reduce_mean(
                                tf.reduce_sum(
                                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classifier_logits, labels=self.y, name='loss'),
                                             list(range(1,4)))
                                          )
            self.all_loss += regularzation_loss
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=1,decay_rate=0.995,staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_op = self.optimizer.minimize(self.all_loss)
            
            self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
            
            self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
            
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
            
            with tf.variable_scope('mean_iou_train'):
                label = tf.reshape(self.y,[tf.shape(self.y)[0],-1])
                predict = tf.reshape(self.y_out_argmax,[tf.shape(self.y_out_argmax)[0],-1])
                self.iou, self.update_op = tf.metrics.mean_iou(label, predict, self.num_classes)

            with tf.name_scope('train-summary-per-iteration'):
                tf.summary.scalar('loss', self.all_loss)
                tf.summary.scalar('acc', self.accuracy)
                tf.summary.scalar('iou', self.iou)
                self.summaries_merged = tf.summary.merge_all()
    def __init_input(self):
        with tf.variable_scope('input'):
            self.input_image = tf.placeholder(tf.float32,[None, self.input_size, self.input_size, self.input_size, 1],name='zsc_input')#N,D,H,W,C
            self.y = tf.placeholder(tf.int32, [None,self.input_size,self.input_size,self.input_size],name='zsc_input_target')#N,D,H,W
            self.dropout = tf.placeholder(tf.float32,name='zsc_dropout')
            self.is_training = tf.placeholder(tf.float32,name='zsc_is_train')
            self.is_training = tf.equal(self.is_training,1.0)
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

################################################################################################################
################################################################################################################
################################################################################################################
##Clique_conv
def FCDenseBlock(name,x,num_block=6,num_filters=80,norm='group_norm',activate='selu',dropout=0.2,is_training=True):
    with tf.variable_scope(name):
        FeatureList = []
        
        for i in range(num_block):
            _x = _B_conv_block('_B_conv_{}'.format(i),x,num_filters,3,1,'SAME',norm,activate,dropout,is_training)
            x = tf.concat([x,_x],axis=-1)
            FeatureList.append(_x)#0,1,2,3,4,5
            
        x = tf.concat(FeatureList,axis=-1)
        
        return x
##Transition
def DownTransition(name,x,norm='group_norm',activate='selu',dropout=0.2,is_training=True):
    with tf.variable_scope(name):
        C = x.get_shape().as_list()[-1]
        
        x = _B_conv_block('_B_conv_block',x,C,1,1,'SAME',norm,activate,dropout,is_training)
        
        x = tf.nn.max_pool3d(x,[1,4,4,4,1],[1,2,2,2,1],'SAME')
        return x
def UpTransition(name,x):
    with tf.variable_scope(name):
        N,D,H,W,C = x.get_shape().as_list()
        w = GetWeight('weight',[3,3,3,C,C])
        x = tf.nn.conv3d_transpose(x,w,tf.concat([[tf.shape(x)[0]],[2*tf.shape(x)[1]],[2*tf.shape(x)[2]],[2*tf.shape(x)[3]],[C]],axis=-1),[1,2,2,2,1],'SAME')
        return x
##primary_conv
def PrimaryConv(name,x,num_filters=48,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        #none,none,none,3
        x = _conv_block('Conv',x,num_filters,3,1,'SAME',None,None,is_training)
        return x
##_B_conv_block
def _B_conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',dropout=0.2,is_training=True):
    with tf.variable_scope(name):
        C = x.get_shape().as_list()[-1]
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x,training=is_training,epsilon=0.001,name='batch_norm')
        elif norm=='group_norm':
            x = group_norm(x,name='group_norm')
        else:
            pass
        if activate=='leaky':
            x = LeakyRelu(x,leak=0.1,name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x,name='prelu')
        else:
            pass
        
        w = GetWeight('weight',[kernel_size,kernel_size,kernel_size,C,num_filters])
        x = tf.nn.conv3d(x,w,[1,stride,stride,stride,1],'SAME')
        
        x = tf.layers.dropout(x, rate=dropout, training=is_training, name='dropout')
        
        return x
##_conv_block
def _conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        C = x.get_shape().as_list()[-1]
        w = GetWeight('weight',[kernel_size,kernel_size,kernel_size,C,num_filters])
        x = tf.nn.conv3d(x,w,[1,stride,stride,stride,1],'SAME')
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',num_filters,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x,name='prelu')
        else:
            pass
        
        return x
##weight variable
def GetWeight(name,shape,weights_decay = 0.0001):
    with tf.variable_scope(name):
        #w = tf.get_variable('weight',shape,tf.float32,initializer=VarianceScaling())
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)
def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out
class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = _compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)
##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
##selu
def selu(x,name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
##swish
def swish(x,name='swish'):
    with tf.variable_scope(name):
        beta = tf.Variable(1.0,trainable=True)
        return x*tf.nn.sigmoid(beta*x)
##crelu 注意使用时深度要减半
def crelu(x,name='crelu'):
    with tf.variable_scope(name):
        x = tf.concat([x,-x],axis=-1)
        return tf.nn.relu(x)
def prelu(inputs,name='prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg
################################################################################################################
################################################################################################################
################################################################################################################

if __name__=='__main__':
    import time
    from functools import reduce
    from operator import mul
    import numpy as np

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params
    
    input_size = 32
    model = FCDenseNet(num_classes=22,input_size=input_size)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
    
        feed_dict={model.input_image:np.random.randn(1,input_size,input_size,input_size,1),
                   model.y:np.array([np.random.randint(0,22) for _ in range(input_size*input_size*input_size)]).reshape((input_size,input_size,input_size))[np.newaxis,:,:,:].astype(np.int32),
                   model.is_training:1.0,
                   model.dropout:0.2}
        
        start = time.time()
        out = sess.run(model.iou,feed_dict=feed_dict)
        print('Spend Time:{}'.format(time.time()-start))
        
        print(out)
