import numpy as np
import tensorflow as tf
import os
import datetime

class test_Densenet_hotel:

  def __init__(self,test_t_dir,test_d_dir,restore_dir,image_dim,N_CLASSES,train_BATCH_SIZE,test_BATCH_SIZE,CAPACITY,learning_rate):

    image_list2, label_list2 =self.get_stool_files(test_t_dir,test_d_dir)

    self.run_model(image_list2,label_list2,restore_dir, image_dim, N_CLASSES, train_BATCH_SIZE,test_BATCH_SIZE, CAPACITY,learning_rate)

  def get_stool_files(self,file_dir1,file_dir2):
    image_t_list = []
    label_t_list = []
    image_d_list = []
    label_d_list = []
    for file in os.listdir(file_dir1):
       image_t_list.append(file_dir1 + file)
       label_t_list.append(0)
    for file in os.listdir(file_dir2):
       image_d_list.append(file_dir2 + file)
       label_d_list.append(1)

    image_list = np.hstack((image_t_list, image_d_list))
    label_list = np.hstack((label_t_list, label_d_list))
    temp = np.array([image_list, label_list])

    temp = temp.transpose()     
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

  def get_batch(self,image, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)

    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)   
    return image_batch, label_batch

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def conv2d(self,input, in_features, out_features, kernel_size, s, with_bias=False): #s: strides
    W = self.weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, s, s, 1 ], padding='SAME')
    if with_bias:
      return conv + self.bias_variable([ out_features ])
    return conv

  def batch_activ_conv(self,current, in_features, out_features, kernel_size, s, is_training, keep_prob):#s: strides
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = self.conv2d(current, in_features, out_features, kernel_size,s) 
    current = tf.nn.dropout(current, keep_prob)
    return current

  def block(self,input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in xrange(layers):
      #tmp = self.batch_activ_conv(current, features, 4*growth, 1, is_training, keep_prob)   #bottleneck layer
      tmp = self.batch_activ_conv(current, features, growth, 3, 1, is_training, keep_prob) #ksize=3 strides=1
      current = tf.concat((current, tmp), axis=3)
      features += growth
    return current, features

  def transition_layers(self,current, in_features, out_features, ksize, s, is_training, keep_prob):#s: strides
    current = self.batch_activ_conv(current, in_features, out_features, ksize, s, is_training, keep_prob) 
    current = self.avg_pool(current, 2)#ksize=2 strides=2
    return current
    
  def avg_pool(self,input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

  def run_model(image_list2,label_list2,restore_dir, image_dim, N_CLASSES, test_BATCH_SIZE, CAPACITY,learning_rate):

    numpic=len(image_list2)
    image_batch2, label_batch2 = self.get_batch(image_list2, label_list2, image_dim, image_dim, test_BATCH_SIZE, CAPACITY)

    weight_decay = 1e-4
    layers = 7

    xs = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, 3])
    ys= tf.placeholder(tf.int64, shape=[None])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    current = self.conv2d(xs, 3, 16, 11, 3) # ksize=11 srides=3
    #---Dense block1---#
    current, features = self.block(current, layers, 16, 12, is_training, keep_prob)
    #---Transition layers---#
    current = self.transition_layers(current, features, 48, 1, 1, is_training, keep_prob) #output 48 ksize=1 srides=1
    #---Dense block2---#
    current, features = self.block(current, layers, 48, 12, is_training, keep_prob)
    #---Transition layers---#
    current = self.transition_layers(current, features, 96, 1, 1, is_training, keep_prob) #output 96 ksize=1 srides=1
    #---Dense block3---#
    current, features = self.block(current, layers, 96, 12, is_training, keep_prob)
    #---Transition layers---#
    current = self.transition_layers(current, features, 150, 1, 1, is_training, keep_prob) #output 150 ksize=1 srides=1
    #---Dense block4---#
    current, features = self.block(current, layers, 150, 12, is_training, keep_prob)
    #---Transition layers---#
    current = self.transition_layers(current, features, 192, 1, 1, is_training, keep_prob) #output 192 ksize=1 srides=1
    #---Dense block4---#
    current, features = self.block(current, layers, 192, 12, is_training, keep_prob)
    #---global average pooling---#
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    last_pool_kernel = int(current.get_shape()[-2])
    current = self.avg_pool(current, last_pool_kernel)

    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])

    Wfc = self.weight_variable([ final_dim, N_CLASSES ])
    bfc = self.bias_variable([ N_CLASSES ])
    ys_ = tf.add(tf.matmul(current, Wfc), bfc)

    correct_prediction = tf.equal(tf.argmax(ys_,1), ys)
    test_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    saver=tf.train.Saver()
    model_file=tf.train.latest_checkpoint(restore_dir)
    saver.restore(sess,model_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    count_img=0
    count_acc=0.0
    
    f=open('aaaa', 'a')
    date=datetime.datetime.now().strftime('%Y-%m-%d')
    f.write(date+'\n') 

    try:
      for step in np.arange(numpic-1):
        if coord.should_stop() :
          break
        test_images,test_labels = sess.run([image_batch2, label_batch2])
        t_acc=sess.run(test_acc, feed_dict={xs:test_images, ys:test_labels, is_training: False,keep_prob:1.})
        if(t_acc==1):
          count_acc+=1
          print("step: %d, recognised successfully"%(step+1))
        else:
          print("step: %d, recognised unsuccessfully"%(step+1))
          now_time = datetime.datetime.now().strftime('%H:%M:%S')
          f.write(now_time+' '+image[step+1]+'\n') #save unrecognised image path
      acc_rate=count_acc/numpic
      print("Accuracy rate: %.2f"%(acc_rate))
    except tf.errors.OutOfRangeError:
      print("done!")
    finally:
      coord.request_stop()
    coord.join(threads)


N_CLASSES=2
test_BATCH_SIZE = 1
CAPACITY = 256
image_dim = 640 
learning_rate=0.0001

test_t_dir='/mnt/disk1/lvsikai/test/bed/standard/'
test_d_dir='/mnt/disk1/lvsikai/test/bed/nonstandard/'
restore_dir='save/bed/'

Densenet_hotel(test_t_dir,test_d_dir,restore_dir,image_dim,N_CLASSES,train_BATCH_SIZE,test_BATCH_SIZE,CAPACITY,learning_rate)
