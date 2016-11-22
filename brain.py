import tensorflow as tf

class Brain():
    def __init__(self,actions):
        self.actions=actions
        

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    
    def createNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])
    
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
    
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
    
        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])
    
        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])
    
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
    
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)
    
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)
    
        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    
        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    
        Q_max=tf.reduce_max(readout)
        tf.scalar_summary('Q_max', Q_max)
    
        return s, readout
    