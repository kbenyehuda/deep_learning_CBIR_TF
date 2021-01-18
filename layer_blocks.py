import tensorflow as tf

def conv_block(input_tensor,filter_dim,kernel,strides,leak = 0.2,padding = 'same'):
    '''
    Arguments :
    input_tensor : The input tensor .
    filter_dim : Number of filters created by the convolutional layer .
    training : Training parameter for batch normalization( differing between
    training and inference).
    kernel : Filter size for the convolutional filters .
    strides : Stride size for the convolutional filters .
    Description :
    Defines a convolution block which includes a convolutional layer , batch
    normalization layer , and a leaky ReLU activation function( for encoders).
    '''

    initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)

    h = tf.keras.layers.Conv2D(filter_dim, kernel, strides, padding=padding,
                                activation=None,kernel_initializer=initializer,bias_initializer='truncated_normal')(input_tensor)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU(0.5)(h)
    h = tf.keras.layers.MaxPool2D()(h)
    h = tf.keras.layers.SpatialDropout2D(0.2)(h)
    return h

def conv_block_decoder(input_tensor, filter_dim, kernel, strides ,padding =  'same'):
  '''
  Arguments :
  input_tensor : The input tensor .
  filter_dim : Number of filters created by the convolutional layer .
  training : Training parameter for batch normalization( differing between training and
  inference).
  kernel : Filter size for the convolutional filters .
  strides : Stride size for the convolutional filters .
  name : Name of layer .
  padding : padding type of the input to the convolution .
  Description :
  Defines a convolution block which includes a convolutional layer , batch
  normalization layer , and a ReLU activation function(for decoders).
  '''
  h = tf.keras.layers.Conv2D(filter_dim, kernel, strides, padding = padding,
                             activation = None, use_bias = True)(input_tensor)
  h = tf.keras.layers.BatchNormalization()(h)
  h = tf.keras.activations.relu(h)
  # h = tf.keras.layers.MaxPool2D()(h)
  h = tf.keras.layers.SpatialDropout2D(0.2)(h)
  return h

def deconv_block(input_tensor, filter_dim, padding = 'same',add_relu=True):
  '''
  Arguments :
  input_tensor : The input tensor .
  filter_dim : Number of filters created by the convolutional layer .
  training : Training parameter for batch normalization( differing between training and
  inference).
  name : Name of layer .
  padding : padding type of the input to the convolution .
  Description :
  Defines a deconvolution block which includes a deconvolutional layer , batch
  normalization layer , and a ReLU activation function(for the decoder).
  '''

  h = tf.keras.layers.Conv2DTranspose(filter_dim, 4, 2, padding = padding, activation = None, use_bias = False)(input_tensor)
  h = tf.keras.layers.BatchNormalization(epsilon =1e-5, momentum =0.1,
                                         gamma_initializer = tf.random_normal_initializer(1.0 , 0.02))(h)
  if add_relu:
    h = tf.keras.activations.relu(h)
  return h