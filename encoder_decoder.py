import tensorflow as tf
from layer_blocks import *

def build_encoder(input_size):
    input_img = tf.keras.layers.Input(shape=input_size)
    layer_1 = conv_block(input_img,filter_dim=64,kernel=3,strides=1)
    layer_2 = conv_block(layer_1,filter_dim=128,kernel=3,strides=1)
    layer_3 = conv_block(layer_2,filter_dim=256,kernel=3,strides=1)
    layer_4 = conv_block(layer_3, filter_dim=512, kernel=3, strides=1)
    layer_5 = conv_block(layer_4, filter_dim=512, kernel=3, strides=1)
    layer_6 = conv_block(layer_5, filter_dim=512, kernel=3, strides=1)
    layer_7 = conv_block(layer_6, filter_dim=512, kernel=3, strides=1)
    layer_8 = conv_block(layer_7, filter_dim=512, kernel=3, strides=1)

    layer_9 = tf.keras.layers.Flatten()(layer_8)
    initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    layer_9 = tf.keras.layers.Dense(4,kernel_initializer=initializer,bias_initializer='truncated_normal')(layer_9)
    # layer_9 = tf.keras.layers.BatchNormalization()(layer_9)
    encoded = tf.keras.layers.LeakyReLU(0.2)(layer_9)

    encoder = tf.keras.models.Model(inputs=input_img, outputs=encoded)
    return encoder,encoded,input_img

def build_decoder(inp_img,encoded):
    inp = tf.keras.layers.Input(shape=encoded.shape)
    layer_1 = tf.keras.layers.Dense(2048,activation='relu',kernel_initializer='he_uniform')(inp)
    layer_2 = tf.keras.layers.Reshape((2,2,512))(layer_1)
    layer_3 = conv_block_decoder(layer_2,filter_dim=512,kernel=3,strides=1)
    layer_3 = deconv_block(layer_3,512)
    layer_4 = conv_block_decoder(layer_3, filter_dim=256, kernel=3, strides=1)
    layer_4 = deconv_block(layer_4, 256)
    layer_5 = conv_block_decoder(layer_4, filter_dim=128, kernel=3, strides=1)
    layer_5 = deconv_block(layer_5, 128)
    layer_6 = conv_block_decoder(layer_5, filter_dim=64, kernel=3, strides=1)
    layer_6 = deconv_block(layer_6, 64)
    layer_7 = conv_block_decoder(layer_6, filter_dim=32, kernel=3, strides=1)
    layer_7 = deconv_block(layer_7, 32)
    layer_8 = conv_block_decoder(layer_7, filter_dim=16, kernel=3, strides=1)
    layer_8 = deconv_block(layer_8, 16)
    layer_9 = conv_block_decoder(layer_8, filter_dim=8, kernel=3, strides=1)
    layer_9 = deconv_block(layer_9, 8)
    layer_10 = conv_block_decoder(layer_9, filter_dim=4, kernel=3, strides=1)
    layer_10 = deconv_block(layer_10, 1,add_relu=False)
    decoded = tf.keras.activations.sigmoid(layer_10)


    decoder = tf.keras.models.Model(inputs=inp, outputs=decoded)

    return decoder,decoded