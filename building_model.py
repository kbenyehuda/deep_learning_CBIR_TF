import tensorflow as tf
import tensorflow_hub as hub


feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224,3), trainable=False)

input_feature_vector = tf.keras.layers.Input(shape=1280,)

layer_300 = tf.keras.layers.Dense(300,kernel_initializer='truncated_normal',bias_initializer='truncated_normal')(input_feature_vector)
layer_300 = tf.keras.layers.BatchNormalization()(layer_300)
layer_300 = tf.keras.layers.LeakyReLU(0.3)(layer_300)
layer_300 = tf.keras.layers.Dropout(0.5)(layer_300)

layer_300_5 = tf.keras.layers.Dense(5,kernel_initializer='truncated_normal',bias_initializer='truncated_normal')(layer_300)
layer_300_5 = tf.keras.activations.sigmoid(layer_300_5)

layer_50 = tf.keras.layers.Dense(50,kernel_initializer='truncated_normal',bias_initializer='truncated_normal')(layer_300)
layer_50 = tf.keras.layers.BatchNormalization()(layer_50)
layer_50 = tf.keras.layers.LeakyReLU(0.3)(layer_50)
layer_50 = tf.keras.layers.Dropout(0.5)(layer_50)

output = tf.keras.layers.Dense(5,kernel_initializer='truncated_normal',bias_initializer='truncated_normal')(layer_50)
output = tf.keras.activations.sigmoid(output)

encoder = tf.keras.Model(inputs = input_feature_vector,outputs=[layer_300,layer_300_5,layer_50,output])

print('built encoder')