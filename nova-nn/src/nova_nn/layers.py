
#create custom layer to use in model
import tensorflow as tf

from tensorflow.keras.layers import  Layer, Dense
class CombineNDandOsc(Layer):
    def __init__(self, nd_spectra):
        super().__init__()
        self.nd_const = tf.constant(nd_spectra, dtype=tf.float32)  
        self.n_sections = nd_spectra.shape[0]
        self.n_bins = nd_spectra.shape[1]
        self.dense = Dense(self.n_sections, activation='softmax')

    def call(self, params):
        batch_size = tf.shape(params)[0]
        nd_tiled = tf.tile(self.nd_const[None, :, :], [batch_size, 1, 1]) 

        weights = self.dense(params)            
        weights = tf.expand_dims(weights, -1)   

        combined = tf.reduce_sum(weights * nd_tiled, axis=1) 
        return combined 