from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import regularizers
import tensorflow
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec


class LinSoftmaxPooling1D(Layer):
    '''
    Keras softmax pooling layer.
    '''

    def __init__(self, axis=0, **kwargs):
        '''
        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0
            (should be time).
        kwargs
        '''
        super(LinSoftmaxPooling1D, self).__init__(**kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(LinSoftmaxPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        square = x * x
        sum_square = K.sum(square, axis=self.axis, keepdims=True)
        sum = K.sum(x, axis=self.axis, keepdims=True)
        fin_vector = sum_square / sum
        return fin_vector
