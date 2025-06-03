from CRNN import *
import os
import tensorflow as tf
from net_flops import net_flops
from keras_flops import get_flops
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

def flops_model(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops


os.environ["CUDA_VISIBLE_DEVICES"]= "0"
student = CRNN_construction(2550,0.1, lr=0.002, classes=1, drop_out=0.1, kernel = 5, num_layers=1, gru_units=32, cs=False)
student.summary()
# res = tensorflow.keras.applications.ResNet50(weights=None)
# net_flops(res,table=True)
# flops = get_flops(res, batch_size=64)
# print(f"FLOPS: {flops / 10 ** 9:.03} G")
# print("The FLOPs is:{}".format(flops_model(res)) ,flush=True )

#net_flops(student,table=True) # non considera RNN
#flops = get_flops(student, batch_size=64) # python library, in teoria non tiene conto dei layer RNN
#print(f"FLOPS: {flops / 10 ** 9:.03} G")
print("The FLOPs is:{}".format(flops_model(student)) ,flush=True )


# model = tf.keras.models.Sequential()
# model.add( tf.keras.layers.GRU(8, input_shape=(10, 8), recurrent_dropout=0.2, return_sequences=True))
# model.add(tf.keras.layers.GRU(4, recurrent_dropout=0.2))
# model.add(tf.keras.layers.Dense(3, activation='tanh'))
# model.compile(loss='mse', optimizer='adam')
#net_flops(model,table=True)
#print("The FLOPs is:{}".format(flops_model(model)) ,flush=True )
