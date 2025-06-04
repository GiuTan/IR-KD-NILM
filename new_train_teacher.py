import os
import argparse
from utils_func import *
import random
import src.network.new_teach_CRNN_t
import tensorflow as tf
from metrics import *
from src.network.metrics_losses import *
from params import params, uk_params, refit_params
from src.network.new_teach_CRNN_custom import CRNN_custom
from src.network.metrics_losses import *
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import Wrapper
# from tensorflow.python.framework.ops import disable_eager_execution
from src.focal_loss import *

from tensorflow.keras.callbacks import TensorBoard

'''from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)'''

parser = argparse.ArgumentParser(description="Knowledge Distillation for Transfer Learning")

parser.add_argument("--gpu", type=str, default="5", help="GPU")
parser.add_argument("--temperature", type=float, default=2, help="Temperature for KD")
parser.add_argument("--beta", type=float, default=0.9, help="KD loss weight")
parser.add_argument("--fine_tuning", type=bool, default=False, help="Flag to fine-tune the teacher before KD")
parser.add_argument("--model", type=str, default="strong_weakUK", help="UKDALE or REFIT pre-training selection")
parser.add_argument("--xai_weight", type=float, default=0.5, help="loss weight for (strong) xai part")
parser.add_argument("--output_type", type=str, default='weak', help="xai out type")
parser.add_argument("--target_idx", type=int, default=3, help="target idx")
parser.add_argument("--xai_loss", type=str, default='cosine', help="xai loss")
parser.add_argument("--reg", type=str, default='gnorm-w', help="Flag to use regularizer")
parser.add_argument("--reg_weight", type=float, default=0.05, help="Flag to set reg weight")
parser.add_argument("--model_path", type=str, default="/raid/users/eprincipi/XAIKD_extension/GITHUB/models/", help="folder for models")
parser.add_argument("--data_path", type=str, default="/raid/users/eprincipi/", help="folder for models")
arguments = parser.parse_args()


def reverse_enumerate(iterable):
    """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))


def find_penultimate_layer(forward_func, layer_idx, penultimate_layer_idx):
    if penultimate_layer_idx is None:
        for idx, layer in reverse_enumerate(forward_func.layers[:layer_idx - 1]):
            if isinstance(layer, Wrapper):
                layer = layer.layer
            if isinstance(layer, (Conv2D, MaxPooling1D, MaxPooling2D, MaxPooling3D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Conv` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))

    # Handle negative indexing otherwise the next check can fail.
    if layer_idx < 0:
        layer_idx = len(forward_func.layers) + layer_idx
    if penultimate_layer_idx > layer_idx:
        raise ValueError('`penultimate_layer_idx` needs to be before `layer_idx`')

    return forward_func.layers[penultimate_layer_idx]


if __name__ == '__main__':
    trial = 1
    os.environ['PYTHONHASHSEED'] = str(123)
    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.experimental.numpy.random.seed(1234)

    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu

    # UK-DALE path
    print(tf.config.list_physical_devices('GPU'))

    print('Parameter setting:')
    print('GPU', arguments.gpu)
    print('Temperature', arguments.temperature)
    print('Beta', arguments.beta)
    print('Fine-tuning', arguments.fine_tuning)
    print('Model', arguments.model)
    print('Xai weight', arguments.xai_weight)
    print('Reg weight', arguments.reg_weight)
    print('Output type', arguments.output_type)
    print('Target idx', arguments.target_idx)

    regularizer = arguments.reg
    reg_weight = arguments.reg_weight



    print('Start')

    model_ = arguments.model

    # Flag Inizialization
    flag = arguments.fine_tuning
    if flag:
        test = False
    else:
        test = True
    print('Flag', flag)
    strong = False
    strong_weak = True
    test_unseen = True
    weak_counter = True
    validation_refit = False
    val_only_weak_re = False
    classes = 6

    # validation data used in every part of the framework
    X_val_u = np.load(arguments.data_path + 'KD_agg_UKDALE/new_X_val.npy')
    Y_val_u = np.load(arguments.data_path + 'KD_labels_UKDALE/new_Y_val.npy')
    Y_val_weak_u = np.load(arguments.data_path + 'KD_labels_UKDALE/new_Y_val_weak.npy')

    # weakly labelled training data for Teacher fine-tuning and Student distillation
    X_train_r = np.load(arguments.data_path + 'KD_agg_REFIT/new2_X_train.npy')
    Y_train_r = np.negative(np.ones((10481, 2550, 6)))  # this mimics the absence of strong labels
    Y_train_weak_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_train_weak.npy')

    # test data used in every part of the framework
    X_test_r = np.load(arguments.data_path + 'KD_agg_REFIT/new2_X_test.npy')
    Y_test_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_test.npy')
    Y_test_weak_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_test_weak.npy')


    val_l = round(len(X_test_r) / 100 * 10 /2 )

    X_val_tot = X_val_u
    Y_val =  Y_val_u
    Y_val_weak = Y_val_weak_u
    X_test_r = X_test_r[val_l:-val_l]
    Y_test_r = Y_test_r[val_l:-val_l]
    Y_test_weak_r = Y_test_weak_r[val_l:-val_l]

    assert(len(X_val_u)==len(Y_val_u))
    assert(len(Y_val_u)==len(Y_val_weak_u))

    x_train = X_train_r
    y_strong_train = Y_train_r
    y_weak_train = Y_train_weak_r



    # Standardization with uk-dale values
    if model_ == 'solo_weakUK' or model_ == 'strong_weakUK' or model_ == 'mixed':
        train_mean = uk_params['mean']
        train_std = uk_params['std']
    else:
        train_mean = refit_params['mean']
        train_std = refit_params['std']


    x_train = standardize_data(x_train, train_mean, train_std)
    X_val = standardize_data(X_val_tot, train_mean, train_std)
    X_test = standardize_data(X_test_r, train_mean, train_std)

    batch_size = 64
    window_size = 2550
    drop = params[model_]['drop']
    kernel = params[model_]['kernel']
    num_layers = params[model_]['layers']
    gru_units = params[model_]['GRU']
    cs = params[model_]['cs']
    only_strong = params[model_]['no_weak']
    temperature = arguments.temperature
    bet = arguments.beta
    pat = 30

    lr = 0.002
    weight = 1
    classes = 6
    reg_weight = arguments.reg_weight
    xai_weight = arguments.xai_weight
    beta = 0.9
    alpha = 1 - beta
    gamma = K.variable(1.0)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_F1_score', mode='max', patience=5, restore_best_weights=True)

    # Carico il pre-trained model
    pre_trained = params[model_]['pre_trained']


    teacher = src.network.new_teach_CRNN_t.CRNN_construction(window_size, weight, lr=lr, classes=classes, drop_out=drop,
                                                   kernel=kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                                                   path=pre_trained, only_strong=only_strong, temperature=temperature)

    MODEL = CRNN_custom(teacher)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    MODEL.compile(student_optimizer=optimizer, teacher_optimizer=optimizer,
                  loss={"strong_loss": zero_strong_label_loss_T,
                        "weak_loss": BinaryFocalLoss(gamma=0.2)},
                  loss_weights=[gamma], Temperature=1, F1_score=StatefullF1(),
                  regularizer=regularizer, reg_constant=reg_weight
                  )

    if not test:
        history = MODEL.fit(x=x_train, y=[y_strong_train, y_strong_train, y_weak_train], shuffle=True, epochs=1000,

                            batch_size=batch_size,

                            validation_data=(X_val, [Y_val, Y_val, Y_val_weak]),

                            callbacks=[early_stop], verbose=1)

    # PREDIZIONE TEACHER E STUDENT
        MODEL.teacher.save_weights(
        f"REV_teach_reg_{regularizer}_{reg_weight}_{arguments.output_type}_{arguments.target_idx}_model_T"+ str(trial) +".h5")
    else:
        MODEL.teacher.load_weights('/raid/users/eprincipi/XAIKD_extension/teacher_optimization/multilabel/teacher_frame_loss_T1.h5') # /raid/users/eprincipi/XAIKD_extension/teacher_optimization/new_teacher_loss_reg_gnorm-w.h5
    val_soft_strong, output_strong, output_weak = MODEL.predict(x=X_val)
    test_soft_strong, output_strong_test_o, output_weak_test = MODEL.predict(x=X_test)

    print(Y_val.shape)
    print(output_strong.shape)

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]


    appliance = arguments.target_idx
    Y_val_a = Y_val[:, :, appliance]
    Y_test_a = Y_test_r[:, :, appliance]
    output_strong_a = output_strong[:, :, appliance]
    output_strong_test_o_a = output_strong_test_o[:, :, appliance]
    output_weak_test_a = output_weak_test[:, :, appliance]
    output_weak_a = output_weak[:, :, appliance]

    Y_val_a = Y_val_a.reshape(shape, 1)
    Y_test_a = Y_test_a.reshape(shape_test, 1)
    Y_test_a = Y_test_a.reshape(shape_test, 1)

    output_strong_a = output_strong_a.reshape(shape, 1)
    output_strong_test_o_a = output_strong_test_o_a.reshape(shape_test, 1)

    thres_strong_stu = thres_analysis(Y_val_a, output_strong_a, 1)

    output_weak_test_a = output_weak_test_a.reshape(output_weak_test_a.shape[0] * output_weak_test_a.shape[1], 1)
    output_weak_a = output_weak_a.reshape(output_weak_a.shape[0] * output_weak_a.shape[1], 1)

    assert (Y_val_a.shape == output_strong_a.shape)

    print("Estimated best thresholds stu:", thres_strong_stu)
        # print("Estimated best thresholds teacher:", thres_strong_teach)

    output_strong_test_a = np.where(output_strong_test_o_a >= thres_strong_stu, 1, output_strong_test_o_a)
    output_strong_test_a = np.where((output_strong_test_a != -1) & (output_strong_test_a < thres_strong_stu), 0,
                                      output_strong_test_a)
    output_strong_test_a = np.expand_dims(output_strong_test_a, axis=2)



    output_strong_a = np.where(output_strong_a >= thres_strong_stu, 1, output_strong_a)
    output_strong_a = np.where((output_strong_a != -1) & (output_strong_a < thres_strong_stu), 0, output_strong_a)
    output_strong_a = np.expand_dims(output_strong_a, axis=2)

    output_strong_a = output_strong_a.reshape(shape, 1)
    output_strong_test_a = output_strong_test_a.reshape(shape_test, 1)

    print("STRONG SCORES:")
    print(f"Validation for appliance {appliance}")
    b = classification_report(Y_val_a, output_strong_a)
    print(b)
    print(f"Teacher Test for appliance {appliance}")
    a = classification_report(Y_test_a, output_strong_test_a)
    print(a)
