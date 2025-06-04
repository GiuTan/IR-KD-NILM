import os
import argparse
import src.network.CRNN_t
import src.network.CRNN
from utils_func import *
import random
from src.network.new_TEACH_STU import *
from src.network.new_TEACH_STU import WeightAdjuster_TS
import src.network.CRNN_t
import tensorflow as tf
from utils_func import *
from params import params, uk_params, refit_params
from src.network.metrics_losses import *

'''from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)'''


parser = argparse.ArgumentParser(description="Knowledge Distillation for Transfer Learning")

parser.add_argument("--gpu", type=str, default="0", help="GPU")
parser.add_argument("--temperature", type=float, default=2, help="Temperature for KD")
parser.add_argument("--beta", type=float, default=0.9, help="KD loss weight")
parser.add_argument("--fine_tuning", type=bool, default=False, help="Flag to fine-tune the teacher before KD")
parser.add_argument("--model", type=str, default="strong_weakUK", help="UKDALE or REFIT pre-training selection")
parser.add_argument("--num_conv", type=int, default=1, help="Number of convolutional blocks")
parser.add_argument("--num_GRUnits", type=int, default=32, help="Number of GRUnits")
parser.add_argument("--xai_weight", type=float, default=0.3, help="loss weight for (strong) xai part")
parser.add_argument("--reg_weight", type=float, default=0.05, help="loss weight for reg part")
parser.add_argument("--output_type", type=str, default='weak', help="xai out type")
parser.add_argument("--target_idx", type=int, default=3, help="target idx")
parser.add_argument("--xai_loss", type=str, default='cosine', help="xai loss")
parser.add_argument("--conf_use", type=bool, default=False, help="uso confidenza")
parser.add_argument("--reg_use", type=bool, default=False, help="uso regolarizzazione")
parser.add_argument("--precision", type=bool, default=False, help="zero-strong label loss")
parser.add_argument("--test_test", type=bool, default=False, help="test flag")
parser.add_argument("--model_path", type=str, default="/raid/users/eprincipi/XAIKD_extension/GITHUB/models/", help="folder for models")
parser.add_argument("--data_path", type=str, default="/raid/users/eprincipi/", help="folder for models")

arguments = parser.parse_args()


if __name__ == '__main__':

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

    print('Parameter setting:')
    print('GPU', arguments.gpu)
    print('Temperature', arguments.temperature)
    print('Beta', arguments.beta)
    print('Fine-tuning', arguments.fine_tuning)
    print('Model', arguments.model)
    print('Number of convolutional blocks', arguments.num_conv)
    print('Number of gated recurrent units', arguments.num_GRUnits)
    print('Xai weight', arguments.xai_weight)
    print('Reg weight', arguments.reg_weight)
    print('Output type', arguments.output_type)
    print('Target idx', arguments.target_idx)
    print('Uso confidenza', arguments.conf_use)
    print('Uso regolarizzazione', arguments.reg_use)

    #disable_eager_execution()

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
    strong_weak = True  # se avessi voluto escludere da alcuni segmenti le labels weak
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
    Y_train_r = np.negative(np.ones((10481, 2550, 6))) # this mimics the absence of strong labels
    Y_train_weak_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_train_weak.npy')

    # test data used in every part of the framework
    X_test_r = np.load(arguments.data_path + 'KD_agg_REFIT/new2_X_test.npy')
    Y_test_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_test.npy')
    Y_test_weak_r = np.load(arguments.data_path + 'KD_labels_REFIT/new2_Y_test_weak.npy')


    val_l = round(len(X_test_r) / 100 * 10 / 2)

    X_val_tot = X_val_u  #
    Y_val = Y_val_u[:, :, arguments.target_idx]  #
    Y_val_weak = Y_val_weak_u[:, :, arguments.target_idx]  #
    X_test_r = X_test_r[val_l:-val_l]
    Y_test_r = Y_test_r[val_l:-val_l]
    Y_test_r = Y_test_r[:, :, arguments.target_idx]

    Y_val_weak = np.expand_dims(Y_val_weak, axis=2)  #

    Y_val = np.expand_dims(Y_val, axis=2)
    Y_test_r = np.expand_dims(Y_test_r, axis=2)

    assert (len(X_val_u) == len(Y_val_u))
    assert (len(Y_val_u) == len(Y_val_weak_u))

    x_train = X_train_r
    y_strong_train = Y_train_r[:, :, arguments.target_idx]
    y_weak_train = Y_train_weak_r[:, :, arguments.target_idx]


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
    theta = K.variable(1.0)
    weight_dyn = WeightAdjuster_TS(weights=gamma)
    weight_dyn2 = WeightAdjuster_TS2(weights=theta)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_stu', mode='max', patience=5, restore_best_weights=True)




    teacher_model_path = arguments.model_path + "best_teachers/new_teach_loss_reg_gnorm-w.h5"
    teacher = src.network.CRNN_t.CRNN_construction(window_size, weight, lr=lr, classes=classes, drop_out=drop,
                                                   kernel=kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                                                   path=teacher_model_path, only_strong=only_strong, temperature=temperature)



    student = src.network.CRNN.CRNN_construction(window_size, weight, lr=lr, classes=1, drop_out=drop, kernel=kernel,
                                                 num_layers=1, gru_units=32, cs=cs)


    MODEL_ = STU_TEACH(student, teacher)



    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # xai part
    layer_student = find_penultimate_layer(MODEL_.student, -1, None)
    student_grad_model = tf.keras.models.Model([MODEL_.student.inputs], [layer_student.output, MODEL_.student.output])

    layer_teacher = find_penultimate_layer(MODEL_.teacher, -1, None)
    teacher_grad_model = tf.keras.models.Model([MODEL_.teacher.inputs], [layer_teacher.output, MODEL_.teacher.output])

    if arguments.xai_loss == 'cosine':
        xai_loss = xai_cosine_loss
    elif arguments.xai_loss == 'rmse':
        xai_loss = root_mean_squared_error

    kd_loss = zero_strong_label_loss_S

    MODEL_.compile(student_optimizer=optimizer, teacher_optimizer=optimizer,
                   loss={"student_loss": loss_fn_sup, "KD_loss": kd_loss, "xai_loss": xai_loss},
                   loss_weights=[alpha, beta, gamma, xai_weight], Temperature=temperature, F1_score=StatefullF1(n_class=1),
                   student_grad_model=student_grad_model,
                   teacher_grad_model=teacher_grad_model,
                   target_idx=arguments.target_idx,
                   output_type=arguments.output_type,
                   regularizer='gnorm',
                   reg_constant=reg_weight,
                   conf_use=arguments.conf_use,
                   reg_use = arguments.reg_use)
    if not arguments.test_test:
        history_ = MODEL_.fit(x=x_train, y=[y_strong_train, y_strong_train, y_weak_train],
                              shuffle=True, epochs=1000,
                              validation_data=(X_val, [Y_val, Y_val, Y_val_weak]),
                              callbacks=[early_stop, weight_dyn,weight_dyn2],
                              batch_size=batch_size, verbose=1)

        losses = history_.history['val_f1_score_stu']

        best_loss = round(np.max(losses), 3)
        best_epoch = np.argmax(losses)

        # PREDIZIONE TEACHER E STUDENT
        PATH_TEST = arguments.model_path + f"PROVA_precision_loss_{model_}_weight_{xai_weight}_{reg_weight}_{arguments.target_idx}_{arguments.output_type}_{best_epoch}_{best_loss}_model.h5"
        MODEL_.student.save_weights(PATH_TEST)
    else:
        PATH_TEST= arguments.model_path + 'best_students/DW.h5'
        MODEL_.student.load_weights(PATH_TEST)

    val_soft_strong, output_strong, output_weak = MODEL_.predict(x=X_val)
    test_soft_strong, output_strong_test_o, output_weak_test = MODEL_.predict(x=X_test)

    print(Y_val.shape)
    print(output_strong.shape)

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]

    Y_val = Y_val.reshape(shape, 1)
    Y_test = Y_test_r.reshape(shape_test, 1)

    output_strong = output_strong.reshape(shape, 1)
    thres_strong_stu = thres_analysis(Y_val, output_strong, 1)
    output_strong = np.where(output_strong >= thres_strong_stu, 1, output_strong)
    output_strong = np.where((output_strong != -1) & (output_strong < thres_strong_stu), 0, output_strong)

    output_strong_test = output_strong_test_o.reshape(shape_test, 1)
    thres = thres_strong_stu
    output_bin = np.where(output_strong_test >= thres, 1, output_strong_test)
    output_bin = np.where((output_bin != -1) & (output_bin < thres), 0, output_bin)
    output_strong = output_strong.reshape(shape, 1)
    output_strong_test = output_bin.reshape(shape_test, 1)

    np.save(arguments.model_path + 'best_students/'+str(arguments.target_idx) +'_pred.npy',output_strong_test)


    print("STRONG SCORES:")
    print("Validation")
    b = classification_report(Y_val, output_strong)
    print(b)
    print("Student Test")
    a = classification_report(Y_test, output_strong_test)
    print(a)

    if not arguments.test_test:
        with open(arguments.model_path + PATH_TEST +
                  'scores.txt',
                  'a+') as f:
            print('classification report validation student: %s' % b, file=f)
            print('classification report test student: %s' % a, file=f)
            print('optimized threshold: %s' % thres_strong_stu, file=f)
            print('', file=f)
    else:
        with open(arguments.model_path + PATH_TEST +
                  'scores.txt',
                  'a+') as f:
            print('classification report validation student: %s' % b, file=f)
            print('classification report test student: %s' % a, file=f)
            print('optimized threshold: %s' % thres_strong_stu, file=f)
            print('', file=f)

