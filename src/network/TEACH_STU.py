import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import Wrapper

import numpy as np
from src.network.metrics_losses import StatefullF1


class WeightAdjuster_TS(tf.keras.callbacks.Callback):
    def __init__(self, weights: float):
        """
    Args:
    weights (list): list of loss weights
    """
        self.gamma = weights

    def on_epoch_end(self, epoch, logs=None):
        int_strong = np.log10(logs['KD_loss'])
        int_weak = np.log10(logs['weak_loss'])
        int_loss = int_weak - int_strong
        int_loss2 = 10 ** (- int_loss)
        # Updated loss weights
        K.set_value(self.gamma, int_loss2)
        print(int_loss)
        tf.summary.scalar('balancing_factor', data=int_loss, step=epoch)


# class WeightAdjuster(tf.keras.callbacks.Callback):
#     def __init__(self, weights: list):
#         """
#         Args:
#         weights (list): list of loss weights
#         """
#         self.alpha = weights[0]
#         self.beta = weights[1]
#
#     def on_epoch_begin(self, epoch, logs=None):
#         x = epoch / 80
#         val_ = np.exp(-5.0 * (1.0 - x) ** 2.0)
#         val = tf.cast(val_, tf.float32)
#         new_beta = 50 * val
#         # Updated loss weights
#         K.set_value(self.beta, new_beta)
#         print(new_beta)
#         print(self.beta)
#
#         tf.summary.scalar('weight_unsup', data=new_beta, step=epoch)

class PredictionControl(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data):
        self.x_train = train_data
        # self.x_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        # here you can get the model reference.
        pred = self.model.predict(self.x_train)
        tf.summary.histogram('soft_student_train_preds', data=pred[0], step=epoch)


def loss_fn_sup(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true, new_pred)

    return tf.reduce_mean(loss)


def xai_cosine_loss(teacher_explanation, student_explanation, axis=1):
    teacher = tf.linalg.l2_normalize(teacher_explanation, axis=axis)
    student = tf.linalg.l2_normalize(student_explanation, axis=axis)

    cosine_similarity_loss = -tf.reduce_sum(teacher * student, axis=axis)
    return tf.reduce_mean(cosine_similarity_loss)#tf.reduce_mean(cosine_similarity_loss)


def root_mean_squared_error(teacher_explanation, student_explanation):
    return K.abs(K.sqrt(K.mean(K.square(student_explanation - teacher_explanation))))


def loss_fn_unsup(y_true, y_pred):
    # y_true = tf.cast(y_true, tf.float32)
    # new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    # new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    loss = K.mean(K.sum(K.square(y_true - y_pred)))
    return loss


def normalize_tensor(tensor):
    norm_tensor = tf.math.divide(
                            tf.math.subtract(
                                tensor,
                                tf.reduce_min(tensor)
                            ),
                            tf.math.subtract(
                                tf.reduce_max(tensor),
                                tf.reduce_min(tensor)
                            )
                        )
    return norm_tensor

import matplotlib.pyplot as plt


class STU_TEACH(tf.keras.Model):
    def __init__(self, student, teacher):
        super(STU_TEACH, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, student_optimizer, teacher_optimizer, loss, loss_weights, Temperature, F1_score,
                student_grad_model, teacher_grad_model, target_idx, output_type, batch_size):
        super(STU_TEACH, self).compile()
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.f1_score = F1_score
        self.student_grad_model = student_grad_model
        self.teacher_grad_model = teacher_grad_model
        self.target_idx = target_idx
        self.output_type = output_type
        self.batch_size = batch_size


    def train_step(self, data):

        x = data[0]
        y_w = data[1][2]
        tot_loss = tf.zeros([1])
        # computes the original student loss
        with tf.GradientTape() as tape1:


            for sample in range(self.batch_size):
                #print(sample)
                predictions_stu = self.student(tf.expand_dims(x[sample], axis=0))
                predictions_teach = self.teacher(tf.expand_dims(x[sample], axis=0))

                KD_loss = self.loss["KD_loss"](predictions_teach[0][:, :, self.target_idx], predictions_stu[0][:, :, 0])
                classification_loss = self.loss["student_loss"](y_w[:], predictions_stu[2][:, : , :]) #
                sum_loss =  self.loss_weights[1] * KD_loss  + self.loss_weights[0] * self.loss_weights[2] * classification_loss
                tot_loss = sum_loss + tot_loss

        #tot_loss = tot_loss / tf.cast(12,tf.float32)
        grads1 = tape1.gradient(tot_loss, self.student.trainable_weights)
        self.student_optimizer.apply_gradients(zip(grads1, self.student.trainable_weights))

        # computes the gradients of the strong (or weak) prediction of the teacher model
        # Used later to generate the teacher heatmap
        with tf.GradientTape() as tape2:
            conv_outputs_teach, predictions_teach_grad = self.teacher_grad_model(x)
            if self.output_type == 'weak':
                output_teach = predictions_teach_grad[2][:, :, self.target_idx]
            if self.output_type == 'strong':
                output_teach = predictions_teach_grad[1][:, :, self.target_idx]
            loss_teach = output_teach

        grads_teach = tape2.gradient(loss_teach, conv_outputs_teach,unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # computes the gradients of the strong (or weak) prediction of the student model
        # Used later to generate the student heatmap
        with tf.GradientTape() as tape3:
            conv_outputs_stu, predictions_stu_grad = self.student_grad_model(x)
            if self.output_type == 'weak':
                output_stu = predictions_stu_grad[2][:, :, 0]
            if self.output_type == 'strong':
                output_stu = predictions_stu_grad[1][:, :, 0]
            loss_stu = output_stu

        grads_stu = tape3.gradient(loss_stu, conv_outputs_stu,unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # creates the heatmaps based on the output and the computed gradients
        # compares the distance between the two heatmaps (cosine or rmse)
        with tf.GradientTape() as tape4:
            conv_outputs_stu, predictions_stu_grad = self.student_grad_model(x)
            conv_outputs_teach, predictions_teach_grad = self.teacher_grad_model(x)

            cams_stu = self.heatmap(conv_outputs_stu, grads_stu)
            #cams_stu = normalize_tensor(cams_stu)
            cams_teach = self.heatmap(conv_outputs_teach, grads_teach)
            #cams_teach = normalize_tensor(cams_teach)

            xai_loss = self.loss["xai_loss"](cams_teach, cams_stu)
            xai_loss = self.loss_weights[3] * xai_loss

        # applies the xai_loss
        grads_xai = tape4.gradient(xai_loss, self.student.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.student_optimizer.apply_gradients(zip(grads_xai, self.student.trainable_weights))

        #
        #

        return {"loss": sum_loss, "weak_loss": classification_loss, "KD_loss": KD_loss, "xai_loss": xai_loss} #, "xai_loss_student": xai_loss_student}  # , "f1_score_stu":f1_stu, "f1_score_teach":f1_teach

    def test_step(self, data):
        x = data[0]
        y = data[1][0]
        y_w = data[1][2]

        predictions_stu = self.student(x)
        predictions_teach = self.teacher(x)


        f1_stu = self.f1_score(y, predictions_stu[1])

        # return {"loss": sum_loss, "weak_loss": classification_loss,
        #         "strong_loss": strong_classification_loss, "KD_loss": KD_loss,
        #         "f1_score_stu": f1_stu}  # , "f1_score_teach":f1_teach
        return {"f1_score_stu": f1_stu}

    @tf.function
    def reverse_enumerate(self, iterable):
        """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
        """
        return zip(reversed(range(len(iterable))), reversed(iterable))

    @tf.function
    def find_penultimate_layer(self, forward_func, layer_idx, penultimate_layer_idx):
        if penultimate_layer_idx is None:
            for idx, layer in self.reverse_enumerate(forward_func.layers[:layer_idx - 1]):
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

    def call(self, inputs, *args, **kwargs):
        return self.student(inputs)

    @tf.function
    def generate_heatmap(self, outputs, grads):

        maps = [self.heatmap(output, grad) for output, grad in zip(outputs, grads)]

        return maps

    @tf.function
    def heatmap(self, output, grad):

        weights = tf.reduce_mean(grad, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

        # only return positive
        cam = tf.nn.relu(cam)

        return cam