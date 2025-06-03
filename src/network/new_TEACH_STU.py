import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import Wrapper

import numpy as np
from src.network.metrics_losses import  StatefullF1


class STU_TEACH(tf.keras.Model):
    def __init__(self, student, teacher):
        super(STU_TEACH, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, student_optimizer, teacher_optimizer, loss, loss_weights, Temperature, F1_score,
                student_grad_model, teacher_grad_model, target_idx, output_type, regularizer, reg_constant,conf_use,reg_use):
        super(STU_TEACH, self).compile()
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.Temperature = Temperature
        self.f1_score = F1_score
        self.student_grad_model = student_grad_model
        self.teacher_grad_model = teacher_grad_model
        self.target_idx = target_idx
        self.output_type = output_type
        self.regularizer = regularizer
        self.reg_constant = reg_constant
        self.conf_use = conf_use
        self.reg_use = reg_use

    def teach_student_loss(self, x, output_type, confidences):

        with tf.GradientTape() as tape1:
            conv_outputs_teach, predictions_teach_grad = self.teacher_grad_model(x)
            if output_type == 'weak':
                output_teach = predictions_teach_grad[2][:, :, self.target_idx]
            if output_type == 'strong':
                output_teach = predictions_teach_grad[1][:, :, self.target_idx]
            loss_teach = output_teach

        grads_teach = tape1.gradient(loss_teach, conv_outputs_teach)

        with tf.GradientTape() as tape2:
            conv_outputs_stu, predictions_stu_grad = self.student_grad_model(x)
            if output_type == 'weak':
                output_stu = predictions_stu_grad[2][:, :, 0]
            if output_type == 'strong':
                output_stu = predictions_stu_grad[1][:, :, 0]
            loss_stu = output_stu

        grads_stu = tape2.gradient(loss_stu, conv_outputs_stu)

        with tf.GradientTape() as tape3:
            conv_outputs_stu, predictions_stu_grad = self.student_grad_model(x)
            conv_outputs_teach, predictions_teach_grad = self.teacher_grad_model(x)

            cams_stu = self.heatmap(conv_outputs_stu, grads_stu)
            #cams_stu = normalize_tensor(cams_stu)
            cams_teach = self.heatmap(conv_outputs_teach, grads_teach)
            #cams_teach = normalize_tensor(cams_teach)

            # IL PESO DELLE CONFIDENZE LO APPLICO QUI
            xai_loss = self.loss["xai_loss"](cams_teach, cams_stu, confidences=confidences, conf_used=self.conf_use)

            xai_loss =  self.loss_weights[3] * xai_loss    #

        grads_xai = tape3.gradient(xai_loss, self.student.trainable_weights)
        self.student_optimizer.apply_gradients(zip(grads_xai, self.student.trainable_weights))

        return xai_loss

    def train_step(self, data):

        x = data[0]
        y = data[1][0]
        y_w = data[1][2]

        if self.regularizer == 'gnorm' and self.reg_use:

            with tf.GradientTape() as regtape:

                regtape.watch(x)

                predictions_stu_reg = self.student(x, training=True) #todo why is true?
                #predictions_teach = self.teacher(x)

                raw_loss = self.loss["student_loss"](y_w, predictions_stu_reg[2])
                #raw_loss = self.loss["KD_loss"](predictions_teach[0][:, :, self.target_idx], predictions_stu[0][:, :, 0])

            grad = regtape.gradient(raw_loss, x)

            gradnorm = 1e5 * tf.reduce_sum(tf.square(grad)) / tf.cast(tf.shape(x)[0], tf.float32)

            reg_loss = raw_loss + 1e-3 * gradnorm

        else:

            reg_loss = 0

        with tf.GradientTape() as tape4:
            predictions_stu = self.student(x)
            predictions_teach = self.teacher(x)


            fake = tf.cast(tf.ones_like(y_w[:]), tf.float32)

            KD_loss = self.loss["KD_loss"](predictions_teach[0][:, :, self.target_idx], predictions_stu[0][:, :, 0]) # , y_w

            classification_loss = self.loss["student_loss"](y_w, predictions_stu[2])
            sum_loss = self.loss_weights[1] * KD_loss  + self.loss_weights[0] * self.loss_weights[2] * classification_loss

            # if self.reg_use:
            #     sum_loss = sum_loss +  self.loss_weights[3] * self.reg_constant * reg_loss   #todo mettere

        grads1 = tape4.gradient(sum_loss, self.student.trainable_weights)
        self.student_optimizer.apply_gradients(zip(grads1, self.student.trainable_weights))
        xai_loss = self.teach_student_loss(x, self.output_type, fake)

        return {"sum_loss": sum_loss,
                "weak_c_loss": classification_loss,
                "KD_loss": KD_loss,
                "xai_loss": xai_loss,
                "reg_loss": reg_loss}

    def test_step(self, data):
        x = data[0]
        y = data[1][0]
        y_w = data[1][2]

        predictions_stu = self.student(x)

        f1_stu = self.f1_score(y, predictions_stu[1])

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

    def call(self, inputs, training=False,*args, **kwargs):
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

class WeightAdjuster_TS(tf.keras.callbacks.Callback):
    def __init__(self, weights: float):
        """
    Args:
    weights (list): list of loss weights
    """
        self.gamma = weights

    def on_epoch_end(self, epoch, logs=None):
        int_strong = np.log10(logs['KD_loss']*0.9)
        int_weak = np.log10(logs['weak_c_loss']*0.1)
        int_loss = int_weak - int_strong
        int_loss2 = 10 ** (- int_loss)
        # Updated loss weights
        K.set_value(self.gamma, int_loss2)
        print('balance KD-WEAK',int_loss)
        tf.summary.scalar('balancing_factor', data=int_loss, step=epoch)

class WeightAdjuster_TS2(tf.keras.callbacks.Callback):
    def __init__(self, weights: float):
        """
    Args:
    weights (list): list of loss weights
    """
        self.gamma = weights

    def on_epoch_end(self, epoch, logs=None):
        int_strong = np.log10(logs['KD_loss'])
        int_weak = np.log10(logs['reg_loss'])
        int_loss = int_weak - int_strong
        int_loss2 = 10 ** (- int_loss)
        # Updated loss weights
        K.set_value(self.gamma, int_loss2)
        print('balance KD-REG',int_loss)
        tf.summary.scalar('balancing_factor', data=int_loss, step=epoch)


class PredictionControl(tf.keras.callbacks.Callback):
    def __init__(self, train_data):
        self.x_train = train_data


    def on_epoch_end(self, epoch, logs=None):
        # here you can get the model reference.
        pred = self.model.predict(self.x_train)
        tf.summary.histogram('soft_student_train_preds', data=pred[0], step=epoch)


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

