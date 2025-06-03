import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


class WeightAdjuster(tf.keras.callbacks.Callback):
  def __init__(self, weights: float):
    """
    Args:
    weights (list): list of loss weights
    """
    self.gamma = weights

  def on_epoch_end(self, epoch, logs=None):
      int_strong = np.log10(logs['strong_loss'])
      int_weak = np.log10(logs['weak_loss'])
      int_loss = round(int_weak) - round(int_strong)
      int_loss2 =  10 ** (- int_loss)
      # Updated loss weights
      K.set_value(self.gamma, int_loss2)
      print(int_loss)
      tf.summary.scalar('balancing_factor', data=int_loss, step=epoch)


class CRNN_custom(tf.keras.Model):
    def __init__(self, teacher):
        super(CRNN_custom, self).__init__()
        self.teacher = teacher

    def compile(self, student_optimizer, teacher_optimizer, loss, loss_weights,Temperature, F1_score, regularizer, reg_constant):
        super(CRNN_custom, self).compile()
        self.student_optimizer = student_optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.Temperature = Temperature
        self.f1_score = F1_score
        self.regularizer = regularizer
        self.reg_constant = reg_constant

    def train_step(self, data):

        # x = data[0]
        # y = tf.squeeze(data[1][0])
        # y_w = tf.squeeze(data[1][2])
        x = data[0]
        y = data[1][0]
        y_w = data[1][2]

        if self.regularizer == 'gnorm-w':
            if self.reg_constant == 0:
                reg_loss = 1
            else:
                with tf.GradientTape() as regtape:
                    regtape.watch(x)
                    predictions_teach = self.teacher(x)
                    weak_loss = self.loss["weak_loss"](y_w, predictions_teach[2])

                grad = regtape.gradient(weak_loss, x)
                gradnorm = 1e5 * tf.reduce_sum(tf.square(grad)) / tf.cast(tf.shape(x)[0], tf.float32)
                reg_loss = weak_loss + 1e-3 * gradnorm
        elif self.regularizer == 'gnorm-s':
            if self.reg_constant == 0:
                reg_loss = 1
            else:
                with tf.GradientTape() as regtape:
                    regtape.watch(x)
                    predictions_teach = self.teacher(x)
                    strong_loss = self.loss["strong_loss"](y, predictions_teach[1],y_w)

                grad = regtape.gradient(strong_loss, x)
                gradnorm = 1e5 * tf.reduce_sum(tf.square(grad)) / tf.cast(tf.shape(x)[0], tf.float32)
                reg_loss = strong_loss + 1e-3 * gradnorm

        # Train the student
        with tf.GradientTape() as tape:
            predictions_teach = self.teacher(x)
            strong_loss = self.loss["strong_loss"](y, predictions_teach[1],y_w)
            weak_loss = self.loss["weak_loss"](y_w, predictions_teach[2])
            sum_loss = strong_loss + weak_loss + self.reg_constant * reg_loss   # gamma eventualmente lo usiamo per pesare reg e weak

        grads = tape.gradient(sum_loss, self.teacher.trainable_weights)
        self.student_optimizer.apply_gradients(zip(grads, self.teacher.trainable_weights))

        return {"sum_loss": sum_loss, "weak_loss": weak_loss, "strong_loss": strong_loss ,"reg_loss": reg_loss}

    def test_step(self, data):

        x = data[0]
        y = data[1][0]
        y_w = data[1][2]

        predictions_stu = self.teacher(x)

        # strong_loss = self.loss["strong_loss"](y, predictions_stu[1])
        # weak_loss = self.loss["weak_loss"](y_w, predictions_stu[2])

        #sum_loss =   strong_loss + self.loss_weights[0] * weak_loss
        f1_stu = self.f1_score(y, predictions_stu[1])

        return {"F1_score": f1_stu}

    def call(self, inputs, *args, **kwargs):
        return self.teacher(inputs)
