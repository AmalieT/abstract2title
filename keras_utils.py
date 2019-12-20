import tensorflow as tf
from keras.callbacks import Callback
import random
import logging
from keras.utils.io_utils import HDF5Matrix


# Keras doesn't like doing a sparse cross-entropy loss with unknown dims, so let's implement our own
def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

# A callback to print a decoded sequence from the validation set every n batches


class DecodeVal(Callback):
    def __init__(self, eval_every, decode_function, validation_inputs, validation_outputs, index2word, n_eval=1):
        super(DecodeVal, self).__init__()
        self.eval_every = eval_every
        self.batch = 0
        self.val_size = validation_inputs.shape[0]
        self.decode_function = decode_function
        self.validation_inputs = validation_inputs
        self.validation_outputs = validation_outputs
        self.n_eval = n_eval
        self.index2word = index2word

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.eval_every == 0:
            for _ in range(self.n_eval):
                eval_index = random.randint(0, int(self.val_size / 10))
                eval_input = self.validation_inputs[eval_index]
                eval_output = self.validation_outputs[eval_index]
                eval_sequence = self.decode_function(eval_input)
                eval_output_sequence = " ".join(
                    [self.index2word[t] for t in eval_output])
                print("\n")
                print("Predicted Sequence: ")
                print(eval_sequence)
                print("True Sequence: ")
                print(eval_output_sequence)

        self.batch += 1

# A single epoch takes forever, and we want to checkpoint frequently
# So we implement a callback that will checkpoint based on batch number
# Adds a tiny bit of batch_size dependence, which sucks, but c'est la vie


class BatchCheckpoint(Callback):
    def __init__(self, name, save_every):
        super(BatchCheckpoint, self).__init__()
        self.save_every = save_every
        self.batch = 0
        self.name = name

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.save_every == 0:
            self.model.save_weights(self.name)
        self.batch += 1

# Likewise we need a batch early stopping callback


class BatchEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of batches with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the batch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=True):
        super(BatchEarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_batch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('BatchEarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_batch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_batch_end(self, batch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_batch = batch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best batch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_batch > 0 and self.verbose > 0:
            print('Batch %05d: early stopping' % (self.stopped_batch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

# The keras hdf5Matrix class isn't happy if the normalizer function changes the number of dimensions
# It checks the shape of the pre-normalized input when used in a model, so it will fail validation
# We can get around this by overriding the shape property to return the post-normalization shape
# I didn't end up using this (Went with sparse_cross_entropy loss instead), but it seems like a handy thing to have lying around


class NormalizedHDF5Matrix(HDF5Matrix):
    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        def ds_norm(x): return x if normalizer is None else normalizer(x)
        super(NormalizedHDF5Matrix, self).__init__(
            datapath, dataset, start=start, end=end, normalizer=ds_norm)
        t_val = self[0:1]
        self._base_shape = t_val.shape[1:]
        self._base_dtype = t_val.dtype

    @property
    def shape(self):
        """Gets a numpy-style shape tuple giving the dataset dimensions.
        # Returns
            A numpy-style shape tuple.
        """
        return (self.end - self.start,) + self._base_shape

    @property
    def dtype(self):
        """Gets the datatype of the dataset.
        # Returns
            A numpy dtype string.
        """
        return self._base_dtype
