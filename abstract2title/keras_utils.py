import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import random
import logging
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.layers import Layer


# Sparse cross entropy with mask on padding
def sparse_cross_entropy(y_true, y_pred):
    # y_true = tf.reshape(y_true, shape=(-1, title_maxlen))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 2), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

# A custom learning rate schedule for transformer networks


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model=256, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def sample_without_replacement(logits, K):
    """
    Sample from a tensor of logits without replacement using the Gumbel max trick

    https://arxiv.org/pdf/1903.06059.pdf
    http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    This trick is so fucking cool. Who comes up with this shit?

    Note that the distribution doesn't depend on logits or K, meaning no costly distribution instantiations
    The world is your oyster with the Gumbel max trick.
    """
    dist = tfp.distributions.Gumbel(loc=0, scale=1)
    z = dist.sample(tf.shape(logits))
    _, indices = tf.nn.top_k(logits + z, K)
    return indices


def stochastic_beam_search(model, sentence, maxlen, beam_width=5, bos_ind=0, eos_ind=1):
    # Make a prediction using a stochastic beam search
    initial_output = [bos_ind]
    beams = [(initial_output, 0)]

    sentence = tf.expand_dims(sentence, axis=0)

    isComplete = False

    while not isComplete:
        new_beams = []
        for beam in beams:
            if len(beam[0]) > maxlen:
                isComplete = True
            if beam[0][-1] == eos_ind:
                new_beams.append(beam)
                continue
            output = tf.expand_dims(beam[0], 0)
            predictions = model(
                inputs=[sentence, output], training=False)

            predictions = tf.nn.log_softmax(predictions[:, -1:, :])

            predictions = tf.squeeze(predictions)

            top_k_indices = sample_without_replacement(predictions, beam_width)

            for ind in top_k_indices.numpy()[:beam_width]:
                prob = predictions[ind].numpy() + beam[1]
                new_output = beam[0] + [ind]
                new_beams.append((new_output, prob))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
            :beam_width]
        if all([beam[0][-1] == eos_ind for beam in beams]):
            isComplete = True

    return beams


def predict(model, sentence, word2index, index2word, beam_width=5, n_beams=2, maxlen=32):
    beams = stochastic_beam_search(model,
                                   sentence, maxlen=maxlen, beam_width=beam_width, bos_ind=word2index['<BOS>'], eos_ind=word2index['<EOS>'])

    predictions = []
    for beam in beams:
        predicted_sentence = " ".join([index2word[t] for t in beam[0]])
        predictions.append((predicted_sentence, math.exp(beam[1])))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[
        :n_beams]

    return predictions

# A callback to print a decoded sequence from the validation set every n batches


class DecodeVal(Callback):
    def __init__(self, eval_every, decode_function, validation_data, word2index, index2word, model, maxlen=32, n_eval=1, beam_width=None):
        super(DecodeVal, self).__init__()
        self.eval_every = eval_every
        self.batch = 0
        self.val_size = len(validation_inputs)
        self.decode_function = decode_function
        self.validation_inputs, self.validation_outputs = validation_data
        self.n_eval = n_eval
        self.word2index = word2index
        self.index2word = index2word
        self.beam_width = beam_width
        self.model = model
        self.maxlen = maxlen

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.eval_every == 0:
            for _ in range(self.n_eval):

                eval_index = random.randint(0, int(self.val_size))
                eval_input = self.validation_inputs[eval_index]
                eval_output = self.validation_outputs[eval_index]
                eval_output_sequence = " ".join(
                    [self.index2word[t] for t in eval_output])

                print("\n")
                print("True Sequence: ")
                print(eval_output_sequence.replace(
                    "@@ ", "").replace(" <PAD>", ""))

                if self.beam_width is None:
                    eval_sequence = self.decode_function(
                        self.model, eval_input, self.word2index, self.index2word, maxlen=self.maxlen)
                    print("\n")
                    print("Predicted Sequence: ")
                    print(eval_sequence.replace(
                        "@@ ", "").replace(" <PAD>", ""))

                else:
                    eval_sequences = self.decode_function(self.model,
                                                          eval_input, self.word2index, self.index2word, beam_width=self.beam_width, maxlen=self.maxlen)

                    print("\n")
                    print("Predicted Sequences: ")
                    for eval_sequence in eval_sequences:
                        print("\n")
                        print("Prob: {}, Sequence: {}".format(
                            eval_sequence[1], eval_sequence[0].replace("@@ ", "").replace(" <PAD>", "")))

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

# The tf.keras hdf5Matrix class isn't happy if the normalizer function changes the number of dimensions
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


"""
Attention functions
"""


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask)

        self.attention_weights = attention_weights

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 2), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) /
                            tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

def classifier_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.GlobalAveragePooling1D()(attention)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.layers.Input(
        shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(
        units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def encoder_classifier(vocab_size,
                       num_layers,
                       units,
                       d_model,
                       num_heads,
                       dropout,
                       num_classes,
                       abstract_maxlen,
                       name="transformer"):
    inputs = tf.keras.Input(shape=(abstract_maxlen,), name="inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, abstract_maxlen),
        name='enc_padding_mask')(inputs)

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers-1):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, enc_padding_mask])

    outputs = classifier_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="classifier_layer",
        )([outputs, enc_padding_mask])


    outputs = tf.keras.layers.Dense(
        units=2*d_model, activation='relu')(outputs)        

    outputs = tf.keras.layers.Dense(
        units=num_classes, name="outputs", activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
