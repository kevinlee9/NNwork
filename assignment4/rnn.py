# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Config:
    hidden_size = 64
    num_nodes = 64
    dropout = 0.5
    num_layers = 2
    num_classes = 3
    learning_rate = 0.5 * 1e-4
    beta1 = 0.9
    max_steps = 1000
    padding_size = 270
    batch_size = 80

params = Config()


class RNN:
    def build(self):
        final_state = self._add_rnn_layers(self.X, self.lengths)

        logits = self._add_fc_layers(final_state)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            # labels=tf.one_hot(self.y, params.num_classes), logits=logits
            labels=self.y, logits=logits
        ))
        self.loss = cross_entropy
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _get_input_tensor(self):
        pass

    def _get_lstm_cell(self, config, is_training):
        return tf.contrib.rnn.BasicLSTMCell(
            config.hidden_size, forget_bias=1.0, state_is_tuple=True,
            reuse=not is_training)

    def _add_regular_rnn_layers(self, convolved, lengths):
        """Adds RNN layers."""
        cell = tf.nn.rnn_cell.BasicLSTMCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
        outputs, fw, bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lengths,
            dtype=tf.float32,
            scope="rnn_classification")
        return outputs, fw, bw

    def _add_rnn_layers(self, convolved, lengths):
        """Adds recurrent neural network layers depending on the cell type."""
        outputs, fw, bw = self._add_regular_rnn_layers(convolved, lengths)
        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
        # the number of nodes in the last layer.
        # mask = tf.tile(
        #     tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
        #     [1, 1, tf.shape(outputs)[2]])
        # zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        # outputs, last_states = tf.reduce_sum(outputs, axis=1)
        # outputs = tf.gather_nd(outputs, lengths)
        return tf.concat([fw[1].h, bw[1].h], -1)

    def _add_fc_layers(self, final_state):
        """Adds a fully connected layer."""
        return tf.layers.dense(final_state, params.num_classes)

    def __init__(self, time_step):
        self.time_step = time_step
        self.X = tf.placeholder('float32', shape=[None, None, 310], name='input_data')
        self.y = tf.placeholder('int64', shape=[None], name='labels')
        self.lengths = tf.placeholder('int64', shape=[None], name='lengths')
        self.build()

