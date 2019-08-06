from collections import namedtuple

import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc
from baselines.common.models import register

NetworkOutput = namedtuple('NetworkOutput', 'policy_latent recurrent_tensors extra')


def softmax_pixelwise(tensor):
    b, h, w, c = tensor.shape
    tensor = tf.reshape(tensor, (-1, h * w, c))
    tensor = tf.nn.softmax(tensor, axis=1)
    tensor = tf.reshape(tensor, (-1, h, w, c))
    return tensor


def sparse_block(x, relu=True):
    init = {'init_scale': np.sqrt(2)}
    x = conv(x, 'c1', nf=32, rf=8, stride=4, **init)
    x = tf.nn.relu(x)
    x = conv(x, 'c2', nf=64, rf=4, stride=2, **init)
    x = tf.nn.relu(x)
    x = conv(x, 'c3', nf=64, rf=3, stride=1, **init)
    if relu:
        x = tf.nn.relu(x)
    return x


def dense_block(x, relu=True):
    init = {'init_scale': np.sqrt(2)}
    x = conv(x, 'c1', nf=32, rf=7, stride=1, pad='SAME', **init)
    x = tf.nn.relu(x)
    x = conv(x, 'c2', nf=64, rf=5, stride=1, pad='SAME', **init)
    x = tf.nn.relu(x)
    x = conv(x, 'c3', nf=64, rf=3, stride=1, pad='SAME', **init)
    if relu:
        x = tf.nn.relu(x)
    return x


def fls_module(x, filter_size=3, base2=False):
    init = {'init_scale': np.sqrt(2)}
    h = conv(x, 'h1', nf=256, rf=filter_size, stride=1, pad='SAME', **init)
    h = tf.nn.relu(h)
    h = conv(h, 'h2', nf=1, rf=filter_size, stride=1, pad='SAME', **init)
    if base2:
        log2 = np.log(2)
        h = tf.nn.softplus(h * log2) / log2
    else:
        h = tf.nn.softplus(h)
    assert h.shape[-1] == 1
    return h


def final_linear(x):
    init = {'init_scale': np.sqrt(2)}
    x = fc(x, 'fc1', nh=512, **init)
    x = tf.nn.relu(x)
    return x


@register("cnn")
def cnn():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = tf.zeros(shape=tf.shape(x)[:-1])

        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_daqn')
def cnn_daqn():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        init = {'init_scale': np.sqrt(2)}
        h = conv(x, 'h1', nf=256, rf=1, stride=1, pad='SAME', **init)
        h = tf.nn.tanh(h)
        h = conv(h, 'h2', nf=1, rf=1, stride=1, pad='SAME', **init)
        h = softmax_pixelwise(h)
        h = tf.squeeze(h, axis=3)

        x = tf.einsum('bhwc,bhw->bc', x, h)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_rsppo')
def cnn_rsppo():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        init = {'init_scale': np.sqrt(2)}

        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = sparse_block(x)
        x = tf.nn.l2_normalize(x, axis=3)

        h = conv(x, 'h1', nf=512, rf=1, stride=1, pad='SAME', **init)
        h = tf.nn.elu(h)
        h = conv(h, 'h2', nf=2, rf=1, stride=1, pad='SAME', **init)
        h = softmax_pixelwise(h)
        h = tf.reduce_sum(h, axis=-1, keepdims=True)

        x = x * h
        x = tf.reshape(x, (-1, 3136))

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_rsppo_nopad')
def cnn_rsppo_nopad():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        init = {'init_scale': np.sqrt(2)}

        x = sparse_block(x)
        x = tf.nn.l2_normalize(x, axis=3)

        h = conv(x, 'h1', nf=512, rf=1, stride=1, pad='SAME', **init)
        h = tf.nn.elu(h)
        h = conv(h, 'h2', nf=2, rf=1, stride=1, pad='SAME', **init)
        h = softmax_pixelwise(h)
        h = tf.reduce_sum(h, axis=-1, keepdims=True)

        x = x * h
        x = tf.reshape(x, (-1, 3136))

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls')
def cnn_sparse_fls():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = fls_module(x)

        x = x * h
        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_pool')
def cnn_sparse_fls_pool():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = fls_module(x)
        h = tf.squeeze(h, axis=3)

        x = tf.einsum('bhwc,bhw->bc', x, h)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_norm')
def cnn_sparse_fls_norm():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = fls_module(x)
        h = h / tf.reduce_sum(h, axis=[1, 2, 3], keepdims=True)

        x = x * h
        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_1x1')
def cnn_sparse_fls_1x1():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = fls_module(x, filter_size=1)

        x = x * h
        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_sp2')
def cnn_sparse_fls_sp2():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x)

        h = fls_module(x, base2=True)

        x = x * h
        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_norelu')
def cnn_sparse_fls_norelu():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x, relu=False)

        h = fls_module(x)

        x = x * h
        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_norelu_pool')
def cnn_sparse_fls_norelu_pool():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = sparse_block(x, relu=False)

        h = fls_module(x)
        h = tf.squeeze(h, axis=3)

        x = tf.einsum('bhwc,bhw->bc', x, h)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_sparse_fls_h1')
def cnn_sparse_fls_h1():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        init = {'init_scale': np.sqrt(2)}

        x = conv(x, 'c1', nf=32, rf=8, stride=4, **init)
        x = tf.nn.relu(x)
        with tf.variable_scope('h1'):
            h1 = fls_module(x)
        x = x * h1

        x = conv(x, 'c2', nf=64, rf=4, stride=2, **init)
        x = tf.nn.relu(x)

        x = conv(x, 'c3', nf=64, rf=3, stride=1, **init)
        x = tf.nn.relu(x)

        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h1,
        )

    return network_fn


@register('cnn_sparse_fls_x3')
def cnn_sparse_fls_x3():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        init = {'init_scale': np.sqrt(2)}

        x = conv(x, 'c1', nf=32, rf=8, stride=4, **init)
        x = tf.nn.relu(x)
        with tf.variable_scope('h1'):
            h1 = fls_module(x)
        x = x * h1

        x = conv(x, 'c2', nf=64, rf=4, stride=2, **init)
        x = tf.nn.relu(x)
        with tf.variable_scope('h2'):
            h2 = fls_module(x)
        x = x * h2

        x = conv(x, 'c3', nf=64, rf=3, stride=1, **init)
        x = tf.nn.relu(x)
        with tf.variable_scope('h3'):
            h3 = fls_module(x)
        x = x * h3

        h21 = tf.nn.conv2d_transpose(
            h2,
            tf.ones(shape=(4, 4, 1, 1), dtype=tf.float32),
            output_shape=tf.shape(h1),
            strides=(1, 2, 2, 1),
            padding='VALID',
            data_format='NHWC',
        )

        h31 = tf.nn.conv2d_transpose(
            h3,
            tf.ones(shape=(8, 8, 1, 1), dtype=tf.float32),
            output_shape=tf.shape(h1),
            strides=(1, 2, 2, 1),
            padding='VALID',
            data_format='NHWC',
        )

        h = h1 + h21 + h31

        x = conv_to_fc(x)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_dense_fls')
def cnn_dense_fls():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = dense_block(x)

        h = fls_module(x)
        h = tf.squeeze(h, axis=3)

        x = tf.einsum('bhwc,bhw->bc', x, h)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


@register('cnn_dense_fls_norelu')
def cnn_dense_fls_norelu():
    def network_fn(X):
        x = tf.cast(X, tf.float32) / 255.

        x = dense_block(x, relu=False)

        h = fls_module(x)
        h = tf.squeeze(h, axis=3)

        x = tf.einsum('bhwc,bhw->bc', x, h)

        x = final_linear(x)

        return NetworkOutput(
            policy_latent=x,
            recurrent_tensors=None,
            extra=h,
        )

    return network_fn


def make_params():
    dicts = [
        {
            model_name: {
                'receptive_field': 36,
                'stride': 8,
                'padding': 0,
            }
            for model_name in [
                'cnn',
                'cnn_daqn',
                'cnn_rsppo',
                'cnn_rsppo_nopad',
                'cnn_sparse_fls',
                'cnn_sparse_fls_pool',
                'cnn_sparse_fls_norm',
                'cnn_sparse_fls_1x1',
                'cnn_sparse_fls_sp2',
                'cnn_sparse_fls_norelu',
                'cnn_sparse_fls_norelu_pool',
        ]},
        {
            model_name: {
                'receptive_field': 8,
                'stride': 4,
                'padding': 0,
            }
            for model_name in [
                'cnn_sparse_fls_h1',
                'cnn_sparse_fls_x3',
        ]},
        {
            model_name: {
                'receptive_field': 13,
                'stride': 1,
                'padding': 6,
            }
            for model_name in [
                'cnn_dense_fls',
                'cnn_dense_fls_norelu',
        ]},
    ]

    return {
        model_name: model_params
        for d in dicts
        for model_name, model_params in d.items()
    }


attention_visualization_params = make_params()
