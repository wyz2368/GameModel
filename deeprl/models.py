import tensorflow as tf
import tensorflow.contrib.layers as layers

def _mlp(hiddens,input,num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope,reuse=reuse):
        out = input
        for hidden in hiddens:
            out = layers.fully_connected(out,num_outputs=hidden,activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out,center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out,num_outputs=num_actions,activation_fn=None)
    return q_out

def mlp(hiddens=[], layer_norm=False):
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)