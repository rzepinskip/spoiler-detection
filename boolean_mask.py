import tensorflow as tf

all_out = tf.ones([1, 2, 3])
P_mask = tf.constant([[True, False]])
_P_mask = tf.cast(P_mask, tf.float32)

_P_mask_ = tf.broadcast_to(_P_mask, shape=(tf.shape(all_out)[2],tf.shape(all_out)[0],tf.shape(all_out)[1]))
P_mask_ = tf.transpose(_P_mask_, perm=[1,2,0])
P_ = tf.multiply(all_out, P_mask_)
P_