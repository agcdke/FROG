################################################################################
# SIS: For No Conference
################################################################################

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Flatten, MaxPooling2D,
                                     AveragePooling2D, Dropout, BatchNormalization, LayerNormalization, DepthwiseConv2D, Permute, Layer,
                                     Concatenate, Conv2D, Conv1D, Add, Activation, Lambda, Rescaling, Normalization, MultiHeadAttention,
                                     Embedding, ReLU, LeakyReLU, Attention, Rescaling)
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
import sys, math
sys.path.append("..")

'''--------------------------------------------------------------------CBAM--------------------------------------------------------------------'''
# Vanilla CBAM Channel Attention Module
@tf.keras.utils.register_keras_serializable()
class ChannelAttention2D(Layer):
    def __init__(self, ratio=8, name="ChannelAttention2D", **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.add = Add()
        self.activation_sigmoid = Activation('sigmoid')
        self.multiply = Multiply()

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        self.shared_layer_1 = Dense(units = self.channel//self.ratio, activation="relu", kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_2 = Dense(units = self.channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.reshape = Reshape((1,1,self.channel))
        super().build(input_feature_shape)

    def call(self, input_feature):
        x_gap = self.gap(input_feature)
        x_gap = self.reshape(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel)
        x_gap = self.shared_layer_1(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel//self.ratio)
        x_gap = self.shared_layer_2(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel)

        x_gmp = self.gmp(input_feature)
        x_gmp = self.reshape(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel)
        x_gmp = self.shared_layer_1(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel//self.ratio)
        x_gmp = self.shared_layer_2(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel)

        channel_attn_feature = self.add([x_gap,x_gmp])
        channel_attn_feature = self.activation_sigmoid(channel_attn_feature)
        return self.multiply([input_feature, channel_attn_feature])

@tf.keras.utils.register_keras_serializable()
class ChannelwiseAvgPool2D(Layer):
    def __init__(self, axis=3, name="ChannelwiseAvgPool2D", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, self.axis, keepdims=True)

@tf.keras.utils.register_keras_serializable()
class ChannelwiseMaxPool2D(Layer):
    def __init__(self, axis=3, name="ChannelwiseMaxPool2D", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_max(inputs, self.axis, keepdims=True)

# Vanilla CBAM Spatial Attention Module
@tf.keras.utils.register_keras_serializable()
class SpatialAttention2D(Layer):
    def __init__(self, kernel_size=7, name="SpatialAttention2D", **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.channelwise_avg_pool = ChannelwiseAvgPool2D(axis=3)
        self.channelwise_max_pool = ChannelwiseMaxPool2D(axis=3)
        self.channelwise_concat = Concatenate(axis=3)
        self.conv_1 = Conv2D(filters = 1, kernel_size=self.kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.multiply = Multiply()

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        self.channel_attn_feature = input_feature
        x_avgpool = self.channelwise_avg_pool(self.channel_attn_feature)
        assert x_avgpool.shape[-1] == 1
        x_maxpool = self.channelwise_max_pool(self.channel_attn_feature)
        assert x_maxpool.shape[-1] == 1

        concat = self.channelwise_concat([x_avgpool, x_maxpool])
        assert concat.shape[-1] == 2
        spatial_attn_feature = self.conv_1(concat)
        assert spatial_attn_feature.shape[-1] == 1
        return self.multiply([input_feature, spatial_attn_feature])

@tf.keras.utils.register_keras_serializable()
class Cbam2D(Layer):
    def __init__(self, name="Cbam2D", **kwargs):
        super().__init__(**kwargs)
        self.channel_attn_feature = ChannelAttention2D(ratio=8)
        self.cbam_feature = SpatialAttention2D(kernel_size=7)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        x = self.channel_attn_feature(input_feature)
        x = self.cbam_feature(x)
        return x

'''---------------------------------------------------SIS-MobileViT-------------------------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, transformer_layers, projection_dim, num_heads, ff_dim, dropout_rate, name="TransformerBlock", **kwargs):
        super().__init__()
        self.transformer_layers = transformer_layers
        self.mh_attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(projection_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, inputs):
        x = inputs
        for _ in range(self.transformer_layers):
          attn_output = self.mh_attn(inputs, inputs)
          out1 = self.layernorm1(inputs + attn_output)
          ffn_output = self.ffn(out1)
          ffn_output = self.dropout2(ffn_output)
          x = self.layernorm2(out1 + ffn_output)
        return x

@tf.keras.utils.register_keras_serializable()
class SisMobileViT(Layer):
    def __init__(self, transformer_layers, projection_dim, num_heads, ff_dim, dropout_rate, strides, per_row_no_patch=4, name="SisMobileViT", **kwargs):
        super().__init__(**kwargs)
        self.per_row_no_patch = per_row_no_patch
        self.conv_local_feature = Conv2D(filters=projection_dim, kernel_size = (1, 1), padding="same", strides=strides)
        self.transformer_block = TransformerBlock(transformer_layers=transformer_layers, projection_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
        self.conv_folded_feature_map = Conv2D(filters=projection_dim, kernel_size = (1, 1), padding="same", strides=strides, activation=tf.keras.activations.swish)
        self.conv_local_global_features = Conv2D(filters=projection_dim, kernel_size=(3, 3), padding="same", strides=strides, activation=tf.keras.activations.swish)
        self.concat = Concatenate(axis=-1)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        self.feature_spatial_dim = input_feature_shape[1]
        print("self.channel, self.feature_spatial_dim: ", self.channel, " , ", self.feature_spatial_dim)
        self.patch_size = int(math.pow(self.feature_spatial_dim/self.per_row_no_patch,2))
        super().build(input_feature_shape)

    def call(self, input_feature):
        x = input_feature
        # Local projection
        local_features = self.conv_local_feature(x)
        # Unfold and Transformer
        num_patches = int((local_features.shape[1] * local_features.shape[2]) / self.patch_size)
        non_overlapping_patches = Reshape((self.patch_size, num_patches, self.channel))(local_features)
        global_features = self.transformer_block(non_overlapping_patches)
        # Fold into Conv-datacube
        folded_feature_map = Reshape((*local_features.shape[1:-1], self.channel))(global_features)
        # Point-wise Conv and Concat
        folded_feature_map = self.conv_folded_feature_map(folded_feature_map)
        local_global_features = self.concat([x, folded_feature_map]) # experiment
        # Fuse local and global features
        local_global_features = self.conv_local_global_features(local_global_features)
        return local_global_features

'''---------------------------------------------------CBwSSAM-------------------------------------------------------------------------------------'''
# CBwSSAM (MobileViT on Image Feature)
@tf.keras.saving.register_keras_serializable()
class MobileViT2D(Layer):
    def __init__(self, projection_dim, ff_dim, num_heads, num_blocks, strides, name="T0MobileViT2D", **kwargs):
        super().__init__(**kwargs)
        self.sis_mvit = SisMobileViT(transformer_layers=num_blocks, projection_dim=projection_dim, ff_dim=ff_dim, num_heads=num_heads, strides=strides, dropout_rate=0.1)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        spatial_selfattn_feature = self.sis_mvit(input_feature)
        assert spatial_selfattn_feature.shape[-1] == self.channel
        return spatial_selfattn_feature

# Type-1 CBwSSAM
@tf.keras.saving.register_keras_serializable()
class SpatialMobileViT2D(Layer):
    def __init__(self, projection_dim, ff_dim, num_heads, num_blocks, strides, name="T1CBwSSAM", **kwargs):
        super().__init__(**kwargs)
        self.sis_mvit = SisMobileViT(transformer_layers=num_blocks, projection_dim=projection_dim, ff_dim=ff_dim, num_heads=num_heads, strides=strides, dropout_rate=0.1)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        spatial_selfattn_feature = self.sis_mvit(input_feature)
        assert spatial_selfattn_feature.shape[-1] == self.channel
        return spatial_selfattn_feature

# Type-2 CBwSSAM
@tf.keras.saving.register_keras_serializable()
class ConvBlockSelfAttention2D(Layer):
    def __init__(self, projection_dim, ff_dim, num_heads, num_blocks, strides, kernel_size=7, name="T2CBwSSAM", **kwargs):
        super().__init__(**kwargs)
        self.channelwise_avg_pool = ChannelwiseAvgPool2D(axis=3)
        self.channelwise_max_pool = ChannelwiseMaxPool2D(axis=3)
        self.channelwise_concat = Concatenate(axis=3)
        self.conv_1 = Conv2D(filters = 1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.sis_mvit = SisMobileViT(transformer_layers=num_blocks, projection_dim=projection_dim, ff_dim=ff_dim, num_heads=num_heads, strides=strides, dropout_rate=0.1)
        self.multiply = Multiply()

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        # MViT on input_feature (Channel Attention Feature)
        channel_and_selfattn_feature = self.sis_mvit(input_feature)
        assert channel_and_selfattn_feature.shape[-1] == self.channel
        # CBAM Spatial Attention
        input_spatial_attn_feature = input_feature
        x_avgpool = self.channelwise_avg_pool(input_spatial_attn_feature)
        assert x_avgpool.shape[-1] == 1
        x_maxpool = self.channelwise_max_pool(input_spatial_attn_feature)
        assert x_maxpool.shape[-1] == 1

        concat = self.channelwise_concat([x_avgpool, x_maxpool])
        assert concat.shape[-1] == 2
        spatial_attn_feature = self.conv_1(concat)
        assert spatial_attn_feature.shape[-1] == 1
        # CBAM Feature
        cbam_feature = self.multiply([channel_and_selfattn_feature, spatial_attn_feature])
        assert cbam_feature.shape[-1] == self.channel
        return cbam_feature

'''------------------------------------------------------CBwSSAM Layer--------------------------------------------------------------------------------'''
# For Type-1 or Type-2 CBwSSAM, choose 'cbam' parameter Manually, e.g., cbam='t1mvit_h4b4' or cbam='t2cbmvit_h4b4'
@tf.keras.saving.register_keras_serializable()
class Cbwssam2D(Layer):
    def __init__(self, projection_dim, ff_dim, strides, cbam_block='mvit_h4b4',**kwargs):
        super().__init__(**kwargs)
        self.channel_attn_feature = ChannelAttention2D(ratio=8)
        self.cbam_info = cbam_block
        if cbam_block == 't2mvit_h4b4':
          self.cbam_feature = ConvBlockSelfAttention2D(projection_dim=projection_dim, ff_dim=ff_dim, kernel_size=7, num_heads=4, num_blocks=4, strides=strides)

        elif cbam_block == 't1mvit_h4b4':
          self.cbam_feature = SpatialMobileViT2D(projection_dim=projection_dim, ff_dim=ff_dim, num_heads=4, num_blocks=4, strides=strides)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        super().build(input_feature_shape)

    def call(self, input_feature):
        print("cbam_info: ", self.cbam_info)
        x = self.channel_attn_feature(input_feature)
        x = self.cbam_feature(x)
        return x


'''--------------------------------------------------------Blaze Block--------------------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class SingleBlazeBlock(Layer):
  def __init__(self, filters, strides=1, name="SingleBlazeBlock", **kwargs):
    super(SingleBlazeBlock, self).__init__(**kwargs)
    self.strides = strides
    self.filters = filters
    if strides == 2:
      self.pool = MaxPooling2D()

    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()
    self.activation_1 = ReLU()

  def build(self, input_feature_shape):
    self.channel = input_feature_shape[-1]
    super().build(input_feature_shape)

  def call(self, input_feature):
    x = self.dw_conv_1(input_feature)
    x = self.proj_conv_1(x)
    if self.strides == 2:
      input_feature = self.pool(input_feature)


    padding = self.filters - input_feature.shape[-1]
    if padding != 0:
       padding_values = [[0, 0], [0, 0], [0, 0], [0, padding]]
       input_feature = tf.pad(input_feature, paddings=padding_values)

    x = x + input_feature
    x = self.norm_1(x)
    x = self.activation_1(x)
    return x

@tf.keras.saving.register_keras_serializable()
class DoubleBlazeBlock(Layer):
  def __init__(self, proj_filters, expand_filters, strides=1, name="DoubleBlazeBlock", **kwargs):
    super(DoubleBlazeBlock, self).__init__(**kwargs)
    self.strides = strides
    self.proj_filters = proj_filters
    self.expand_filters = expand_filters
    if strides == 2:
      self.pool = MaxPooling2D()

    # Project
    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(proj_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()
    self.activation_1 = ReLU()

    # Expand (always strides=1)
    self.dw_conv_2 = DepthwiseConv2D(kernel_size=(5, 5), strides=1, padding="same")
    self.proj_conv_2 = Conv2D(expand_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_2 = BatchNormalization()
    self.activation_2 = ReLU()

  def build(self, input_feature_shape):
    self.channel = input_feature_shape[-1]
    super().build(input_feature_shape)

  def call(self, input_feature):
    x = self.dw_conv_1(input_feature)
    x = self.proj_conv_1(x)
    x = self.activation_1(x)
    if self.strides == 2:
      input_feature = self.pool(input_feature)

    x = self.dw_conv_2(x)
    x = self.proj_conv_2(x)


    padding = self.expand_filters - input_feature.shape[-1]
    if padding != 0:
       padding_values = [[0, 0], [0, 0], [0, 0], [0, padding]]
       input_feature = tf.pad(input_feature, paddings=padding_values)

    x = x + input_feature
    x = self.norm_2(x)
    x = self.activation_2(x)
    return x

'''--------------------------------------------------------Blaze Image Classification Model--------------------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class BlazeImgClsModel(Model):
  def __init__(self, no_classes, imgreg, cbam_block=None, name="BlazeImgClsModel", **kwargs):
    super(BlazeImgClsModel, self).__init__(**kwargs)
    self.cbam_block = cbam_block
    self.no_classes = no_classes
    self.imgreg = imgreg
    self.conv_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding="same")
    # Single BlazeBlocks
    self.activation_1 = ReLU()
    self.single_block_1 = SingleBlazeBlock(filters=24)
    self.single_block_2 = SingleBlazeBlock(filters=24)
    self.single_block_3 = SingleBlazeBlock(filters=48, strides=2)

    if self.cbam_block == 't0mvit_h4b4':
       self.cbwssam_1 = MobileViT2D(projection_dim=48, ff_dim=48, num_heads=4, num_blocks=4, strides=1)
    elif self.cbam_block == 't1mvit_h4b4':
       self.cbwssam_1 = Cbwssam2D(projection_dim=48, ff_dim=48, strides=1, cbam_block='t1mvit_h4b4')
    elif self.cbam_block == 't2mvit_h4b4':
       self.cbwssam_1 = Cbwssam2D(projection_dim=48, ff_dim=48, strides=1, cbam_block='t2mvit_h4b4')

    self.single_block_4 = SingleBlazeBlock(filters=48)
    # ICTAI paper mismatch with BlazeFace feature Extraction: "no stride" at 5th BlazeBlock in original BlazeFace
    self.single_block_5 = SingleBlazeBlock(filters=48)
    # Double BlazeBlocks
    self.double_block_1 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)

    if self.cbam_block == 't0mvit_h4b4':
       self.cbwssam_2 = MobileViT2D(projection_dim=96, ff_dim=96, num_heads=4, num_blocks=4, strides=1)
    elif self.cbam_block == 't1mvit_h4b4':
       self.cbwssam_2 = Cbwssam2D(projection_dim=96, ff_dim=96, strides=1, cbam_block='t1mvit_h4b4')
    elif self.cbam_block == 't2mvit_h4b4':
       self.cbwssam_2 = Cbwssam2D(projection_dim=96, ff_dim=96, strides=1, cbam_block='t2mvit_h4b4')

    self.double_block_2 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_3 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_4 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)

    if self.cbam_block == 't0mvit_h4b4':
       self.cbwssam_3 = MobileViT2D(projection_dim=96, ff_dim=96, num_heads=4, num_blocks=4, strides=1)
    elif self.cbam_block == 't1mvit_h4b4':
       self.cbwssam_3 = Cbwssam2D(projection_dim=96, ff_dim=96, strides=1, cbam_block='t1mvit_h4b4')
    elif self.cbam_block == 't2mvit_h4b4':
       self.cbwssam_3 = Cbwssam2D(projection_dim=96, ff_dim=96, strides=1, cbam_block='t2mvit_h4b4')

    self.double_block_5 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_6 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)

    self.gap = GlobalAveragePooling2D()
    if self.imgreg == True:
     self.dense = Dense(units=no_classes)
    else:
     self.dense = Dense(units=no_classes, activation = 'softmax')

  def get_config(self):
    config = super().get_config()
    config_params = {
      'no_classes': self.no_classes,
      'cbam_block': self.cbam_block,
      'imgreg': self.imgreg,
    }
    config.update(config_params)
    return config

  def call(self, x):
    x_batch, x_height, x_width, x_channel = x.shape
    print("cbam block: ", self.cbam_block)
    x = self.conv_1(x)
    x = self.activation_1(x)
    x = self.single_block_1(x)
    x = self.single_block_2(x)
    x = self.single_block_3(x)

    # Type-0: MobileViT
    if self.cbam_block == 't0mvit_h4b4':
      x = self.cbwssam_1(x)
    # Type-1: Type-1 CBwSSAM
    elif self.cbam_block == 't1mvit_h4b4':
      x = self.cbwssam_1(x)
    # Type-2: Type-2 CBwSSAM
    elif self.cbam_block == 't2mvit_h4b4':
      x = self.cbwssam_1(x)

    x = self.single_block_4(x)
    x = self.single_block_5(x)
    x = self.double_block_1(x)

    # Type-0: MobileViT
    if self.cbam_block == 't0mvit_h4b4':
      x = self.cbwssam_2(x)
    # Type-1: Type-1 CBwSSAM
    elif self.cbam_block == 't1mvit_h4b4':
      x = self.cbwssam_2(x)
    # Type-2: Type-2 CBwSSAM
    elif self.cbam_block == 't2mvit_h4b4':
      x = self.cbwssam_2(x)

    x = self.double_block_2(x)
    x = self.double_block_3(x)
    x = self.double_block_4(x)

    # Type-0: MobileViT
    if self.cbam_block == 't0mvit_h4b4':
      x = self.cbwssam_3(x)
    # Type-1: Type-1 CBwSSAM
    elif self.cbam_block == 't1mvit_h4b4':
      x = self.cbwssam_3(x)
    # Type-2: Type-2 CBwSSAM
    elif self.cbam_block == 't2mvit_h4b4':
      x = self.cbwssam_3(x)

    x = self.double_block_5(x)
    x = self.double_block_6(x)
    x = self.gap(x)
    x = self.dense(x)
    return x
'''--------------------------------------------------------End of AgriBlazeNet Model--------------------------------------------------------------------------------'''
'''--------------------------------------------------------MobileNet-V2 Image Classification Model------------------------------------------------------------------'''
"""
MobileNet v2 models for Keras.
# Code: https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py
"""
@tf.keras.utils.register_keras_serializable()
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@tf.keras.utils.register_keras_serializable()
def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

@tf.keras.utils.register_keras_serializable()
def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)

@tf.keras.utils.register_keras_serializable()
def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])
    return x

@tf.keras.utils.register_keras_serializable()
def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """
    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)
    return x

@tf.keras.utils.register_keras_serializable()
def MobileNetV2(input_shape, cbam_block, no_classes, imgreg, alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        no_classes: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].

    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)
    # No Normalization added
    x = inputs
    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(x, first_filters, (3, 3), strides=(2, 2)) # inputs
    # Added Attention Module
    if cbam_block == 't2mvit_h4b4':
      x = Cbwssam2D(projection_dim=first_filters, ff_dim=first_filters, strides=1, cbam_block='t2mvit_h4b4')(x)
    elif cbam_block == 't1mvit_h4b4':
      x = Cbwssam2D(projection_dim=first_filters, ff_dim=first_filters, strides=1, cbam_block='t1mvit_h4b4')(x)
    elif cbam_block == 't0mvit_h4b4':
      x = MobileViT2D(projection_dim=first_filters, ff_dim=first_filters, strides=1, num_heads=4, num_blocks=4)(x)
    else:
      x = x

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    # Added Attention Module
    if cbam_block == 't2mvit_h4b4':
      x = Cbwssam2D(projection_dim=32, ff_dim=16, strides=1, cbam_block='t2mvit_h4b4')(x)
    elif cbam_block == 't1mvit_h4b4':
      x = Cbwssam2D(projection_dim=32, ff_dim=16, strides=1, cbam_block='t1mvit_h4b4')(x)
    elif cbam_block == 't0mvit_h4b4':
      x = MobileViT2D(projection_dim=32, ff_dim=16, strides=1, num_heads=4, num_blocks=4)(x)
    else:
      x = x

    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)
    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_filters))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(no_classes, (1, 1), padding='same')(x)
    if imgreg == True:
      x = Activation('linear', name='linear')(x)
    else:
      x = Activation('softmax', name='softmax')(x)
    output = Reshape((no_classes,))(x)
    model = Model(inputs, output)
    return model
'''---------------------------------------------------------------End of MobileNet-V2 Model---------------------------------------------------------------------------------------'''
