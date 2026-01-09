################################################################################
# AgriBlazeNet: Different Architecture Selection
################################################################################
import tensorflow as tf
import sys
sys.path.append("..")
from .sis_msi_blazenet import (BlazeImgClsModel, MobileNetV2)

'''-------------------------------------------------------------------MSI BlazeFace Image Classification Model------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def sis_msi_imgcls_model(input_shape, model_arch_module, model_arch_type, no_classes):
    if model_arch_module == "imgcls_blaze":
       model = BlazeImgClsModel(no_classes, cbam_block=None, imgreg=False)
       return model

    elif model_arch_module == "imgcls_blaze_cbwssam":
       model = BlazeImgClsModel(no_classes, cbam_block=model_arch_type, imgreg=False)
       return model

'''--------------------------------------------------------------------MSI MobileNetv2 Image Classification Model-----------------------------------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def sis_msi_imgcls_MobileNetV2_model(input_shape, model_arch_module, model_arch_type, no_classes):
    if model_arch_module == "imgcls_mv2":
       model = MobileNetV2(input_shape=input_shape, cbam_block=None, no_classes=no_classes, alpha=1.0, imgreg=False)
       return model

    elif model_arch_module == "imgcls_mv2_cbwssam":
       model = MobileNetV2(input_shape=input_shape, cbam_block=model_arch_type, no_classes=no_classes, alpha=1.0, imgreg=False)
       return model
'''--------------------------------------------------------------------MSI BlazeFace Image Regression Model-----------------------------------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def sis_msi_imgreg_model(input_shape, model_arch_module, model_arch_type, no_classes=1):
    if model_arch_module == "imgcls_blaze":
       model = BlazeImgClsModel(no_classes, cbam_block=None, imgreg=True)
       return model

    elif model_arch_module == "imgcls_blaze_cbwssam":
       model = BlazeImgClsModel(no_classes, cbam_block=model_arch_type, imgreg=True)
       return model

'''--------------------------------------------------------------------MSI MobileNetv2 Image Regression Model-----------------------------------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def sis_msi_imgreg_MobileNetV2_model(input_shape, model_arch_module, model_arch_type):
    if model_arch_module == "imgcls_mv2":
       model = MobileNetV2(input_shape=input_shape, cbam_block=None, no_classes=1, imgreg=True, alpha=1.0)
       return model

    elif model_arch_module == "imgcls_mv2_cbwssam":
       model = MobileNetV2(input_shape=input_shape, cbam_block=model_arch_type, no_classes=1, imgreg=True, alpha=1.0)
       return model
