################################################################################
# Utility Function
# Acknowledgement: wandb.ai for evaluation and plotting metrics info utility functions.
################################################################################

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn, itertools, time, os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, normalize)
from itertools import cycle
from tensorflow.keras.applications import (MobileNetV2)
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Flatten, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization,
                                     Permute, Concatenate, Conv2D, Add, Activation, Lambda, Normalization)
from tensorflow.keras import (Model)
from tensorflow.keras.utils import (to_categorical)
from sklearn.preprocessing import (label_binarize, LabelBinarizer, LabelEncoder, OneHotEncoder)
from skimage.transform import resize
import sys
import tifffile as tiff
sys.path.append("..")

'''-------------------------------FLOPS---------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops

@tf.keras.utils.register_keras_serializable()
def plot_acc_loss(history, plot_filename, title_type='trainval'):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    if title_type == 'trainval':
     plt.title('Training and Validation Loss & Accuracy')
    else:
     plt.title('Training Loss & Accuracy')
    plt.savefig(fname=plot_filename)
    plt.clf()
    plt.close()

'''----------------------------Get (train,val,test) Multi-Spectral Classification Dataset---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def get_img_label_nparray(img_dirpath, band_config, csv_file, standard_scalar, minmax_scalar, target_shape):
  df = pd.read_csv(csv_file)
  images = []
  labels = []
  for idx, row in df.iterrows():
    img_name = df.loc[idx,'Filename']
    label = df.loc[idx,'ClassName']
    tif_img = tiff.imread(os.path.join(img_dirpath,img_name))
    if standard_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=True, minmax_scalar=False, target_shape=target_shape)
    elif minmax_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=True, target_shape=target_shape)
    else:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=False, target_shape=target_shape)
    images.append(img)
    labels.append(label)

  # Encode string labels as integers
  encoder = LabelEncoder()
  labels_int = encoder.fit_transform(labels)
  images_np = np.array(images)
  labels_np = np.array(labels_int)
  return images_np, labels_np

'''----------------------------Get (train,val,test) Multi-Spectral BrixIndex Dataset---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def get_img_brixindex_nparray(img_dirpath, band_config, csv_file, standard_scalar, minmax_scalar, target_shape):
  df = pd.read_csv(csv_file)
  images = []
  labels = []
  for idx, row in df.iterrows():
    img_name = df.loc[idx,'Filename']
    label = df.loc[idx,'BrixIndex']
    tif_img = tiff.imread(os.path.join(img_dirpath,img_name))
    if standard_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=True, minmax_scalar=False, target_shape=target_shape)
    elif minmax_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=True, target_shape=target_shape)
    else:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=False, target_shape=target_shape)
    images.append(img)
    labels.append(label)

  # Create Numpy arrays
  images_np = np.array(images)
  labels_np = np.array(labels)
  return images_np, labels_np

'''----------------------------Get (train,val,test) Multi-Spectral Anthocyanines Dataset---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def get_img_anthocyanines_nparray(img_dirpath, band_config, csv_file, standard_scalar, minmax_scalar, target_shape):
  df = pd.read_csv(csv_file)
  images = []
  labels = []
  for idx, row in df.iterrows():
    img_name = df.loc[idx,'Filename']
    label = df.loc[idx,'Anthocyanines']
    tif_img = tiff.imread(os.path.join(img_dirpath,img_name))
    if standard_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=True, minmax_scalar=False, target_shape=target_shape)
    elif minmax_scalar == True:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=True, target_shape=target_shape)
    else:
      img = get_tif_img(tif_img, band_config, standard_scalar=False, minmax_scalar=False, target_shape=target_shape)
    images.append(img)
    labels.append(label)

  # Create Numpy arrays
  images_np = np.array(images)
  labels_np = np.array(labels)
  return images_np, labels_np

'''----------------------------Read Multi-Spectral Image Dataset---------------------------------'''
# Following the GDAL convention, these are indexed starting with the number 1.
@tf.keras.utils.register_keras_serializable()
def raster_read_scalar(tif_img, band, standard_scalar, minmax_scalar, target_shape):
  tif_img = tif_img[:,:,band]
  resized_img = resize(tif_img, target_shape, anti_aliasing=True)
  np_img = (resized_img/255).astype(np.float32)
  # print("img: ", np_img.shape)
  if standard_scalar== True:
    np_img = StandardScaler().fit_transform(np_img)
  elif minmax_scalar==True:
    np_img = MinMaxScaler().fit_transform(np_img)
  return np_img

'''
def raster_read_normalise(raster, band):
  raster_img = raster.read(band)
  np_img = (raster_img/255).astype(np.float32)
  # print("np_img.shape: ", np_img.shape)
  # Check image shape
  n_channels, img_h, img_w = np_img.shape
  # Reshape to (pixels, img_channels)
  pixels = np_img.reshape(-1, n_channels)
  # Normalize each pixel vector to L2 unit norm (axis=1)
  normalized_pixels = normalize(pixels, norm='l2', axis=1)
  # Reshape back
  np_img = normalized_pixels.reshape(img_h, img_w, n_channels)
  # print("np_img.shape: ", np_img.shape)
  return np_img
'''
@tf.keras.utils.register_keras_serializable()
def get_tif_img(tif_img, band_config, standard_scalar, minmax_scalar, target_shape):
    temp_arr_list = list()
    if (standard_scalar==False and minmax_scalar==False):
      tif_img = tif_img[:,:,:]
      resized_img = resize(tif_img, target_shape, anti_aliasing=True)
      np_img = (resized_img/255).astype(np.float32)
      temp_arr_list.append(np_img)
      img = np.dstack(temp_arr_list) # all channels
    else:
      # band_config: based on index-i
      for i in range(0,len(band_config)):
        # print("band_config: ", band_config[i])
        temp_arr = raster_read_scalar(tif_img, band_config[i], standard_scalar, minmax_scalar, target_shape)
        temp_arr_list.append(temp_arr)
      # print("len(temp_arr: ", len(temp_arr_list))
      img = np.dstack(temp_arr_list) # all/selected channels at band_config
    return img

'''----------------------------Get (Mean,Std-Dev) for Multi-Spectral Dataset---------------------------------'''
'''
def get_eurosat_mean_std(no_channel, band_config, test_band_config, train_ds_path, val_ds_path, test_ds_path, apply_pca=False):
    per_fold_num_images, mean, std = 0, 0., 0.
    if apply_pca == False:
      train_ds, train_classes = get_eurosat_msi_dataset(train_ds_path, band_config, apply_pca=False)
      val_ds, val_classes = get_eurosat_msi_dataset(val_ds_path, band_config, apply_pca=False)
      test_ds, test_classes= get_eurosat_msi_dataset(test_ds_path, test_band_config, apply_pca=False)
    else:
      train_ds, train_classes = get_eurosat_msi_dataset(train_ds_path, band_config, apply_pca=True)
      val_ds, val_classes = get_eurosat_msi_dataset(val_ds_path, band_config, apply_pca=True)
      test_ds, test_classes= get_eurosat_msi_dataset(test_ds_path, test_band_config, apply_pca=True)

    print("ds_info: ", type(train_ds), " , ", type(val_ds), ", ", type(test_ds))
    train_val_ds = train_ds.concatenate(val_ds)

    train_images, train_labels = tuple(zip(*train_ds))
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print("train: ", train_images.shape, " , ", train_labels.shape, " , ", type(train_images))

    val_images, val_labels = tuple(zip(*val_ds))
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    print("val: ", val_images.shape, " , ", val_labels.shape, " , ", type(val_images))

    test_images, test_labels = tuple(zip(*test_ds))
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print("test: ", test_images.shape, " , ", test_labels.shape, " , ", type(test_images))

    train_val_images, train_val_labels = tuple(zip(*train_val_ds))
    train_val_images = np.array(train_val_images)
    train_val_labels = np.array(train_val_labels)
    print("train_val: ", train_val_images.shape, " , ", train_val_labels.shape, " , ", type(train_val_images))

    per_fold_num_images += len(train_images)
    i = 1
    for img in train_images:
        if (isinstance(img, np.ndarray)):
            img = (img / 255).astype(np.float32)
            img = np.reshape(np.transpose(img, (2, 0, 1)), (no_channel, -1))
            mean += img.mean(-1)
            std += img.std(-1)

    mean = tuple((mean / per_fold_num_images).tolist())
    std = tuple((std / per_fold_num_images).tolist())
    mean_list = [round(elem, 3) for elem in mean ]
    std_list = [round(elem, 3) for elem in std ]
    return mean_list, std_list, per_fold_num_images
'''
'''----------------------------Augment Dataset---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def augment_dataset(ds, data_augmentation, augment=None, AUTOTUNE=tf.data.AUTOTUNE):
  # Use image augmentation only on the training set.
  if augment == True:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

  elif augment == False:
    # Use buffered prefetching on datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

'''----------------------------Evaluate Dataset---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def evaluate_ds(test_dataloader, model, batch_size):
  true_labels = []
  raw_preds = []
  thres_preds = []
  for imgs, labels in iter(test_dataloader):
    preds = model.predict(imgs, batch_size=batch_size)
    true_labels.extend(labels)
    raw_preds.extend(preds.copy())
    preds = np.argmax(preds, axis=1)
    thres_preds.extend(preds)
  return np.array(true_labels), np.array(raw_preds), np.array(thres_preds)

'''----------------------------Plot---------------------------------'''
@tf.keras.utils.register_keras_serializable()
def plot_cm(cm, classes, filename, normalize=False, title='', cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  #plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')
  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)
  plt.clf()
  plt.close()

@tf.keras.utils.register_keras_serializable()
def plot_roc_binary(test_labels, predictions, filename, roconly_filename, n_class, cm_plot_labels, dataset='test'):

  print("cm_plot_labels: ", cm_plot_labels)

  if n_class == 2:
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    y_test_int_encoded = label_encoder.fit_transform(test_labels)
    y_pred_int_encoded = label_encoder.fit_transform(predictions)

    y_test_int_encoded = y_test_int_encoded.reshape(len(y_test_int_encoded), 1)
    y_pred_int_encoded = y_pred_int_encoded.reshape(len(y_pred_int_encoded), 1)

    y_test = onehot_encoder.fit_transform(y_test_int_encoded)
    y_pred = onehot_encoder.fit_transform(y_pred_int_encoded)

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()

  for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("For class-",i,": ", fpr[i], " and ", tpr[i], " and also ", roc_auc[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel(), pos_label=1)
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
  # Compute macro-average ROC curve and ROC area First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_class

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  print(f"Macro-averaged One-vs-Rest ROC AUC score:  {roc_auc['macro']:.2f}")

  # Plot linewidth.
  lw = 2
  # Plot all ROC curves
  plt.figure(1)
  plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
  plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig(roconly_filename)

  for i, color in zip(range(n_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  if dataset == 'train':
    plt.title('ROC Curve to multi-class Train dataset')
  elif dataset == 'val':
    plt.title('ROC Curve to multi-class Val dataset')
  else:
    plt.title('ROC Curve to multi-class Test dataset')
  plt.legend(loc="lower right")
  plt.savefig(filename)
  plt.clf()
  plt.close()

@tf.keras.utils.register_keras_serializable()
def plot_roc(test_labels, predictions, filename, roconly_filename, n_class, cm_plot_labels, dataset='test'):

  print("cm_plot_labels: ", cm_plot_labels)

  if n_class == 2:
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    y_test_int_encoded = label_encoder.fit_transform(test_labels)
    y_pred_int_encoded = label_encoder.fit_transform(predictions)

    y_test_int_encoded = y_test_int_encoded.reshape(len(y_test_int_encoded), 1)
    y_pred_int_encoded = y_pred_int_encoded.reshape(len(y_pred_int_encoded), 1)
    y_test = onehot_encoder.fit_transform(y_test_int_encoded)
    y_pred = onehot_encoder.fit_transform(y_pred_int_encoded)

  if n_class > 2:
    print("\n----------++++++++++----------++++++++++----------\n")
    print("label_binarize: ", label_binarize(y=cm_plot_labels, classes=cm_plot_labels))
    print("\n----------++++++++++----------++++++++++----------\n")
    y_test = label_binarize(test_labels, classes=np.arange(n_class))
    y_pred = label_binarize(predictions, classes=np.arange(n_class))


  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()

  for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i],)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("For class-",i,": ", fpr[i], " and ", tpr[i], " and also ", roc_auc[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
  # Compute macro-average ROC curve and ROC area First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_class

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  print(f"Macro-averaged One-vs-Rest ROC AUC score:  {roc_auc['macro']:.2f}")

  # Plot linewidth.
  lw = 2
  # Plot all ROC curves
  plt.figure(1)
  plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
  plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig(roconly_filename)

  for i, color in zip(range(n_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  if dataset == 'train':
    plt.title('ROC Curve to multi-class Train dataset')
  elif dataset == 'val':
    plt.title('ROC Curve to multi-class Val dataset')
  else:
    plt.title('ROC Curve to multi-class Test dataset')
  plt.legend(loc="lower right")
  plt.savefig(filename)
  plt.clf()
  plt.close()
'''----------------------------End of Utility---------------------------------'''
