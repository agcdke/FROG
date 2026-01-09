#################################################################################
#  Multi-Spectral BlazeFace and MobileNetv2
################################################################################
import os
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import (CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, classification_report)
import datetime, gc, time, json
import pandas as pd
from sisutils.msi_utils import (augment_dataset, plot_acc_loss, evaluate_ds, plot_cm, plot_roc, get_img_brixindex_nparray)
from sisutils.sis_msi_model import (sis_msi_imgreg_model, sis_msi_imgreg_MobileNetV2_model)
from sklearn.utils.class_weight import compute_class_weight

SEED_VAL=np.random.randint(100)
print("TensorFlow Version and seed: ", tf.__version__, " and ", SEED_VAL)

# Dir. Structure - "Train-Val-Test"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

CONFIG = dict (
    run = 13,
    model_arch_module = 'imgcls_mv2_cbwssam', # model: imgcls_blaze/mv2_cbwssam (T0,T1,T2), NA (imgcls_blaze/_mv2)
    model_arch_type = 't1mvit_h4b4', # NA, (T0) t0mvit_h4b4, (T1) t1mvit_h4b4, (T2) t2mvit_h4b4
    img_width = 128, # (hxw)
    img_height = 128,
    band_config = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    # band_config = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], # IR: 25
    # band_config = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # Visible: 12
    channels = 37, # MSI Channels: channels based on band_config (sisutils/utils.py), channels=37(all), 12(Visible), 25(IR)
    batch_size = 32, # batch: (batch_size x 2 GPU)
    epochs = 100,
    learning_rate = 0.001,
    min_delta = 0.0001,
    reduce_lr_factor = 0.2,
    class_weight = {0: 1.29, 1: 0.64, 2: 3.05, 3: 0.51, 4: 2.71},
    img_dirpath = '../../../share/data/plantscience/msi/GrapesTDM/MSI_GrapesTDM/',
    label_dirpath = '../../../share/data/plantscience/msi/GrapesTDM/Labels_MSI_GrapesTDM/',
    modelweight_parentdir = 'weight/imgcls/brixregmsi128/',
    kerasweight_parentdir = 'weight/keras/brixregmsi128/',
    model_logdir = 'metricinfo/brixregmsi128/',
    standard_scalar = True,
    minmax_scalar = False,
)

tf.keras.utils.set_random_seed(SEED_VAL)
tf.config.experimental.enable_op_determinism()
AUTOTUNE = tf.data.experimental.AUTOTUNE



if __name__ == '__main__':

    exp_name='tdm_brix_' + str(CONFIG['model_arch_module']) + "_" + str(CONFIG['model_arch_type'])  + "_img_" + str(CONFIG['img_width']) + "_batch_"  + str(CONFIG['batch_size']) + "_run_" + str(CONFIG['run'])
    print("Experiment Name: ", exp_name)
    # Basic Config Information
    print("model_architecture, img_size, batch_size: ", CONFIG['model_arch_module'], " , img=(", CONFIG['img_width'] , " X ", CONFIG['img_height'], "), bs=", CONFIG['batch_size'])
    print("standard_scalar = ", CONFIG['standard_scalar'], ", minmax_scalar = ", CONFIG['minmax_scalar'])
    # Image Augmentation
    train_data_augmentation = tf.keras.Sequential([layers.RandomFlip(mode="horizontal_and_vertical", seed=42)])

    # Model Artefacts
    model_artefact_dir = str(CONFIG['model_logdir'])
    model_log_dir = Path(model_artefact_dir).mkdir(parents=True, exist_ok=True)
    plot_filename = model_artefact_dir + "Acc_Loss_{}.png".format(exp_name)
    logfile = model_artefact_dir + "Log_{}.csv".format(exp_name)
    log_dir = model_artefact_dir + "logs/{}".format(exp_name)
    plot_valcm_filename = model_artefact_dir + "Val_CM_{}.png".format(exp_name)
    plot_valroc_filename = model_artefact_dir + "Val_ROC_{}.png".format(exp_name)
    plot_valroconly_filename = model_artefact_dir + "Val_ROC_only_{}.png".format(exp_name)
    plot_testcm_filename = model_artefact_dir + "Test_CM_{}.png".format(exp_name)
    plot_testroc_filename = model_artefact_dir + "Test_ROC_{}.png".format(exp_name)
    plot_testroconly_filename = model_artefact_dir + "Test_ROC_only_{}.png".format(exp_name)

    # Model weight in keras format
    modelweight_dirpath = str(CONFIG['modelweight_parentdir'])
    modelweight_dir = Path(modelweight_dirpath).mkdir(parents=True, exist_ok=True)
    weight_filepath = modelweight_dirpath + "{}.keras".format(exp_name)

    kerasweight_dirpath = str(CONFIG['kerasweight_parentdir'])
    kerasweight_dir = Path(kerasweight_dirpath).mkdir(parents=True, exist_ok=True)
    kerasweight_filepath = kerasweight_dirpath + "{}.keras".format(exp_name)

    # Logging
    csv_logger = CSVLogger(logfile, append=True, separator=';')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Multi-GPU Strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPUs in workstation: {}'.format(strategy.num_replicas_in_sync))
    mgpu_batch_size = int(CONFIG['batch_size']) * strategy.num_replicas_in_sync
    print("Multi-GPU Batch Size: ", mgpu_batch_size)

    # Datasets
    label_map_file = str(Path(CONFIG['label_dirpath']+'label_map.json'))
    with open(label_map_file, "r") as f:
      class_names_encoded = json.load(f)
    class_names = list(class_names_encoded.keys())
    num_classes = len(class_names)
    print("Sanity Check class_names_encoded:\n", class_names_encoded)

    train_csvfile = str(Path(CONFIG['label_dirpath']+'train_imgreg.csv'))
    val_csvfile = str(Path(CONFIG['label_dirpath']+'val_imgreg.csv'))
    test_csvfile = str(Path(CONFIG['label_dirpath']+'test_imgreg.csv'))

    # Create Train, Val, Test datasets
    X_train, y_train = get_img_brixindex_nparray(CONFIG['img_dirpath'], CONFIG['band_config'], csv_file=train_csvfile, standard_scalar=CONFIG['standard_scalar'],
                                             minmax_scalar=CONFIG['minmax_scalar'], target_shape=(CONFIG['img_height'], CONFIG['img_width']))
    X_val, y_val = get_img_brixindex_nparray(CONFIG['img_dirpath'], CONFIG['band_config'], csv_file=val_csvfile, standard_scalar=CONFIG['standard_scalar'],
                                         minmax_scalar=CONFIG['minmax_scalar'], target_shape=(CONFIG['img_height'], CONFIG['img_width']))
    X_test, y_test = get_img_brixindex_nparray(CONFIG['img_dirpath'], CONFIG['band_config'], csv_file=test_csvfile, standard_scalar=CONFIG['standard_scalar'],
                                           minmax_scalar=CONFIG['minmax_scalar'], target_shape=(CONFIG['img_height'], CONFIG['img_width']))

    print("Shape (Train/Val/Test): ", X_train.shape, y_train.shape, " / ", X_val.shape, y_val.shape, " / ", X_test.shape, y_test.shape)
    print("Some Labels (Train/Val/Test): ", y_train[0:5], " / ", y_val[0:5], " / ", y_test[0:5])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(buffer_size=1000)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(buffer_size=1000)

    # Image Augmentation
    aug_train_ds = augment_dataset(ds=train_ds, data_augmentation=train_data_augmentation, augment=True)
    train_ds = aug_train_ds.batch(CONFIG['batch_size']).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(CONFIG['batch_size']).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(CONFIG['batch_size']).prefetch(AUTOTUNE)
    print("Dataset Cardinality(train, val, test): ", train_ds.cardinality(), " , ", val_ds.cardinality(), " , ", test_ds.cardinality())

    # Other Params
    model_save_checkpoint = ModelCheckpoint(weight_filepath, save_best_only=True, save_weights_only=False, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=CONFIG['min_delta'], verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=CONFIG['reduce_lr_factor'], patience=4, min_delta=CONFIG['min_delta'], mode='min', verbose=1)

    input_shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels'])
    print("input_shape: ", input_shape)
    # Model MGPU-Strategy Scope
    with strategy.scope():
        
        model = sis_msi_imgreg_model(input_shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']), model_arch_module=CONFIG['model_arch_module'], model_arch_type=CONFIG['model_arch_type'])
        '''
        model = sis_msi_imgreg_MobileNetV2_model(input_shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']), model_arch_module=CONFIG['model_arch_module'],
                                                 model_arch_type=CONFIG['model_arch_type'])
        '''
        print("Model Summary:")
        model.build(input_shape=(None,CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']))
        print(model.summary())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0),
                      metrics=['mean_absolute_error', 'mean_squared_error', tf.keras.metrics.RootMeanSquaredError()])

    # Model Fit with Batch Size
    history = model.fit(train_ds, validation_data = val_ds, epochs = CONFIG['epochs'], verbose = 1, batch_size = mgpu_batch_size,
                        callbacks = [model_save_checkpoint, early_stop, reduce_lr, csv_logger, tensorboard_callback])

    # Model save Keras weight
    model.save(kerasweight_filepath)

    # Plot Metrics
    plot_acc_loss(history, plot_filename = plot_filename, title_type = 'trainval')
    val_score = model.evaluate(val_ds, batch_size=mgpu_batch_size, verbose="auto", return_dict=True)
    print("Valset Evaluation Score: ", val_score)
    test_score = model.evaluate(test_ds, batch_size=mgpu_batch_size, verbose="auto", return_dict=True)
    print("Testset Evaluation Score: ", test_score)
    print("------------------------------------------------------------------------")
    # Calculate Inference Time
    N_warmup_run = 2
    N_run = 100
    elapsed_time = []
    for i in range(N_warmup_run):
        preds = model.predict(test_ds, batch_size=mgpu_batch_size)
    for i in range(N_run):
        start_time = time.time()
        preds = model.predict(test_ds, batch_size=mgpu_batch_size)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0:
            print('Step {}: {:5.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
    print('Throughput: {:.0f} images in {:7.1f} sec'.format(N_run * mgpu_batch_size, elapsed_time.sum()))
    print('Throughput: {:.0f} images/s'.format(N_run * mgpu_batch_size / elapsed_time.sum()))
    del model, train_ds, val_ds, test_ds
    _ = gc.collect()
