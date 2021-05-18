"""
  Train a Keras tempral convolution network model on the time series descriptor data only

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, keras, tensorflow 
"""

import os
import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Lambda, Conv1D, Convolution1D, SpatialDropout1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.layers

from keras import backend as K
from keras import optimizers

from sklearn.metrics import classification_report, confusion_matrix

from utils import parse_args, load_dataset, split_data_sets, split_descriptors, \
  plot_losses, plot_confusion_matrix

# -------------------------------------------------------------------------------------------------

def residual_block(x, s, i, activation, nb_filters, kernel_size):

    def channel_normalization(x):
        # Normalize by the highest activation
        max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
        out = x / max_values
        return out

    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='causal',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        tanh_out = Activation('tanh')(conv)
        sigm_out = Activation('sigmoid')(conv)
        x = keras.layers.multiply([tanh_out, sigm_out])
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(0.05)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x

# -------------------------------------------------------------------------------------------------
# parse args

args = parse_args()
args['number_of_runs'] = 1

# -------------------------------------------------------------------------------------------------
# load dataset

df_labels, df_ids, df_data, num_classes, feature_names = load_dataset(
    os.path.join(args['classifier_root_path'], 'afec-ll.csv'))

# -------------------------------------------------------------------------------------------------
# split (and randomize) datasets

X_train_sets, X_test_sets, y_test_sets, y_train_sets = split_data_sets(
    df_labels, df_ids, df_data, num_classes, args['number_of_runs'])

# -------------------------------------------------------------------------------------------------
# train

validation_accuracies = []
for i in range(len(X_train_sets)):
    X_train = X_train_sets[i]
    X_test = X_test_sets[i]
    y_train = y_train_sets[i]
    y_test = y_test_sets[i]

    ###### Prepare data

    # split data into 2d time series data and 1d statistics
    X_train_time_series, X_train_statistics = split_descriptors(X_train)
    X_test_time_series, X_test_statistics = split_descriptors(X_test)

    # run TCN on time series only
    X_train = X_train_time_series
    X_test = X_test_time_series

    ###### Build model

    print("\n*** Training model %s of %s..." % (i + 1, len(X_train_sets)))

    nb_filters=32
    kernel_size=5
    dilatations=[1, 2, 4, 8]
    nb_stacks=1
    activation='relu' # 'norm_relu', 'wavenet' or some other regular activation
    use_skip_connections=True
    
    x1_input = Input(X_train.shape[1:3])
    x1 = Convolution1D(nb_filters, kernel_size, padding='causal')(x1_input)
    skip_connections = []
    for s in range(nb_stacks):
        for i in dilatations:
            x1, skip_out = residual_block(x1, s, i, activation, nb_filters, kernel_size)
            skip_connections.append(skip_out)
    if use_skip_connections:
        x1 = keras.layers.add(skip_connections)
    x1 = Activation('relu')(x1)
    x1 = Lambda(lambda tt: tt[:, -1, :])(x1) # first 0 or last -1.
    x1 = Dropout(0.2)(x1)
    out = Dense(num_classes, activation='softmax')(x1)
    model = Model(x1_input, out)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # dump model (layer) summary
    # model.summary()

    ###### Train model

    batch_size = 64
    epochs = 1000
    
    # save the best model so far
    model_filepath = os.path.join(args['classifier_root_path'], 'model_tcn.hdf5')
    modelCheckpoint = ModelCheckpoint(filepath=model_filepath, 
        monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[modelCheckpoint],
        validation_data=(X_test, y_test),
        verbose=2)

    # load best model weights from ModelCheckpoint
    K.clear_session()
    model = load_model(model_filepath)

    # evaluate
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('-> Test score: %s, accuracy %s' % (score, acc))
    validation_accuracies.append(acc)

    # dump classification report
    # class_pred = model.predict_classes(X_test)
    # print('Classification Report:')
    # print(classification_report(np.argmax(y_test, axis=1), class_pred))

    # plot confusion matrix 
    # cm = confusion_matrix(np.argmax(y_test, axis=1), class_pred)
    # print(cm)
    # plot_confusion_matrix(cm)

# -------------------------------------------------------------------------------------------------
# dump final results

# convert to percentages
validation_accuracies = np.array(validation_accuracies) * 100

print("\nAll Results: %s" % validation_accuracies)
print("-> Mean Accuracy: %.2f%% +- %.2f%%" % 
	(np.mean(validation_accuracies), np.std(validation_accuracies)))
