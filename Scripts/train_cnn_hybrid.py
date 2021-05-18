"""
  Train Keras CNN model on the time series data and statistics with an ANN on separate branches

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, keras, tensorflow 
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Concatenate 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from sklearn.metrics import classification_report, confusion_matrix

from utils import parse_args, load_dataset, split_data_sets, split_descriptors, \
  plot_losses, plot_confusion_matrix
  
# -------------------------------------------------------------------------------------------------
# parse args

args = parse_args()

# -------------------------------------------------------------------------------------------------
# load dataset

df_labels, df_ids, df_data, num_classes, feature_names = load_dataset(
    os.path.join(args['classifier_root_path'], 'afec-ll.csv'))

# -------------------------------------------------------------------------------------------------
# randomize and split datasets

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

    # plt.imshow(X_train_time_series[99])
    # plt.show()
    
    ###### Build model

    print("\n*** Training model %s of %s..." % (i + 1, len(X_train_sets)))

    ### left branch: pretrain CNN on time series
    x1_input_shape = X_train_time_series.shape[1:3]
    x1_input = Input(x1_input_shape)
    x1 = Reshape(x1_input_shape + (1,))(x1_input)
    # Convolution Layer 
    x1 = Conv2D(50, kernel_size=(5,5), padding='same')(x1)
    x1 = Conv2D(25, kernel_size=(1,1), padding='valid', activation="relu")(x1)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x1 = Conv2D(50, kernel_size=(5,5), padding='same')(x1)
    x1 = Conv2D(25, kernel_size=(1,1), padding='valid', activation="relu")(x1)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x1 = Dropout(0.5)(x1)
    # Fully Connected Layer
    x1 = Flatten()(x1)
    x1 = Dense(units=80, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    # Output
    out = Dense(num_classes, activation='softmax')(x1)
    pretrained_cnn_model = Model(inputs = x1_input, outputs = [out])
    pretrained_cnn_model.compile(loss='categorical_crossentropy', 
        optimizer='adadelta', metrics=['accuracy'])

    # fit CNN model    
    cnn_batch_size = 150
    cnn_epochs = 500
    
    history = pretrained_cnn_model.fit(
        X_train_time_series, y_train, 
        batch_size=cnn_batch_size, 
        epochs=cnn_epochs, 
        validation_data=(X_test_time_series, y_test),
        verbose=2)

    # remove last Dense model from the CNN
    pretrained_cnn_model.layers.pop()

    ### create right branch: ANN on statistics
    x2_input_shape = X_train_statistics.shape[1:3]
    x2_input = Input(x2_input_shape)
    x2 = Dense(300, activation='relu')(x2_input)
    x2 = Dropout(0.5)(x2)

    ### create final model: merge trained CNN and ANN
    x12 = Concatenate()([pretrained_cnn_model.output, x2])
    out = Dense(num_classes, activation='softmax')(x12)
    model = Model(inputs = [x1_input, x2_input], outputs = [out])
    model.compile(loss='categorical_crossentropy', 
        optimizer='rmsprop', metrics=['accuracy'])

    # dump model (layer) summary
    # model.summary()

    ###### Train final model

    batch_size = 64
    epochs = 120
    
    # save the best model so far
    model_filepath = os.path.join(args['classifier_root_path'], 'model_cnn_split.hdf5')
    modelCheckpoint = ModelCheckpoint(filepath=model_filepath, 
        monitor='val_loss', save_best_only=True)

    # train
    history = model.fit(
        [X_train_time_series, X_train_statistics], y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[modelCheckpoint],
        validation_data=([X_test_time_series, X_test_statistics], y_test),
        verbose=2)

    # load best model weights from ModelCheckpoint
    # NB: see http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    K.clear_session()
    model = load_model(model_filepath)

    ###### Evaluate

    score, acc = model.evaluate([X_test_time_series, X_test_statistics], 
        y_test, batch_size=batch_size)
    print('Test score: %s, accuracy %s' % (score, acc))
    validation_accuracies.append(acc)

    # dump classification report
    # class_pred = model.predict_classes(X_test)
    # print('Classification Report:')
    # print(classification_report(np.argmax(y_test, axis=1), class_pred))

    # plot losses
    # plot_losses(history.history['loss'], history.history['val_loss'])
    
    # plot confusion matrix 
    # cm = confusion_matrix(np.argmax(y_test, axis=1), class_pred)
    # plot_confusion_matrix(cm)

# -------------------------------------------------------------------------------------------------
# dump final results

# convert to percentages
validation_accuracies = np.array(validation_accuracies) * 100

print("\nAll Results: %s" % validation_accuracies)
print("-> Mean Accuracy: %.2f%% +- %.2f%%" % 
	(np.mean(validation_accuracies), np.std(validation_accuracies)))
