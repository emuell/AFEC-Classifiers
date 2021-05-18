"""
  Train Keras LSTM model on the time series descriptor data only (eases tuning LSTM split models)

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, keras, tensorflow
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, CuDNNLSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from sklearn.metrics import classification_report, confusion_matrix

from utils import parse_args, load_dataset, split_data_sets, split_descriptors, \
  plot_losses, plot_confusion_matrix
  
# -------------------------------------------------------------------------------------------------
# parse args

args = parse_args()
args['number_of_runs'] = 1

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
    
    # run LSTM on time_series data only
    X_train = X_train_time_series
    X_test = X_test_time_series

    # plt.imshow(X_train_time_series[99])
    # plt.show()
    
    ###### Build model

    print("\n*** Training model %s of %s..." % (i + 1, len(X_train_sets)))

    x1_input_shape = X_train.shape[1:3]
    x1_input = Input(x1_input_shape)
    x1 = Bidirectional(CuDNNLSTM(80, return_sequences=True))(x1_input)
    x1 = Bidirectional(CuDNNLSTM(80))(x1)
    x1 = Dropout(0.5)(x1)
    out = Dense(num_classes, activation='softmax')(x1)
    model = Model(inputs = x1_input, outputs = [out])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # dump model (layer) summary
    # model.summary()
    
    ###### Train model

    batch_size = 32
    epochs = 200

    # early stop training
    earlyStopping = EarlyStopping(monitor='val_loss', patience=80)
    # save the best model so far
    model_filepath = os.path.join(args['classifier_root_path'], 'model_lstm.hdf5')
    modelCheckpoint = ModelCheckpoint(filepath=model_filepath, 
        monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[earlyStopping, modelCheckpoint],
        validation_data=(X_test, y_test),
        verbose=2)

    # load best model weights from ModelCheckpoint
    # NB: see http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    K.clear_session()
    model = load_model(model_filepath)

    # evaluate
    score, acc = model.evaluate(X_test, y_test)
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

print("\nAll Results: %s" % np.array(validation_accuracies))
print("-> Mean Accuracy: %.2f%% +- %.2f%%" % 
	(np.mean(validation_accuracies), np.std(validation_accuracies)))
