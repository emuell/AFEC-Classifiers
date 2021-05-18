"""
  Train a Keras ANN model on the entire descriptor data

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, keras, tensorflow 
"""

import os, sys
import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
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

    # enable, to run ANN on statistics or time series only
    # X_train_time_series, X_train_statistics = split_descriptors(X_train)
    # X_test_time_series, X_test_statistics = split_descriptors(X_test)
    # X_train = X_train_statistics
    # X_test = X_test_statistics

    ###### Build model

    print("\n*** Training model %s of %s..." % (i + 1, len(X_train_sets)))

    params = {
        'dense_neurons': 100, 
        'dense_activation': 'relu',
        'batch_size': 256,
        'epochs': 3000,
        'optimizer': 'adadelta'
    }

    x2_input_shape = X_train.shape[1:2]
    x2_input = Input(x2_input_shape)
    x2 = Dense(params['dense_neurons'], activation=params['dense_activation'])(x2_input)
    x2 = Dropout(0.5)(x2)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x2)
    model = Model(inputs = x2_input, outputs = [out])
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

    ###### Train model

    # early stop training
    earlyStopping = EarlyStopping(monitor='val_loss', patience=60)
    # save the best model so far
    model_filepath = os.path.join(args['classifier_root_path'], 'model_ann.hdf5')
    modelCheckpoint = ModelCheckpoint(filepath=model_filepath, 
        monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=params['batch_size'], 
        epochs=params['epochs'], 
        callbacks=[earlyStopping, modelCheckpoint],
        validation_data=(X_test, y_test),
        verbose=2)

    # load best model weights from ModelCheckpoint
    K.clear_session()
    model = load_model(model_filepath)

    # evaluate
    score, acc = model.evaluate(X_test, y_test, batch_size=params['batch_size'])
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
