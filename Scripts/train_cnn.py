"""
  Train Keras CNN model on the time series descriptor data only (eases tuning CNN split models)

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, keras, tensorflow 
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Dense, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Activation, GlobalAveragePooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from sklearn.metrics import classification_report, confusion_matrix

from utils import parse_args, load_dataset, split_data_sets, split_descriptors, \
  plot_losses, plot_confusion_matrix

# -------------------------------------------------------------------------------------------------

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    # x *= 255
    # if K.image_data_format() == 'channels_first':
    #     x = x.transpose((1, 2, 0))
    # x = np.clip(x, 0, 255).astype('uint8')
    return x

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

    # run CNN on time_series data only
    X_train = X_train_time_series
    X_test = X_test_time_series

    # plt.imshow(X_train_time_series[99])
    # plt.show()
    
    ###### Build model

    print("\n*** Training model %s of %s..." % (i + 1, len(X_train_sets)))

    """
    ### CNN on time series
    x1_input_shape = X_train.shape[1:3]
    x1_input = Input(x1_input_shape)
    x1 = Reshape(x1_input_shape + (1,))(x1_input)
    # Convolution Layer 
    x1 = Conv2D(64, kernel_size=(5,5), padding='same', name="conv2d_3x3")(x1)
    x1 = Conv2D(16, kernel_size=(1,1), padding='valid', activation="elu")(x1)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Conv2D(256, kernel_size=(5,5), padding='same', name="conv2d_5x5")(x1)
    x1 = Conv2D(32, kernel_size=(1,1), padding='valid', activation="elu")(x1)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
    x1 = Dropout(0.25)(x1)
    # Fully Connected Layer
    x1 = Flatten()(x1)
    x1 = Dense(units=200, activation="elu")(x1)
    x1 = Dropout(0.5)(x1)
    out = Dense(num_classes, activation='softmax')(x1)
    model = Model(inputs = x1_input, outputs = [out])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    """
    ### 1D CNN on time series
    x1_input_shape = X_train.shape[1:3]
    x1_input = Input(x1_input_shape)
    x1 = Conv1D(50, 14, activation='elu', padding='same')(x1_input)
    x1 = Conv1D(50, 14, activation='elu', padding='valid')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(100, 7, activation='elu', padding='same')(x1)
    x1 = Conv1D(100, 7, activation='elu', padding='valid')(x1)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dropout(0.5)(x1)
    out = Dense(num_classes, activation='softmax')(x1)
    model = Model(inputs = x1_input, outputs = [out])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # dump model (layer) summary
    # model.summary()

    ###### Train model

    batch_size = 32
    epochs = 500
    
    # save the best model so far
    model_filepath = os.path.join(args['classifier_root_path'], 'model_cnn.hdf5')
    modelCheckpoint = ModelCheckpoint(filepath=model_filepath, 
        monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[modelCheckpoint],
        validation_data=(X_test, y_test),
        verbose=2)

    """
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv2d_3x3'
    filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
    # this is the placeholder for the input images
    input_img = model.input

    img_width = x1_input_shape[0] 
    img_height = x1_input_shape[1]

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    kept_filters = []
    for filter_index in range(50):
        print('Processing filter %d' % filter_index)

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        input_img_data = np.random.random((1, img_width, img_height, 1))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        print('Filter %d processed' % (filter_index))

    # we will stich the best 16 filters on a 4 x 4 grid.
    n = 4

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

    # save the result to disk
    # save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    plt.imshow(stitched_filters)
    plt.show()
    """

    # load best model weights from ModelCheckpoint
    # NB: see http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    K.clear_session()
    model = load_model(model_filepath)

    ###### Evaluate

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
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

# convert to percentages
validation_accuracies = np.array(validation_accuracies) * 100

print("\nAll Results: %s" % validation_accuracies)
print("-> Mean Accuracy: %.2f%% +- %.2f%%" % 
	(np.mean(validation_accuracies), np.std(validation_accuracies)))
