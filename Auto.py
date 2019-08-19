from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
import warnings
from keras import backend as K
import numpy
#  from multiprocessing import Pool
#  from Metrics import *


class Autoencoder(Sequential):
    def __init__(self,hid, layers=None, name=None):
        super(Autoencoder, self).__init__()
        self.layers = []  # Stack of layers.
        self.model = None  # Internal Model instance.
        self.inputs = []  # List of input tensors
        self.outputs = []  # List of length 1: the output tensor (unique).
        self._trainable = True
        self._initial_weights = None
        self.hid = hid
        # Model attributes.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False
        self.pretrain = False

        # Set model name.
        if not name:
            prefix = 'sequential_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0,
            activation='sigmoid', loss='mean_squared_error', classloss='categorical_crossentropy',
            metrics=['mse', 'acc'], pre_epoch=10, droupout=0, lr=0.1, decay=1e-6, momentum=0.2, nesterov=True,
            lastLoss = 'softmax', **kwargs):
        """Trains the model for a fixed number of epochs.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of epochs to train the model.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: list of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: float (0. < x < 1).
                Fraction of the data to use as held-out validation data.
            validation_data: tuple (x_val, y_val) or tuple
                (x_val, y_val, val_sample_weights) to be used as held-out
                validation data. Will override validation_split.
            shuffle: boolean or str (for 'batch').
                Whether to shuffle the samples at each epoch.
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: Numpy array of weights for
                the training samples, used for scaling the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: if the model was never compiled.
        """
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` '
                          'has been renamed `epochs`.')
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        if self.model is None:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        hid=self.hid
        sgd = RMSprop(lr=lr)
        if not self.pretrain:
            x_input=x
            decoder_layers = []
            autoencoder=Sequential()

            for i in range(1, len(self.layers)):
                temp = Sequential()
                temp.add(Dense(hid[i], activation=activation, input_shape=(hid[i - 1],)))
               # temp.add(normalization.BatchNormalization())
                temp.add(Dropout(droupout))
                temp.add(Dense(hid[i - 1], activation=activation, input_shape=(hid[i],)))
                temp.compile(loss=loss, optimizer=sgd, metrics=metrics)
                temp.fit(x_input, x_input, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
                decoder_layers.append(temp.layers[-1])
                autoencoder.add(temp.layers[0])
                #func = K.function([autoencoder.model.input], [autoencoder.model.layers[-1].get_output_at(0)])
                temp2=Sequential()
                # print(x_input.shape)
                temp2.add(Dense(hid[i], activation=activation, input_shape=(hid[i - 1],)))
                temp2.set_weights(temp.layers[0].get_weights())
                x_input=temp2.predict(x_input)
                del temp
                del temp2
            decoder_layers.reverse()
            print('after layer by layer pretrain')
            for i in range(0, len(self.layers)-1):
                 autoencoder.add(decoder_layers[i])
                # autoencoder.layers[i].set_weights(decoder_layers[i].get_weights())
            autoencoder.compile(loss=loss, optimizer=sgd, metrics=metrics)
            autoencoder.fit(x, x, batch_size=batch_size, epochs=pre_epoch, shuffle=True,verbose=0)
            temp = Sequential()
            for i in range(0, len(self.layers)-1):
                temp.add(autoencoder.layers[i])
                temp.layers[i].set_weights(autoencoder.layers[i].get_weights())
            if len(y):
                temp.add(Dense(y.shape[1], activation=lastLoss, input_shape=(hid[-1],)))
                temp.compile(loss=classloss, optimizer=sgd, metrics=metrics)
                temp.fit(x, y, batch_size=batch_size, epochs=pre_epoch, shuffle=True,verbose=0)
                score = temp.evaluate(x, y, batch_size=20, verbose=1)
                print(score)
                print('After supervised Training')
            for i in range(0,len(self.layers)):
                self.layers[i].set_weights(temp.layers[i].get_weights())
            self.layers=temp.layers
            self.pretrain=True
            del temp
            del autoencoder
            del decoder_layers

        # return self.model.fit(x, y,
        #                       batch_size=batch_size,
        #                       epochs=epochs,
        #                       verbose=verbose,
        #                       callbacks=callbacks,
        #                       validation_split=validation_split,
        #                       validation_data=validation_data,
        #                       shuffle=shuffle,
        #                       class_weight=class_weight,
        #                       sample_weight=sample_weight,
        #                       initial_epoch=initial_epoch)


class MDL_CW(Sequential):

    def __init__(self, Auto1, Auto2, layers=None, name=None):
        super(MDL_CW, self).__init__()
        self.layers = []  # Stack of layers.
        self.model = None  # Internal Model instance.
        self.inputs = []  # List of input tensors
        self.outputs = []  # List of length 1: the output tensor (unique).
        self._trainable = True
        self._initial_weights = None
        # Model attributes.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False
        self.pretrain = False
        self.Auto1 = Auto1
        self.Auto2 = Auto2
        self.Unsupervised_train = False
        # Set model name.
        if not name:
            prefix = 'sequential_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

    def cross(self, x1, x2, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
              validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
              initial_epoch=0, activation='sigmoid', loss='mean_squared_error', classloss='categorical_crossentropy',
              metrics=['mse', 'acc'], pre_epoch=10, droupout=0, lr=0.1, decay=1e-6, momentum=0.2, nesterov=True,
              unsupervised_train=False, supervised_train=False, lastLoss='softmax', **kwargs):
        Auto1 = self.Auto1
        Auto2 = self.Auto2
        hid1 = Auto1.hid
        hid2 = Auto2.hid
        inp1 = Auto1.model.input
        outputs1 = [layer.get_output_at(0) for layer in Auto1.model.layers]  # all layer outputs
        functors1 = [K.function([inp1], [out]) for out in outputs1]
        layer_outs1 = [func([x1]) for func in functors1]

        inp2 = Auto2.model.get_input_at(0)
        outputs2 = [layer.get_output_at(0) for layer in Auto2.model.layers]  # all layer outputs
        functors2 = [K.function([inp2], [out]) for out in outputs2]
        layer_outs2 = [func([x2]) for func in functors2]

        sgd = RMSprop(lr=lr)
        if len(Auto1.layers) != len(Auto2.layers):
            raise TypeError('cross currently is for same-length Autoencoders')
        x1_in = x1
        x2_in = x2
        for i in range(1, len(Auto1.layers)):
            temp1 = Sequential()
            temp2 = Sequential()
            self.add(Dense(hid1[i]+hid2[i], activation=activation, input_shape=(hid1[i - 1]+hid2[i-1],)))
            temp1.add(Dense(hid1[i], activation=activation, input_shape=(hid2[i - 1],)))
            temp2.add(Dense(hid2[i], activation=activation, input_shape=(hid1[i - 1],)))
            temp1.compile(loss=classloss, optimizer=sgd, metrics=metrics)
            temp1.fit(x2_in, layer_outs1[i], batch_size=batch_size, epochs=pre_epoch, shuffle=True, verbose=0)
            score = temp1.evaluate(x2_in,layer_outs1[i], batch_size=20, verbose=1)
            print(score)
            print('cross 2->1')
            temp2.compile(loss=classloss, optimizer=sgd, metrics=metrics)
            temp2.fit(x1_in, layer_outs2[i], batch_size=batch_size, epochs=pre_epoch, shuffle=True, verbose=0)
            score = temp2.evaluate(x1_in,layer_outs2[i], batch_size=20, verbose=1)
            print(score)
            print('cross 1->2')
            #  print(Auto1.layers[0].get_weights())
            w1 = numpy.concatenate((Auto1.model.layers[i].get_weights()[0], temp1.model.layers[1].get_weights()[0]), axis=0)
            b1 = Auto1.model.layers[i].get_weights()[1] + temp1.model.layers[1].get_weights()[1]
            w2 = numpy.concatenate((temp2.model.layers[1].get_weights()[0], Auto2.model.layers[i].get_weights()[0]), axis=0)
            b2 = temp2.model.layers[1].get_weights()[1] + Auto2.model.layers[i].get_weights()[1]
            w0 = numpy.hstack((w1, w2))/2
            b0 = numpy.hstack((b1, b2))/2

            self.layers[i-1].set_weights([w0, b0])
            x1_in = layer_outs1[i]
            x2_in = layer_outs2[i]
            del temp1
            del temp2

        print('after cross')
        set1 = numpy.concatenate((x1, x2), axis=1)
        set2 = numpy.concatenate((2*x1, numpy.zeros(x2.shape)), axis=1)
        set3 = numpy.concatenate((numpy.zeros(x1.shape), 2*x2), axis=1)
        dataset = numpy.concatenate((set1, set2, set3), axis=0)
        supervised_set = numpy.concatenate((y, y, y), axis=0)

        if unsupervised_train:
            un_set = numpy.concatenate((layer_outs1[-2][0], layer_outs2[-2][0]), axis=1)
            unsupervised_set = numpy.concatenate((un_set, un_set, un_set), axis=0)
            self.compile(loss=loss, optimizer=sgd, metrics=metrics)
            self.fit(dataset, unsupervised_set,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     shuffle=shuffle,
                     class_weight=class_weight,
                     sample_weight=sample_weight,
                     initial_epoch=initial_epoch)

        if supervised_train:
            self.add(Dense(y.shape[1], activation=lastLoss, input_shape=(hid1[-1]+hid2[-1],)))
            self.compile(loss=classloss, optimizer=sgd, metrics=metrics)
            self.fit(dataset, supervised_set,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     shuffle=shuffle,
                     class_weight=class_weight,
                     sample_weight=sample_weight,
                     initial_epoch=initial_epoch)
