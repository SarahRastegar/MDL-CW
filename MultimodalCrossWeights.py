from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.io as sio
from keras.layers import Dense
from keras.optimizers import RMSprop
from Auto import Autoencoder, MDL_CW
from Metrics import draw
import numpy
from keras import backend as K
from itertools import cycle
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
f = sio.loadmat('pascal.mat')


x_train1 = np.array(f['train_x1'])
x_train2 = np.array(f['train_x2'])
y_train = np.array(f['train_y'])
x_test1 = np.array(f['test_x1'])
x_test2 = np.array(f['test_x2'])
y_test = np.array(f['test_y'])
x_train1 = np.vstack((x_train1, x_test1[0:400, :]))
x_train2 = np.vstack((x_train2, x_test2[0:400, :]))
y_train = np.vstack((y_train, y_test[0:400, :]))
x_test1 = x_test1[400:, :]
x_test2 = x_test2[400:, :]
y_test = y_test[400:, :]
x_test2 = (x_test2-x_train2.min())/(x_train2.max()-x_train2.min())
x_test1 = (x_test1-x_train1.min())/(x_train1.max()-x_train1.min())
x_train2 = (x_train2-x_train2.min())/(x_train2.max()-x_train2.min())
x_train1 = (x_train1-x_train1.min())/(x_train1.max()-x_train1.min())

###########################################################
sgd = RMSprop(lr=0.001)
loss = 'mean_squared_error'
classloss = 'categorical_crossentropy'
activation='relu'
lastLoss = 'softmax'
lr = 0.001
df = 0.1
prt = True
sup = 50
uns = 50
hid1 = (1408, 500, 100)

text = Autoencoder(hid1)
for i in range(1, len(hid1)):
    text.add(Dense(hid1[i], activation=activation, input_shape=(hid1[i-1],)))
text.add(Dense(20, activation=lastLoss, input_shape=(hid1[-1],)))
text.compile(loss=loss, optimizer=sgd, metrics=['mse', 'acc'])
if prt:
    text.fit(x_train1, y_train, batch_size=10, epochs=sup, shuffle=True, verbose=0, activation=activation,
             loss=loss, classloss=classloss, metrics=['mae', 'acc'], lastLoss=lastLoss,
             pre_epoch=uns, droupout=df, lr=lr, decay=1e-6, momentum=0.2, nesterov=True)
#   score = text.evaluate(x_test1, y_test, batch_size=10, verbose=1)
    print('after text')
    text.save_weights('text.h5')
else:
    text.load_weights('text.h5')
######################################################################

hid2 = (260, 500, 100)
image = Autoencoder(hid2)
for i in range(1, len(hid2)):
    image.add(Dense(hid2[i], activation=activation, input_shape=(hid2[i-1],)))
image.add(Dense(20, activation=lastLoss, input_shape=(hid2[-1],)))
image.compile(loss=loss, optimizer=sgd, metrics=['mse', 'acc'])
if prt:
    image.fit(x_train2, y_train, batch_size=10, epochs=sup, shuffle=True, verbose=0, activation=activation,
              loss=loss, classloss=classloss, metrics=['mae', 'acc'], lastLoss=lastLoss,
              pre_epoch=uns, droupout=df, lr=lr, decay=1e-6, momentum=0.2, nesterov=True)
#    score = image.evaluate(x_test2, y_test, batch_size=10, verbose=1)
    print('after image')
    image.save_weights('image.h5')
else:
    image.load_weights('image.h5')
temp = MDL_CW(text, image)
temp.cross(x_train1, x_train2, y_train, unsupervised_train=False, supervised_train=True, verbose=0, batch_size=10,
           epochs=sup, shuffle=True, pre_epoch=uns, activation=activation, loss=loss,lastLoss=lastLoss,
           classloss=classloss, metrics=['mae', 'acc'], droupout=df, lr=lr, decay=1e-6,
           momentum=0.2, nesterov=True)
temp.save_weights('MDL_CW.h5')
################################################################
x1 = numpy.hstack((numpy.zeros(x_test1.shape), x_test2))
x2 = numpy.hstack((x_test1, numpy.zeros(x_test2.shape)))
x_t1 = numpy.hstack((numpy.zeros(x_train1.shape), x_train2))
x_t2 = numpy.hstack((x_train1, numpy.zeros(x_train2.shape)))
x_train = numpy.hstack((x_train1, x_train2))

inp1 = temp.model.input
outputs1 = [layer.get_output_at(0) for layer in temp.model.layers]  # all layer outputs
functors1 = [K.function([inp1], [out]) for out in outputs1]
layer_outs1 = [func([x1]) for func in functors1]
layer_outs2 = [func([x2]) for func in functors1]

layer_outs_train = [func([x_t1]) for func in functors1]
draw(layer_outs1[-1][0], layer_outs_train[-1][0], y_test, y_train, 'image 2 image')
###############################################################################
layer_outs_train = [func([x_train]) for func in functors1]
draw(layer_outs1[-1][0], layer_outs_train[-1][0], y_test, y_train, 'image 2 both')
###############################################################################
layer_outs_train = [func([x_train]) for func in functors1]
draw(layer_outs2[-1][0], layer_outs_train[-1][0], y_test, y_train, 'text 2 text')
############################################################################################
layer_outs_train = [func([x_t2]) for func in functors1]
draw(layer_outs2[-1][0], layer_outs_train[-1][0], y_test, y_train, 'text 2 both')

# y=MDL_CW.predict(temp,numpy.hstack((numpy.zeros(x_test1.shape),x_test2)))
# classes=numpy.argmax(y,axis=1)
# labels=numpy.zeros(y.shape)
# evaluate(y_test, classes,20)
#
# y=MDL_CW.predict(temp,numpy.hstack((x_test1,numpy.zeros(x_test2.shape))))
# classes=numpy.argmax(y,axis=1)
# labels=numpy.zeros(y.shape)
# evaluate(y_test, classes,20)
