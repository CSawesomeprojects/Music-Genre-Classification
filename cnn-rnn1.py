import numpy as np
import keras
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU, Dot
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda, Multiply,TimeDistributed
from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras import regularizers

#import numpy

#import matplotlib.pyplot as plt

dict_genres = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 
               'Instrumental':4,'International':5, 'Pop' :6, 'Rock': 7  }


reverse_map = {v: k for k, v in dict_genres.items()}
#print(reverse_map)
print("mona")
npzfile = np.load('shuffled_train.npz')
print("mona")
print(npzfile.files)
X_train = npzfile['arr_0']
y_train = npzfile['arr_1']
#print(X_train.shape, y_train.shape)
npzfile = np.load('shuffled_valid.npz')
#print(npzfile.files)
X_valid = npzfile['arr_0']
y_valid = npzfile['arr_1']
#print(X_valid.shape, y_valid.shape)
#batch_size = 64
num_classes = 8
n_features = X_train.shape[2]
n_time = X_train.shape[1]
nb_filters1=16 
nb_filters2=32 
nb_filters3=64
nb_filters4=64
nb_filters5=64
ksize = (3,1)
pool_size_1= (2,2) 
pool_size_2= (4,4)
pool_size_3 = (4,2)

dropout_prob = 0.20
dense_size1 = 128
lstm_count = 128
num_units = 120

BATCH_SIZE = 32
EPOCH_COUNT = 50
L2_regularization = 0.001

def conv_recurrent_model_build(model_input):
    print('Building model...')
    layer = model_input
    
    ### Convolutional blocks
    conv_1 = Conv2D(filters = nb_filters1, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_1')(layer)
    pool_1 = MaxPooling2D(pool_size_1)(conv_1)

    conv_2 = Conv2D(filters = nb_filters2, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPooling2D(pool_size_1)(conv_2)

    conv_3 = Conv2D(filters = nb_filters3, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_3')(pool_2)
    pool_3 = MaxPooling2D(pool_size_1)(conv_3)
    
    
    conv_4 = Conv2D(filters = nb_filters4, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_4')(pool_3)
    pool_4 = MaxPooling2D(pool_size_2)(conv_4)
    
    
    conv_5 = Conv2D(filters = nb_filters5, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_5')(pool_4)
    pool_5 = MaxPooling2D(pool_size_2)(conv_5)

    flatten1 = Flatten()(pool_5)
    attention_prob = Activation('softmax')(flatten1)
    attention_prob = RepeatVector(256)(attention_prob)
    attention_prob = Permute([2, 1])(attention_prob)
    #attrention_prob = Dense(256, activation = 'softmax', name='attention-probabilities')(flatten1)
    ### Recurrent Block
    
    # Pooling layer
    pool_lstm1 = MaxPooling2D(pool_size_3, name = 'pool_lstm')(layer)
    
    # Embedding layer

    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)
#     flatten2 = K.squeeze(pool_lstm1, axis = -1)
#     dense1 = Dense(dense_size1)(flatten)
    
    # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  #default merge mode is concat
    
    # Concat Output
    #concat = concatenate([flatten1, lstm], axis=-1, name ='concat')
    sent_representation=Multiply()([attention_prob,lstm])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2))(sent_representation)
    ## Softmax Output
    output = Dense(num_classes, activation = 'softmax', name='preds')(sent_representation)
    #output = Dense(num_classes, activation = 'softmax', name='preds')(concat)
   
    model_output = output
    model = Model(model_input, model_output)
    
#     opt = Adam(lr=0.001)
    opt = RMSprop(lr=0.0005)  # Optimizer
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
    
    print(model.summary())
    return model
def train_model(x_train, y_train, x_val, y_val):
    
    n_frequency = 128
    n_frames = 640
    #reshape and expand dims for conv2d
#     x_train = x_train.reshape(-1, n_frequency, n_frames)
    x_train = np.expand_dims(x_train, axis = -1)
    
#     x_val = x_val.reshape(-1, n_frequency, n_frames)
    x_val = np.expand_dims(x_val, axis = -1)
    
    
    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')
    
    model = conv_recurrent_model_build(model_input)
    
#     tb_callback = TensorBoard(log_dir='./logs/4', histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
#                               write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                               embeddings_metadata=None)
    checkpoint_callback = ModelCheckpoint('new_withoutcuda_weights.best.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=50,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)
    #numpy.save('history.npy', history)
    return model, history

model, history  = train_model(X_train, y_train, X_valid, y_valid)

#from sklearn.metrics import classification_report

y_true = np.argmax(y_valid, axis = 1)
X_valid = np.expand_dims(X_valid, axis = -1)
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
labels = [0,1,2,3,4,5,6,7]
target_names = dict_genres.keys()

print(y_true.shape, y_pred.shape)
#print(classification_report(y_true, y_pred, target_names=target_names))

#from sklearn.metrics import accuracy_score

#print(accuracy_score(y_true, y_pred))
