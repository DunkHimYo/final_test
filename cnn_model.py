import sys
import sys
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalMaxPooling2D,Lambda,concatenate,Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def mk_model(filepath=None):
    
    FILTER_SIZE=3
    NUM_FILTERS=30
    INPUT_SIZE=19
    MAXPOOL_SIZE=2
    BATCH_SIZE=3
    STEPS_PER_EPOCH=48//BATCH_SIZE
    EPOCHS=1000
    
    model_input=Input(shape=(INPUT_SIZE,INPUT_SIZE,7))
    model_output=Conv2D(NUM_FILTERS, (FILTER_SIZE,FILTER_SIZE),activation='relu')(model_input)
    model_output=MaxPooling2D(pool_size=(MAXPOOL_SIZE,MAXPOOL_SIZE))(model_output)
    model_output=Dropout(0.5)(model_output)
    split1 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=1))(model_output)
    densors = []
    for idx in split1:
        split2 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=2))(idx)
        for idx2 in split2:
                densors.append(Conv2D(NUM_FILTERS, (FILTER_SIZE,FILTER_SIZE),activation='relu')(idx2))
    model_output=concatenate(densors)
    model_output=Dropout(0.5)(model_output)
    model_output=GlobalMaxPooling2D()(model_output)
    model_output=Dense(units=240,activation='relu')(model_output)
    model_output=Dense(units=240,activation='relu',kernel_regularizer=l1(0.001))(model_output)
    model_output=Dense(units=240,activation='relu')(model_output)
    model_output=Dense(units=1,activation='sigmoid')(model_output)
    model = Model(inputs = model_input, outputs = model_output)
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    
    return model


