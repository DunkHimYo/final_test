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
    NUM_FILTERS=60
    INPUT_SIZE=19
    MAXPOOL_SIZE=2
    BATCH_SIZE=3
    STEPS_PER_EPOCH=48//BATCH_SIZE
    EPOCHS=1000
    
    model_input=Input(shape=(INPUT_SIZE,INPUT_SIZE,7))
    model_output=Flatten()(model_input)
    model_output=Dense(units=2048,activation='relu')(model_output)
    model_output=Dense(units=2048,activation='relu')(model_output)
    model_output=Dense(units=2048,activation='relu',kernel_regularizer=l1(0.001))(model_output)
    model_output=Dense(units=2048,activation='relu')(model_output)
    model_output=Dropout(0.5)(model_output)
    model_output=Dense(units=2048,activation='relu')(model_output)
    model_output=Dense(units=1024,activation='relu',kernel_regularizer=l1(0.001))(model_output)
    model_output=Dense(units=512,activation='relu')(model_output)
    model_output=Dense(units=1,activation='sigmoid')(model_output)
    model = Model(inputs = model_input, outputs = model_output)
    model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])
    
    return model


