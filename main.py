import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorboard
%load_ext tensorboard

x_tra,x_test,y_tra,y_test=train_test_split(x,y,train_size=0.8,stratify=y,random_state=128)

kf=KFold(5,True )
train_score=[]
test_score=[]
val_score=[]
idx=0

val_list=[]
i=0
logdir=[]

for train_index, test_index in kf.split(x_tra):
    
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callback_list = [
        EarlyStopping( #성능 향상이 멈추면 훈련을 중지
        monitor='val_loss',  #모델 검증 정확도를 모니터링
        patience=150        #1 에포크 보다 더 길게(즉, 2에포크 동안 정확도가 향상되지 않으면 훈련 중지
        ),
        ModelCheckpoint( #에포크마다 현재 가중치를 저장
        filepath=f'./bp{idx}.h5', #모델 파일 경로
        monitor='val_loss',  # val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않음.
        save_best_only=True,
        mode='auto',
        verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ]
    x_train,x_val=x_tra[train_index],x_tra[test_index]
    y_train,y_val=y_tra[train_index],y_tra[test_index]
    
    model=mk_model()
    print(model.summary())
    model.save(f'./mod{idx}.h5')
    
    hist=model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val),batch_size=1,callbacks=callback_list)
    plt.plot(hist.history['loss'],label='train'+str(idx))
    plt.plot(hist.history['val_loss'],label='train'+str(idx))
    plt.title('loss',fontsize=15)
    plt.legend(['train','val'])
    plt.show()
    plt.plot(hist.history['accuracy'],label='train'+str(idx))
    plt.plot(hist.history['val_accuracy'],label='train'+str(idx))
    plt.legend(['train','val'])
    plt.title('acc',fontsize=15)
    plt.show()
    train_score.append(model.evaluate(x_train,y_train))
    test_score.append(model.evaluate(x_test,y_test))
    val_score.append(model.evaluate(x_val,y_val))
    idx+=1
    i+=0

