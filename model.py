from keras import Sequential
from keras import callbacks
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data=ImageDataGenerator(rescale=1./255)
validation_data=ImageDataGenerator(rescale=1./255)

train_gen=train_data.flow_from_directory(  #flow_from_directory-method in imagedatagenerator that generates batches of augmented data
    #from images
    '/dataset/train',
    target_size=(48,48),   #resizes input images.
    batch_size=64, #Number of samples per batch.
    color_mode='grayscale', #converts to grayscale
    class_mode='categorical'   #returns labels
)

valid_gen=validation_data.flow_from_directory(
    '/dataset/test',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)


model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])    #Compiles the model with categorical 
#cross-entropy loss, the Adam optimizer, and accuracy.
earlystopping=callbacks.EarlyStopping(
    monitor='val_loss',   #stops training when validation loss stops improving
    mode='min',          # training should stop when the monitored quantity (validation loss)stops decreasing. 
    patience=3,    #after 3 consecutive passes
    restore_best_weights=True    #Restores the best weights of the model when training stops.
)

info=model.fit(         #This method trains the model on data generated batch-by-batch
        train_gen,
    epochs=30,          #model will iterate 30 times
    validation_data=valid_gen,     #The validation generator (valid_gen) is passed to monitor the model's performance on a
     # separate dataset during training. 
    validation_steps=7178//64, # Specifies the number of batches to draw from the validation generator (valid_gen) for each epoch. 
    callbacks=[earlystopping]
)

model.save('model.h5')