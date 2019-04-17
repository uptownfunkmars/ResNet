import LoadBatches
import ResNet
from keras import optimizers
import math
from keras.callbacks import ModelCheckpoint, TensorBoard

train_images_path = "./CIFAR-10/cifar-10-batches-py/data_batch_1"
val_images_path = "./CIFAR-10/cifar-10-batches-py/data_batch_2"
train_batch_size = 8

epochs = 1000

input_height = 32
input_width = 32

m = ResNet.ResNet(input_height, input_width)
m.compile(loss='categorical_crossentropy',
          optimizer="sgd",
          metrics=['acc'])
sgd = optimizers.SGD(lr=0.01, momentum=0.9)

dict_train = LoadBatches.unpickle(train_images_path)
dict_val = LoadBatches.unpickle(val_images_path)

G = LoadBatches.dataset(train_batch_size, input_height, input_width, dict_train)
G_test = LoadBatches.dataset(train_batch_size, input_height, input_width, dict_val)

checkpoint = ModelCheckpoint(
    filepath="output/ResNet_model.h5",
    monitor='acc',
    mode='auto',
    save_best_only='True')

tensorboard = TensorBoard(log_dir='output/ResNet_model' )

m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(1000. / train_batch_size),
                epochs=epochs, callbacks=[checkpoint, tensorboard],
                verbose=2,
                validation_data=G_test,
                validation_steps=8,
                shuffle=True)