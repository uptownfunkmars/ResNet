from keras.layers import Input
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, Dense


def ResNet(input_height, input_width):

    #this is input
    inputs = Input(shape=(input_height, input_width, 3))

    #Block 1
    x = Conv2D(64, (3, 3),  #原为filters为7 * 7但考虑使用cifar只有32 * 32 所以使用3 * 3
           strides = (2, 2),
           padding = 'valid',
           name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut0 = MaxPooling2D()(x)

    #ReBlock1_Conv1
    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock1_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut1 = x + shortcut0

    #ResBlock2_Conv2

    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut2 = x + shortcut1

    #ResBlock3

    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3),
           strides = (2, 2),
           padding = 'same',
           name = 'ResBlock3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut3 = x + shortcut2

    #ResBlock4

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock4_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock4_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut3 = Conv2D(128, (1, 1),
                   strides=(1, 1),
                   padding='same',
                   name='ResBlock4_Conv3')(x)

    x, shortcut4 = x + shortcut3

    #ResBlock5

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock5_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock5_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut5 = x + shortcut4

    #ResBlock6

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock6_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock6_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut6 = x + shortcut5

    #ResBlock7

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock7_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock7_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut7 = x + shortcut6

    #ResBlock8

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock8_Conv1')(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock8_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut7 = Conv2D(256, (1, 1),
                   strides=(1, 1),
                   padding='same',
                   name='ResBlock_Conv3')(x)

    x, shortcut8 = x + shortcut7

    #ResBlock9

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock9_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock9_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut9 = x + shortcut8

    #ResBlock10

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock10_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock10_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut10 = x + shortcut9

    #ResBlock11

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock11_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock11_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut11 = x + shortcut10

    #ResBlock12

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock12_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock12_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut12 = x + shortcut11

    #ResBlock13

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock13_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock13_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut13 = x + shortcut12

    #ResBlock14

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock14_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock14_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut13 = Conv2D(512,(1, 1),
                    strides=(1, 1),
                    padding="same",
                    name='ResBlock14_Conv3')(x)

    x, shortcut14 = x + shortcut13

    #ResBlock15

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock15_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock15_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut15 = x + shortcut14

    #ReBlock16

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding='same',
           name='ResBlock15_Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3),
           strides=(2, 2),
           padding="same",
           name='ResBlock15_Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x, shortcut16 = x + shortcut15

    #avg pool

    x = AveragePooling2D()(x)

    #fc
    x = Flatten()(x)
    x = Dense(10)(x)
    y = Activation("softmax")(x)

    model = Model(inputs, y)
    model.summary()

    return model











