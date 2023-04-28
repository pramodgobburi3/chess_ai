import tensorflow as tf
from tensorflow.keras import layers

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = layers.Activation('relu')(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = layers.Add()([x, blockInput])
    return x

def upsample_concat_block(x, xskip, filters, size):
    u = layers.UpSampling2D((2,2))(x)
    c = layers.Concatenate()([u, xskip])
    out = layers.Conv2D(filters, size, activation='relu', padding='same')(c)
    return out

# U-Net model
def get_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoding path
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = residual_block(conv1, 32)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = residual_block(conv2, 64)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = residual_block(conv3, 128)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Middle path
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = residual_block(conv4, 256)
    
    # Decoding path
    up5 = upsample_concat_block(conv4, conv3, 128, (3,3))
    up5 = layers.Dropout(0.3)(up5)
    up5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    up5 = residual_block(up5, 128)

    up6 = upsample_concat_block(up5, conv2, 64, (3,3))
    up6 = layers.Dropout(0.3)(up6)
    up6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    up6 = residual_block(up6, 64)

    up7 = upsample_concat_block(up6, conv1, 32, (3,3))
    up7 = layers.Dropout(0.3)(up7)
    up7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    up7 = residual_block(up7, 32)

    outputs = layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(up7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model

# Create the model
unet_model = get_unet(input_shape=(256, 256, 1))
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
