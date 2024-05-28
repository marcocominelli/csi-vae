import io
import os
import numpy as np
import random
import zipfile
import requests
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from enum import Enum

random.seed(42)

"""
Select which model to run
- model: can be 'single', 'early', 'delayed'
- antenna: if the model is 'single' select which antenna to use (from 0 to 3)
"""
model = 'delayed'
antenna = 0  # works only if model=='single'
mlptype = 'large'

train_from_scratch = False

"""
Auxiliary classes and variables
"""
if mlptype == 'large':
    hidden_layers_size = [22, 22]
elif mlptype == 'small':
    hidden_layers_size = [8, 8]
else:
    print('ERROR: MLP type must be "large" or "small"')
    exit()

input_layer_size = 16 if model == 'delayed' else 4

class Activity(Enum):
    WALK = 0
    RUN = 1
    JUMP = 2
    SIT = 3
    EMPTY = 4
    STAND = 5
    WAVE = 6
    CLAP = 7
    LAY_DOWN = 8
    WIPE = 9
    SQUAT = 10
    STRETCH = 11


class ActivityOutput(Enum):
    WALK = 0
    RUN = 1
    JUMP = 2
    WAVE = 3
    CLAP = 4
    WIPE = 5
    SQUAT = 6


feature_list = [
    Activity.WALK.value,
    Activity.RUN.value,
    Activity.JUMP.value,
    Activity.SQUAT.value,
    Activity.WAVE.value,
    Activity.CLAP.value,
    Activity.WIPE.value
]


def extract_entire_dataset(data_raw):
    dataset = []

    target_activities = [
        Activity.WALK.value, Activity.RUN.value, Activity.JUMP.value, Activity.WAVE.value,
        Activity.CLAP.value, Activity.WIPE.value, Activity.SQUAT.value
    ]

    for index in range(len(data_raw[0])):
        activity_index = int(data_raw[1][index])
        tmp = tf.convert_to_tensor(data_raw[0][index], dtype=tf.float32)
        if activity_index == Activity.WALK.value:
            dataset.append((tmp, ActivityOutput.WALK.value))
        elif activity_index == Activity.RUN.value:
            dataset.append((tmp, ActivityOutput.RUN.value))
        elif activity_index == Activity.JUMP.value:
            dataset.append((tmp, ActivityOutput.JUMP.value))
        elif activity_index == Activity.WAVE.value:
            dataset.append((tmp, ActivityOutput.WAVE.value))
        elif activity_index == Activity.CLAP.value:
            dataset.append((tmp, ActivityOutput.CLAP.value))
        elif activity_index == Activity.WIPE.value:
            dataset.append((tmp, ActivityOutput.WIPE.value))
        elif activity_index == Activity.SQUAT.value:
            dataset.append((tmp, ActivityOutput.SQUAT.value))
    
    # Separate training/testing
    random.shuffle(dataset)
    num_training_samples = int(len(dataset) * 0.8)
    data_train = dataset[:num_training_samples]
    data_test = dataset[num_training_samples:]

    return data_train, data_test


def create_mlp_dataset(data):
    csi = tf.zeros([0, input_layer_size], dtype=tf.float32)
    label = []
    for k in range(len(data)):
        tmp = tf.expand_dims(tf.convert_to_tensor(data[k][0], dtype=tf.float32), axis=0)
        csi = tf.concat([csi, tmp], axis=0)
        label.append(data[k][1])
        
    label = np.asarray(label)

    return csi, label


def create_mlp(shape):
    input_shape=(input_layer_size,)
    inputs = tf.keras.Input(shape=input_shape)
    x1 = tf.keras.layers.Dense(shape[0], activation='relu')(inputs)
    x1 = tf.keras.layers.Dense(shape[1], activation='relu')(x1)
    output = tf.keras.layers.Dense(7, activation='softmax')(x1)

    model = tf.keras.Model(inputs, output)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['Accuracy'])
    
    model.summary()

    return model


if model == 'single':
    model_filename = f'./model_weights/{mlptype}_mlp_s1a_a{antenna}_ls2.weights.h5'
elif model == 'early':
    model_filename = f'./model_weights/{mlptype}_mlp_s1a_f_ls2.weights.h5'
elif model == 'delayed':
    model_filename = f'./model_weights/{mlptype}_mlp_s1a_delayed_ls2.weights.h5'


"""
Main
"""
# Download dataset if not present
if not os.path.exists("dataset") or not os.path.exists("model_weights"):
   print("Downloading the dataset...", end=" ")
   r = requests.get('https://zenodo.org/records/11367112/files/deepprobhar_data.zip')
   z = zipfile.ZipFile(io.BytesIO(r.content))
   z.extractall()
   print("Done.")

print("Preparing the dataset...", end=" ")

if model == 'single':
    data_raw = np.load(f"dataset/s1a_a{antenna}_ls2_12.pkl", allow_pickle=True)
elif model == 'early':
    data_raw = np.load("dataset/s1a_f_ls2_12.pkl", allow_pickle=True)
elif model == 'delayed':
    # Load CSI from all the antennas and concatenate them together
    data_raw0 = np.load(f"dataset/s1a_a0_ls2_12.pkl", allow_pickle=True)
    data_raw1 = np.load(f"dataset/s1a_a1_ls2_12.pkl", allow_pickle=True)
    data_raw2 = np.load(f"dataset/s1a_a2_ls2_12.pkl", allow_pickle=True)
    data_raw3 = np.load(f"dataset/s1a_a3_ls2_12.pkl", allow_pickle=True)

    csi_aux = np.zeros([len(data_raw0[0]), 16])
    label_aux = np.zeros([len(data_raw0[0])])
    for n in range(len(data_raw0[0])):
        tmp = np.concatenate([data_raw0[0][n], data_raw1[0][n], data_raw2[0][n], data_raw3[0][n]])
        csi_aux[n, ...] = tmp
        label_aux[n] = data_raw0[1][n]

    data_raw = [csi_aux, label_aux]
else:
    print("ERROR: Invalid model name (allowed names: 'single', 'early', 'delayed')")
    exit()


# Apply standard scaler
scaler = StandardScaler()
scaler.fit(data_raw[0])
scaler.transform(data_raw[0])
data_raw[0] = scaler.transform(data_raw[0])

# Separate training and testing set
data_train, data_test = extract_entire_dataset(data_raw)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

print("Done.")

# Create the MLP
model1 = create_mlp(hidden_layers_size)

# Train the model
if train_from_scratch:
    csi_train, label_train = create_mlp_dataset(data_train)
    model1.fit(csi_train, label_train, epochs=20, shuffle=True, callbacks=[early_stopping_cb])
    model1.save_weights(model_filename)

# Test the model
model1.load_weights(model_filename)
csi_test, label_test = create_mlp_dataset(data_test)
model1.evaluate(csi_test, label_test)