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
train_from_scratch = False

"""
Auxiliary variables, classes and functions
"""
input_layer_size = 16 if model == 'delayed' else 4

if model == 'single':
    dataname = f'a{antenna}'
elif model == 'early':
    dataname = 'f'
elif model == 'delayed':
    dataname = f'delayed'


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


def extract_subset(dataset, activities_true, activities_false):
    data_subset = []
    
    for index in range(len(dataset)):
        activity_index = int(dataset[index][1])
        if activity_index in activities_true:
            data_subset.append((dataset[index][0], 1))
        elif activity_index in activities_false:
            data_subset.append((dataset[index][0], 0))

    return data_subset


def extract_entire_dataset(data_raw):
    dataset = []

    target_activities = [
        Activity.WALK.value, Activity.RUN.value, Activity.JUMP.value, Activity.WAVE.value,
        Activity.CLAP.value, Activity.WIPE.value, Activity.SQUAT.value
    ]

    for index in range(len(data_raw[0])):
        activity_index = int(data_raw[1][index])
        if activity_index in target_activities:
            tmp = tf.convert_to_tensor(data_raw[0][index], dtype=tf.float32)
            dataset.append((tmp, activity_index))
    
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
    output = tf.keras.layers.Dense(2, activation='softmax')(x1)

    model = tf.keras.Model(inputs, output)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['Accuracy'])
    
    return model


"""
MAIN
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

# Load VAE data and apply standard scaler
scaler = StandardScaler()
scaler.fit(data_raw[0])
data_raw[0] = scaler.transform(data_raw[0])

# Separate training and testing set
data_train, data_test = extract_entire_dataset(data_raw)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

print("Done.")

"""
Define the MLPs
"""
### MLP 1: move legs
feature1_list = [Activity.WALK.value, Activity.RUN.value, Activity.JUMP.value, Activity.SQUAT.value]
not_feature1_list = [Activity.WAVE.value, Activity.CLAP.value, Activity.WIPE.value]
model1 = create_mlp([8, 8])

### MLP 2: move legs a lot
feature2_list = [Activity.WALK.value, Activity.RUN.value]
not_feature2_list = [Activity.JUMP.value, Activity.SQUAT.value]
model2 = create_mlp([8, 8])

### MLP 3: move arms a lot
feature3_list = [Activity.RUN.value]
not_feature3_list = [Activity.WALK.value]
model3 = create_mlp([8, 8])

### MLP 4: move lower legs
feature4_list = [Activity.SQUAT.value]
not_feature4_list = [Activity.JUMP.value]
model4 = create_mlp([8, 8])

### MLP 5: move arms
feature5_list = [Activity.WAVE.value]
not_feature5_list = [Activity.CLAP.value, Activity.WIPE.value]
model5 = create_mlp([8, 8])

### MLP 6: move forearms
feature6_list = [Activity.CLAP.value]
not_feature6_list = [Activity.WIPE.value]
model6 = create_mlp([8, 8])

"""
Train the MLPs
"""
if train_from_scratch:
    data_model1 = extract_subset(data_train, feature1_list, not_feature1_list)
    csi_train, label_train = create_mlp_dataset(data_model1)
    model1.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model1.save_weights(f'model_weights/mlp1_s1a_{dataname}.weights.h5')

    data_model2 = extract_subset(data_train, feature2_list, not_feature2_list)
    csi_train, label_train = create_mlp_dataset(data_model2)
    model2.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model2.save_weights(f'model_weights/mlp2_s1a_{dataname}.weights.h5')

    data_model3 = extract_subset(data_train, feature3_list, not_feature3_list)
    csi_train, label_train = create_mlp_dataset(data_model3)
    model3.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model3.save_weights(f'model_weights/mlp3_s1a_{dataname}.weights.h5')

    data_model4 = extract_subset(data_train, feature4_list, not_feature4_list)
    csi_train, label_train = create_mlp_dataset(data_model4)
    model4.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model4.save_weights(f'model_weights/mlp4_s1a_{dataname}.weights.h5')

    data_model5 = extract_subset(data_train, feature5_list, not_feature5_list)
    csi_train, label_train = create_mlp_dataset(data_model5)
    model5.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model5.save_weights(f'model_weights/mlp5_s1a_{dataname}.weights.h5')

    data_model6 = extract_subset(data_train, feature6_list, not_feature6_list)
    csi_train, label_train = create_mlp_dataset(data_model6)
    model6.fit(csi_train, label_train, epochs=10, shuffle=True, callbacks=[early_stopping_cb])
    model6.save_weights(f'model_weights/mlp6_s1a_{dataname}.weights.h5')


"""
Test single MLPs
"""
### MLP 1: move legs
mlp_data_test = extract_subset(data_test, feature1_list, not_feature1_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model1.load_weights(f'model_weights/mlp1_s1a_{dataname}.weights.h5')
print('Model 1 accuracy:')
model1.evaluate(csi, labels)

### MLP 2: move legs a lot
mlp_data_test = extract_subset(data_test, feature2_list, not_feature2_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model2.load_weights(f'model_weights/mlp2_s1a_{dataname}.weights.h5')
print('Model 2 accuracy:')
model2.evaluate(csi, labels)

### MLP 3: move arms a lot
mlp_data_test = extract_subset(data_test, feature3_list, not_feature3_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model3.load_weights(f'model_weights/mlp3_s1a_{dataname}.weights.h5')
print('Model 3 accuracy:')
model3.evaluate(csi, labels)

### MLP 4: move lower legs
mlp_data_test = extract_subset(data_test, feature4_list, not_feature4_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model4.load_weights(f'model_weights/mlp4_s1a_{dataname}.weights.h5')
print('Model 4 accuracy:')
model4.evaluate(csi, labels)

### MLP 5: move arms
mlp_data_test = extract_subset(data_test, feature5_list, not_feature5_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model5.load_weights(f'model_weights/mlp5_s1a_{dataname}.weights.h5')
print('Model 5 accuracy:')
model5.evaluate(csi, labels)

### MLP 6: move forearms
mlp_data_test = extract_subset(data_test, feature6_list, not_feature6_list)
csi, labels = create_mlp_dataset(mlp_data_test)
model6.load_weights(f'model_weights/mlp6_s1a_{dataname}.weights.h5')
print('Model 6 accuracy:')
model6.evaluate(csi, labels)

"""
Test the final rule-based classifier
"""
# Process data using every MLP
csi, label = create_mlp_dataset(data_test)
move_legs      = model1.predict(csi, verbose = False)
move_legs_alot = model2.predict(csi, verbose = False)
move_arms_alot = model3.predict(csi, verbose = False)
move_lowerlegs = model4.predict(csi, verbose = False)
move_arms      = model5.predict(csi, verbose = False)
move_forearms  = model6.predict(csi, verbose = False)

num_samples = len(label)
move_legs      = np.array([np.argmax(move_legs[k])      for k in range(num_samples)])
move_legs_alot = np.array([np.argmax(move_legs_alot[k]) for k in range(num_samples)])
move_arms_alot = np.array([np.argmax(move_arms_alot[k]) for k in range(num_samples)])
move_lowerlegs = np.array([np.argmax(move_lowerlegs[k]) for k in range(num_samples)])
move_arms      = np.array([np.argmax(move_arms[k])      for k in range(num_samples)])
move_forearms  = np.array([np.argmax(move_forearms[k])  for k in range(num_samples)])

# Rule-based classifier
predictions = np.zeros(num_samples)
for k in range(num_samples):
    if move_legs[k]:
        if move_legs_alot[k]:
            if move_arms_alot[k]:
                predictions[k] = Activity.RUN.value
            else:
                predictions[k] = Activity.WALK.value
        else:
            if move_lowerlegs[k]:
                predictions[k] = Activity.SQUAT.value
            else:
                predictions[k] = Activity.JUMP.value
    else:
        if move_arms[k]:
            predictions[k] = Activity.WAVE.value
        else:
            if move_forearms[k]:
                predictions[k] = Activity.CLAP.value
            else:
                predictions[k] = Activity.WIPE.value

# Compute overall accuracy
num_correct = (predictions == label).sum()
print(f'Overall accuracy: {num_correct / num_samples}')
