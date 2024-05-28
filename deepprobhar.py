import io
import os
import torch
import numpy as np
import random
import zipfile
import requests

from deepproblog.dataset import Dataset, DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils.standard_networks import MLP
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau, EpochStop

from problog.logic import Term, Constant
from deepproblog.query import Query
from sklearn.preprocessing import StandardScaler

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
Auxiliary classes and variables
"""
class VAEDataset(Dataset):
    def __init__(self, data_matrix, subset):
        super().__init__()
        self.data = data_matrix
        self.subset = subset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if not isinstance(i, int):
            i = i[0].value
        return self.data[i][0]

    def to_query(self, i):
        _, outcome = self.data[i]
        sub = {Term("a"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("activity", Term("a"), Term(outcome)), sub)
    

def compute_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    correct = 0
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        correct += tp
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)

    total = confusion_matrix.sum()
    accuracy = correct/total

    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {avg_precision}')
    print(f'Recall: {avg_recall}')

    return accuracy, avg_precision, avg_recall


# Define the target activities
activities = [
    'walk',   # A - 0
    'run',    # B - 1
    'jump',   # C - 2
    'sit',    # D - 3, video data NA
    'empty',  # E - 4, video data NA
    'stand',  # F - 5, video data NA
    'wave',   # G - 6
    'clap',   # H - 7
    'laying', # I - 8, video data NA
    'wipe',   # J - 9
    'squat',  # K - 10
    'stretch' # L - 11, video data NA
]

if model == 'single':
    model_filename = f'./model_weights/model_s1a_a{antenna}_ls2.pth'
    results_filename = f'./results_s1a_a{antenna}_ls2.txt'
elif model == 'early':
    model_filename = f'./model_weights/model_s1a_f_ls2.pth'
    results_filename = f'./results_s1a_f_ls2.txt'
elif model == 'delayed':
    model_filename = f'./model_weights/model_s1a_delayed_ls2.pth'
    results_filename = f'./results_s1a_delayed_ls2.txt'


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

# Load dataset
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
data = []
for index in range(len(data_raw[0])):
    activity_index = int(data_raw[1][index])
    if activity_index in [0,1,2,6,7,9,10]:
        data.append((torch.from_numpy(data_raw[0][index]).float(), activities[activity_index]))

# Shuffle dataset for training/testing
random.shuffle(data)

# Create the DeepProbLog dataset
num_training_samples = int(len(data) * 0.8)
train_data = data[:num_training_samples]
test_data = data[num_training_samples:]
train_dataset = VAEDataset(train_data, "train")
test_dataset = VAEDataset(test_data, "test")

print("Done.")

# Create the data loader
batch_size = 20
loader = DataLoader(train_dataset, batch_size)

# Create the six MLPs
lr = 1e-3
input_layer_size = 16 if model == 'delayed' else 4

mlp1 = MLP(input_layer_size, 8, 8, 2)
net1 = Network(mlp1, "net1", batching=True)
net1.optimizer = torch.optim.Adam(mlp1.parameters(), lr=lr)
mlp2 = MLP(input_layer_size, 8, 8, 2)
net2 = Network(mlp2, "net2", batching=True)
net2.optimizer = torch.optim.Adam(mlp2.parameters(), lr=lr)
mlp3 = MLP(input_layer_size, 8, 8, 2)
net3 = Network(mlp3, "net3", batching=True)
net3.optimizer = torch.optim.Adam(mlp3.parameters(), lr=lr)
mlp4 = MLP(input_layer_size, 8, 8, 2)
net4 = Network(mlp4, "net4", batching=True)
net4.optimizer = torch.optim.Adam(mlp4.parameters(), lr=lr)
mlp5 = MLP(input_layer_size, 8, 8, 2)
net5 = Network(mlp5, "net5", batching=True)
net5.optimizer = torch.optim.Adam(mlp5.parameters(), lr=lr)
mlp6 = MLP(input_layer_size, 8, 8, 2)
net6 = Network(mlp6, "net6", batching=True)
net6.optimizer = torch.optim.Adam(mlp6.parameters(), lr=lr)

# Create the DeepProbLog model
model = Model("har.pl", [net1, net2, net3, net4, net5, net6])
model.add_tensor_source("train", train_dataset)
model.add_tensor_source("test", test_dataset)
model.set_engine(ExactEngine(model), cache=True)

# Train the model
if train_from_scratch:
    train = train_model(
        model,
        loader,
        StopOnPlateau("Accuracy", warm_up=10, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(20),
        log_iter = 100,
        profile=0,
    )

    model.save_state(model_filename)


# Test the model
model = Model("har.pl", [net1, net2, net3, net4, net5, net6])
model.add_tensor_source("test", test_dataset)
model.set_engine(ExactEngine(model), cache=True)

model.load_state(model_filename)

# Write results to a text file
with open(results_filename,'w') as f:
    confmat = get_confusion_matrix(model, test_dataset)
    f.write(str(confmat) + 2*'\n')
    f.write(str(confmat.matrix) + 2*'\n')

    accuracy, precision, recall = compute_metrics(confmat.matrix)
    f1score = 2*precision*recall/(precision+recall)

    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1: {f1score}\n')
