import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import ray
from ray import tune

EPSILON = np.finfo(np.float32).eps

def get_activation_fn(name):
    ''' Returns a framework specific activation function, given a name string.'''
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "elu":
        return nn.ELU
    raise ValueError("Unknown activation ({})!".format(name))

def get_lr_scheduler(optimizer, name, params):
    lr_scheduler = None
    if name == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'])
    elif name == "cosine_restart":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params['T_0'],
            T_mult=params['T_mult'])
    elif name == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma'])
    return lr_scheduler

def normc_initializer(std=1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))
    return initializer

def const_initializer(val=0.0):
    def initializer(tensor):
        nn.init.constant_(tensor, val)
    return initializer

class DatasetBase(data.Dataset):
    def __init__(self, X, Y, normalize_x=True, normalize_y=True):
        self.X = X
        self.Y = Y

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        
        if normalize_x:
            self.X_mean, self.X_std = np.mean(self.X, axis=0), np.std(self.X, axis=0)
        if normalize_y:
            self.Y_mean, self.Y_std = np.mean(self.Y, axis=0), np.std(self.Y, axis=0)

    def __getitem__(self, index):
        x = self.preprocess_x(self.X[index])
        y = self.preprocess_y(self.Y[index])

        return x, y

    def __len__(self):
        return len(self.X)

    def preprocess_x(self, x, return_tensor=True):
        if self.normalize_x:
            x_new = (x - self.X_mean) / (self.X_std + EPSILON)
        else:
            x_new = x
        if return_tensor:
            x_new = torch.Tensor(x_new)
        return x_new

    def postprocess_x(self, x, return_tensor=True):
        if self.normalize_x:
            x_new = self.X_mean + np.multiply(x, self.X_std)
        else:
            x_new = x
        if return_tensor:
            x_new = torch.Tensor(x_new)
        return x_new

    def preprocess_y(self, y, return_tensor=True):
        if self.normalize_y:
            y_new = torch.Tensor((y - self.Y_mean) / (self.Y_std + EPSILON))
        else:
            y_new = y
        if return_tensor:
            y_new = torch.Tensor(y_new)
        return y_new

    def postprocess_y(self, y, return_tensor=True):
        if self.normalize_y:
            y_new = self.Y_mean + np.multiply(y, self.Y_std)
        else:
            y_new = y
        if return_tensor:
            y_new = torch.Tensor(y_new)
        return y_new

class DatasetClassifier(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(DatasetClassifier, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        x, y = super(DatasetClassifier, self).__getitem__(index)
        y = torch.squeeze(y).long()
        return x, y

class SimpleFC(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 activation_fn=None,
                 initializer_weight=None,
                 initializer_bias=None,
                 ):
        super(SimpleFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer_weight:
            initializer_weight(linear.weight)
        if initializer_bias:
            initializer_bias(linear.bias)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class FCNN(nn.Module):
    ''' 
    A network with fully connected layers.
    '''
    def __init__(
        self, 
        size_in, 
        size_out, 
        hiddens, 
        activations, 
        init_weights,
        init_bias):
        super(FCNN, self).__init__()
        layers = []
        prev_layer_size = size_in
        for i, size_hidden in enumerate(hiddens+[size_out]):
            layers.append(
                SimpleFC(
                    in_size=prev_layer_size,
                    out_size=size_hidden,
                    activation_fn=get_activation_fn(
                        activations[i]),
                    initializer_weight=normc_initializer(
                        init_weights[i]),
                    initializer_bias=const_initializer(
                        init_bias[i]),
                    )
                )
            prev_layer_size = size_hidden
        self._model = nn.Sequential(*layers)
    def forward(self, x):
        return self._model(x)

class Classifier(nn.Module):
    def __init__(self, model_cls, **kwargs):
        super(Classifier, self).__init__()
        self._model = model_cls(**kwargs)
    def forward(self, x):
        return self._model(x)

def get_loss_fn(loss):
    if loss=="MSE":
        return nn.MSELoss()
    elif loss=="MAE" or loss=="L1":
        return nn.L1Loss()
    elif loss=="CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss=="NLLLoss":
        return nn.NLLLoss()
    else:
        raise NotImplementedError


class TrainModel(tune.Trainable):
    def setup(self, config):
        ''' Load datasets for train and test and prepare the loaders'''
        self.model = self.create_model(config)
        self.prepare_data(config)

        ''' Setup torch settings and AE model '''
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0))
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            config.get("lr_schedule", None),
            config.get("lr_schedule_params", None))
        self.loss_fn = get_loss_fn(config.get("loss", "MSE"))
        self.loss_fn_test = get_loss_fn(config.get("loss_test", "MSE"))
        self.iter = 0

    def step(self):
        self.iter += 1
        
        ''' For train dataset '''
        mean_train_loss = 0.0
        self.model.train()
        for data in self.train_loader:
            x, y = data
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.compute_loss(y, x)
            loss.backward()
            self.optimizer.step()
            mean_train_loss += loss.item()
        mean_train_loss /= len(self.train_loader)
        
        ''' For test dataset '''
        mean_test_loss = 0.0
        if self.test_loader:
            with torch.no_grad():
                for data in self.test_loader:
                    x, y = data
                    x = x.to(self.device)
                    loss = self.compute_test_loss(y, x)
                    mean_test_loss += loss.item()
            mean_test_loss /= len(self.test_loader)

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return {"mean_train_loss": mean_train_loss, "mean_test_loss": mean_test_loss}

    def load_dataset(self, file):
        raise NotImplementedError

    def get_data_loader(self, dataset, batch_size, shuffle):
        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        # with FileLock(os.path.expanduser("~/data.lock")):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle)
        return data_loader

    def prepare_data(self, config):
        dataset_train = config.get("dataset_train")
        dataset_test = config.get("dataset_test")
        batch_size = config.get("batch_size")
        shuffle_data = config.get("shuffle_data")

        if isinstance(dataset_train, str):
            dataset_train = self.load_dataset(dataset_train)
        if isinstance(dataset_test, str):
            dataset_test = self.load_dataset(dataset_test)
        
        self.train_loader = self.get_data_loader(
            dataset_train, batch_size, shuffle_data)
        if dataset_test is not None:
            self.test_loader = self.get_data_loader(
                dataset_test, batch_size, shuffle_data)
        else:
            self.test_loader = None

    def create_model(self, config):
        return config.get("model")

    def compute_model(self, x):
        return self.model(x)

    def compute_loss(self, y, x):
        y_recon = self.compute_model(x)
        return self.loss_fn(y_recon, y)

    def compute_test_loss(self, y, x):
        y_recon = self.compute_model(x)
        return self.loss_fn(y_recon, y)

    def save_checkpoint(self, checkpoint_dir):
        print(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
