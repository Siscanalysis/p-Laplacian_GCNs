import os
import torch.nn as nn
from datetime import datetime
import torch


def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.now()) + '\n')
        f.write("#epoch \t loss \t acc \t avg_epoch_time \t avg_epoch_cost \n")

    return train_log, test_log, valid_log

def printParOnFile(test_name, log_dir, par_list):

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")

class IdFun(nn.Module):
    def forward(self, input):
        return input

def normalize(h):
    return (h-h.mean(0))/h.std(0)

class SimpleMultiLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_act_fun=torch.nn.ReLU(), out_act_fun=lambda x:x, device=None):
        super(SimpleMultiLayerNN, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act_fun=hidden_act_fun
        self.output_act_fun = out_act_fun

        self.hidden_layers = torch.nn.ModuleList().to(self.device)
        hidden_size_prev_layer=input_size
        for current_layer_size in hidden_size:
            self.hidden_layers.append(torch.nn.Linear(hidden_size_prev_layer,current_layer_size,bias=True))
            hidden_size_prev_layer=current_layer_size
        self.hidden_layers.append(torch.nn.Linear(hidden_size_prev_layer,output_size,bias=True))


    def reset_parameters(self):
        for layer in self.hidden_layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x=self.hidden_act_fun(layer(x))
        return self.output_act_fun(self.hidden_layers[-1](x))