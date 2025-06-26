#!/usr/bin/env python

import torch

if torch.cuda.is_available():
    print ('GPU is available')
    dev = "cuda:0" 
else:
    print ('GPU is *NOT* available')
    dev = "cpu" 
device = torch.device(dev) 
a = torch.zeros(4,3) 
a = a.to(device)
