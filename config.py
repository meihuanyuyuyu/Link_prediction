from distutils.command.config import config


class Config1:
    lr = 2e-3
    weight_decay = 1e-4
    epoch =2500
    model_para_name = '/6GCN.pt'

class Config2(Config1):
    num_layers = 20
    weight_decay=1e-5
    model_para_name='/20GCN.pt'

class Config3(Config2):
    'no train_test_split'
    lr = 2e-4
    epoch = 10000
    num_layers=10
    model_para_name='/10SAGE.pt'

arg = Config3