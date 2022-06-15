import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from model import Graph2GCN,SAGE
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam,lr_scheduler,SGD
from config import arg
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

def read_csv(fp)->torch.Tensor:
    with open(fp,'r') as f:
        edge = f.readlines()
        for _ in range(len(edge)):
            edge[_] = edge[_][:-1]
            edge[_]=[int(i) for i in edge[_].split(',')]
    return torch.tensor(edge).T.contiguous()

def label_get(pos_index:torch.Tensor,neg_index:torch.Tensor):
    pos_length = pos_index.shape[1]
    label = torch.zeros(pos_length+neg_index.shape[1],device=pos_index.device)
    label[:pos_length]=1
    return label.float()

def train(data:Data,model:torch.nn.Module):
    '训练样本所有正例和随机负采样负例做为训练标签'
    #print(data.train_pos_edge_index.shape)
    #time.sleep(10)
    train_neg_edge_index = negative_sampling(data.train_pos_edge_index,data.num_nodes)
    edge_index = torch.cat([data.train_pos_edge_index,train_neg_edge_index],dim=-1).cuda()
    output:torch.Tensor =model(data.x.cuda(),data.train_pos_edge_index.cuda())
    prob = (output[edge_index[0]] * output[edge_index[1]]).sum(dim=-1)# n
    label = label_get(data.train_pos_edge_index,train_neg_edge_index).cuda()
    loss = binary_cross_entropy(torch.sigmoid(prob),label)
    optim.zero_grad()
    loss.backward()
    optim.step()
    lr_s.step()
    return loss.item()


def rand_mask_train(data:Data,model:torch.nn.Module):
    # 随机mask
    known_pos_edge_index,pos_edge_index_label = data.edge_index[:,torch.randperm(data.edge_index.shape[1])].split(data.edge_index.shape[1]//3*2,dim=1)
    # 负采样
    neg_edge_index_label = negative_sampling(pos_edge_index,data.num_nodes,pos_edge_index_label.shape[1])
    edge_index =  torch.cat([pos_edge_index_label,neg_edge_index_label],dim=-1)
    output:torch.Tensor =model(data.x.cuda(),known_pos_edge_index.cuda())
    prob = (output[edge_index[0]]*output[edge_index[1]]).sum(-1)
    label = label_get(pos_edge_index_label,neg_edge_index_label).cuda()
    loss = binary_cross_entropy(torch.sigmoid(prob),label)
    optim.zero_grad()
    loss.backward()
    optim.step()
    lr_s.step()
    return loss.item()


def eval(data:Data,data2:Data,model:torch.nn.Module,pos_edge:torch.Tensor):

    known_pos_edge_index = data2.edge_index
    pos_edge_index_label = data.edge_index
    val_neg_edge_index = negative_sampling(pos_edge,data.num_nodes,pos_edge_index_label.shape[1])
    edge_index = torch.cat([pos_edge_index_label,val_neg_edge_index],dim=-1).cuda()
    output:torch.Tensor =model(data.x.cuda(),known_pos_edge_index.cuda())
    label = label_get(pos_edge_index_label,val_neg_edge_index).cuda()
    prob = (output[edge_index[0]] * output[edge_index[1]]).sum(dim=-1)
    result =  (prob>=0.5).long()
    acc = accuracy_score(label.cpu().numpy(),result.cpu().numpy())
    acc_pos_edge =((prob[:pos_edge_index_label.shape[1]]>=0.5)*1.0).mean()
    acc_neg_edge = ((prob[pos_edge_index_label.shape[1]:]<0.5)*1.0).mean()
    return acc,acc_pos_edge.item(),acc_neg_edge.item()


if __name__ == '__main__':
    write = SummaryWriter('exp_data/SAGE')
    node:torch.Tensor = torch.load('emmmm.pt')
    node =node[1:10756].clone() #10755
    train_edge = read_csv('SCHOLAT_LinkPrediction/SCHOLAT_LinkPrediction/train.csv')
    test_edge = read_csv('SCHOLAT_LinkPrediction/SCHOLAT_LinkPrediction/test.csv')

    pos_edge_index = torch.cat([train_edge,test_edge],dim=-1)-1

    net = SAGE().cuda()
    net.load_state_dict(torch.load('model_para/10SAGE.pt'))
    optim = Adam(net.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
    lr_s = lr_scheduler.MultiStepLR(optim,[3000,6000,8000])
    train_data = Data(node,train_edge-1)
    test_data = Data(node,test_edge-1)
    print(train_data.edge_index.shape[1],test_data.edge_index.shape[1])


    #print(splited_data.val_pos_edge_index.shape)


    bar = tqdm(range(arg.epoch))

    for epoch in bar:
        net.train()
        loss= rand_mask_train(train_data,net)
        write.add_scalar('loss',loss,epoch)
        net.eval()
        acc,acc_p,acc_f = eval(test_data,train_data,net,pos_edge=pos_edge_index)
        bar.set_description(f'loss:{loss:.6f}.In eval,acc:{acc:.6f},acc_p:{acc_p:.6f},acc_f:{acc_f:.6f}')
        write.add_scalar('acc',acc,epoch)
        write.add_scalar('positive edge accuracy',acc_p,epoch)
        write.add_scalar('negative edge accuracy',acc_f,epoch)
        torch.save(net.state_dict(),'model_para'+arg.model_para_name)