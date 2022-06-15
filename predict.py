import torch
import time
from model import Graph2GCN,SAGE
from config import arg
from torch_geometric.data import Data
from train import read_csv,eval


class node_recommendation:
    '好友推荐类'
    def __init__(self,user_id,num_rec,model:torch.nn.Module,device) -> None:
        self.model = model
        self.node_idx = user_id
        self.num_rec = num_rec
        self.device = device
        self.NodeVec:torch.Tensor = None
        self.nodes = None
        self.user_neighbours:torch.Tensor = None
        
    def nodeEdge2vec(self,x:torch.Tensor,edge_index:torch.Tensor):
        output_node = self.model(x,edge_index).to(self.device)
        edge_index = torch.cat([edge_index.flip(dims=(0,)),edge_index],dim=1)
        indices = edge_index[0]==self.node_idx
        neighbours = torch.cat([edge_index[1][indices],torch.tensor([self.node_idx],device=self.device)])
        setattr(self,'NodeVec',output_node)
        setattr(self,'user_neighbours',neighbours)

    def recommend(self):
        prob,indices = (self.NodeVec[self.node_idx] * self.NodeVec).sum(-1).sort(descending=True)
        removed_neig_index = (indices.unsqueeze(1) == self.user_neighbours).sum(-1) ==0
        picked_indices = indices[removed_neig_index]
        prob = prob[removed_neig_index]
        return picked_indices[:self.num_rec],torch.sigmoid(prob[:self.num_rec])


# 加载Embedding图节点
node:torch.Tensor = torch.load('emmmm.pt')
node =node[1:10756].clone() #10755

# 训练集已知边与测试集已知边
train_edge = read_csv('SCHOLAT_LinkPrediction/SCHOLAT_LinkPrediction/train.csv')
test_edge = read_csv('SCHOLAT_LinkPrediction/SCHOLAT_LinkPrediction/test.csv')
pos_edge_index = torch.cat([train_edge,test_edge],dim=-1)-1


# 加载模型
net = SAGE().cuda()
net.load_state_dict(torch.load('model_para/10SAGE.pt'))
net.eval()

#模型效果评估
train_data = Data(node,train_edge-1)
test_data = Data(node,test_edge-1)
acc,acc_p,acc_f = eval(test_data,train_data,net,pos_edge_index)
print(f'Accuracy of link_prediction:{acc},positive edge accuracy:{acc_p},negative edge accuracy:{acc_f}')

# 好友推荐功能实现！
start = time.time()
user106 = node_recommendation(106,5,net,device='cuda')
user106.nodeEdge2vec(train_data.x,edge_index=train_data.edge_index)
idxs,prob=user106.recommend()
end = time.time()
print(end-start,idxs,prob)