import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class CDGON_ODE(nn.Module):
    '''
    ODE function
    '''

    def __init__(self, Embedding_dimension, device):
        super(CDGON_ODE, self).__init__()
        self.emb_D = Embedding_dimension
        self.device = device
        self.alpha_n_dedunce = nn.Linear(Embedding_dimension, Embedding_dimension)
        self.gamma_e_dedunce = nn.Linear(Embedding_dimension, Embedding_dimension)

        self.node_to_edge = nn.Linear(Embedding_dimension * 2, Embedding_dimension)
        self.edge_to_value = nn.Sequential(
            nn.Linear(Embedding_dimension, Embedding_dimension//2),
            nn.ReLU(),
            nn.Linear(Embedding_dimension//2, 1))
        self.f_theta = nn.Sequential(
            nn.Linear(Embedding_dimension, Embedding_dimension//2),
            nn.ReLU(),
            nn.Linear(Embedding_dimension//2, 1))
        self.W_3 = nn.Parameter(torch.FloatTensor(
            Embedding_dimension, Embedding_dimension), requires_grad=True)  # [emb,emb]
        self.W_4 = nn.Parameter(torch.FloatTensor(
            Embedding_dimension, Embedding_dimension), requires_grad=True)  # [emb,emb]
        
        self.edge_to_edge = nn.Linear(Embedding_dimension, Embedding_dimension)
        self.node_to_edge_2 = nn.Linear(Embedding_dimension * 2, Embedding_dimension)
        
        init.normal_(self.W_3, mean=0, std=0.1)
        init.normal_(self.W_4, mean=0, std=0.1)

    def forward(self, t, node_edge_embedding):
        '''
        :param node_edge_embedding: [N+N*N, D]
        '''
        node_embedding = node_edge_embedding[:self.node_num]
        edge_embedding = node_edge_embedding[self.node_num:]

        # A computation
        fully_connected = torch.ones([self.node_num, self.node_num])
        start_index = torch.where(fully_connected)[0]
        end_index = torch.where(fully_connected)[1]
        start_node = node_embedding[start_index] 
        end_node = node_embedding[end_index] 
        edges = torch.cat([start_node, end_node], dim=-1) 
        A = F.relu(self.edge_to_value(self.node_to_edge(edges)).reshape(
            self.node_num, self.node_num)) 
        
        # GCN-alpha
        epsilon = 1e-4
        epsilon = torch.tensor(epsilon).to(self.device)
        D = torch.diag(
            1.0 / torch.sqrt(torch.clamp(torch.sum(A, dim=1), min=epsilon)))
        normalized_A = torch.matmul(torch.matmul(D, A), D)
        massage_conv = torch.matmul(torch.matmul(
            normalized_A, node_embedding), self.W_3)
        alpha_n = self.alpha_n_dedunce(massage_conv)
        abs_alpha_n_max = torch.max(
            torch.max(torch.abs(alpha_n)), epsilon) 
        alpha_n = alpha_n / abs_alpha_n_max

        # PIDM
        denominator = self.f_theta(self.node_flow_norm)
        norm_value = self.f_theta(
            node_embedding) / torch.where(denominator == 0, epsilon, denominator)
        node_self_evolve = norm_value * \
            torch.matmul(
                F.relu((self.node_flow_norm - node_embedding)), self.W_4)
        node_self_evolve = alpha_n * node_self_evolve
        
        # NEIM
        gamma_e = F.softmax(self.gamma_e_dedunce(edge_embedding),dim=1)
        edge_indicator = torch.where(A < 1, torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device)).reshape(-1,1)
        Aggr_edge = torch.matmul(self.A_ne, edge_indicator* gamma_e * edge_embedding)

        # Node Computation
        pred_node = node_self_evolve + Aggr_edge
        
        edge_self_evolve = self.edge_to_edge(edge_embedding)
        fully_connected = torch.ones([self.node_num, self.node_num])
        start_index = torch.where(fully_connected)[0]
        end_index = torch.where(fully_connected)[1]
        start_node = node_embedding[start_index] 
        end_node = node_embedding[end_index] 
        edges = torch.cat([start_node, end_node], dim=-1)  
        edge_infonode = self.node_to_edge_2(edges)

        # Edge Computation
        pred_edge = edge_indicator * (edge_self_evolve + edge_infonode)
        pred_node_edge = torch.concat([pred_node,pred_edge],dim=0)
        
        return pred_node_edge,edge_indicator

    def set_region_related_value(self, node_flow_norm, A_ne,node_num):
        self.node_flow_norm = node_flow_norm
        self.A_ne = A_ne
        self.node_num = node_num

