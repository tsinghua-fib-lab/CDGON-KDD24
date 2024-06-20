import torch
import torch.nn as nn
import numpy as np
from utils.utils import init_weights
from lib.ODE_solver import Euler_Solver,rk4_solver
from modules.GraphODE import CDGON_ODE


class CDGON(nn.Module):
    def __init__(self, ODEfunc, method, device, embedding_dim):
        super(CDGON, self).__init__()
        self.ODEfunc = ODEfunc
        self.method = method
        self.device = device
        self.input_size = 1
        self.hidden_size = embedding_dim
        self.output_size = 1

        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.ReLU()
        )
        self.decoder_mlp_node = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.ReLU()
        )
        self.decoder_mlp_edge = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.ReLU()
        )


    def forward(self, normal_node_flow, normal_edge_flow, initial_node_data, initial_edge_data, time_steps_to_predict):
        '''
        :param normal_node_flow: [N, 1]
        :param normal_edge_flow: [N * N, 1]
        :param initial_node_data: [N, 1]
        :param initial_edge_data: [N * N, 1]
        :param time_steps_to_predict: [T]
        :return pred_node: [pred_T, N, 1]
                pred_edge: [pred_T, N*N, 1]
        '''
        
        # Related Parameters
        N = initial_node_data.shape[0]
        A_ne = torch.zeros(N, N * N)
        for node_id in range(N):
            for j in range(N):
                if node_id != j:
                    A_ne[node_id][node_id * N + j] = 1
                    A_ne[node_id][j * N + node_id] = 1
        A_ne = A_ne.to(normal_node_flow.device)

        # Encoder for population mobility graph
        Node_embedding_t0 = self.node_encoder(initial_node_data)
        Node_flow_norm = self.node_encoder(normal_node_flow)
        self.ODEfunc.set_region_related_value(Node_flow_norm, A_ne, N)
        Edge_embedding_t0 = self.edge_encoder(initial_edge_data)
        Node_Edge_embedding = torch.concat(
            [Node_embedding_t0, Edge_embedding_t0], dim=0)  # [N+N*N,emb]

        #  ST Decay Model Informed NeuralODE
        if self.method == 'euler':
            pred_node_edge,edge_indicators = Euler_Solver(
                self.ODEfunc, Node_Edge_embedding, time_steps_to_predict)
        elif self.method == 'rk4':
            pred_node_edge,edge_indicators = rk4_solver(
                self.ODEfunc, Node_Edge_embedding, time_steps_to_predict)


        # Decoder for population mobility graph
        pred_node = pred_node_edge[:, :N, :]
        pred_edge = pred_node_edge[:, N:, :]
        pred_node = self.decoder_mlp_node(pred_node)
        pred_edge = self.decoder_mlp_edge(pred_edge)* edge_indicators

        # Mask and Filter Edges
        T = pred_edge.shape[0]
        pred_edge = pred_edge.view(T,N,N)
        pred_edge = pred_edge * (1-torch.eye(N).unsqueeze(0).expand(T,N,N).to(pred_edge.device))
        pred_edge = pred_edge.view(T,N*N,1)
        pred_edge = torch.where(pred_edge < 1, torch.tensor(0.0).to(pred_edge.device), pred_edge)

        return pred_node, pred_edge


def get_CDGON(args, device):
    '''
    create CDGON
    '''
    # 0. random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # 1. ST Decay Model Informed NeuralODE
    ODEfunc = CDGON_ODE(args.embedding_dim,device)
    ODEfunc.apply(init_weights)

    # 2. CDGON Model
    model = CDGON(ODEfunc, args.method, device, args.embedding_dim).to(device)
    model.apply(init_weights)
    return model
