import torch


def preprocess_data(args, city, data_length, init_day):
    '''
    data without normalizing
    '''
    # load data
    datapath = f'../data/{city}_mobility_data.pt'
    mobility_data = torch.load(datapath)  # [T,N,N]
    node_num = mobility_data.shape[-1]
    mobility_data.to(torch.float32)

    # get Normal Node Flow
    node_flow = []
    for t in range(24): #Aug 1 - Aug 24
        node_data_t = torch.diag(mobility_data[t]).unsqueeze(1)
        node_flow.append(node_data_t)
    node_flow = torch.stack(node_flow, dim=0).to(torch.float32)
    normal_node_flow = node_flow.mean(dim=0)
    
    # get Normal Edge Flow
    edge_flow = []
    for t in range(24):  # Aug 1 - Aug 24
        edge_flow.append(mobility_data[t].fill_diagonal_(
            0).unsqueeze(-1).reshape(-1, 1))
    edge_flow = torch.stack(edge_flow, dim=0).to(torch.float32)
    normal_edge_flow = edge_flow.mean(dim=0)
                    
    # Initial Node and Edge (X)
    init_index = 30 + init_day
    node_x0 = torch.diag(mobility_data[init_index]).unsqueeze(
        1).to(torch.float32)  # [N,1]
    edge_x0 = mobility_data[init_index].fill_diagonal_(
        0).unsqueeze(-1).reshape(-1, 1).to(torch.float32)  # [N*N,1]
    
    # contruct Y
    node_future_Y = []
    edge_future_Y = []
    for t in range(init_index+1, init_index + data_length +1):
        node_future_Y.append(torch.diag(mobility_data[t]).unsqueeze(1))
        edge_future_Y.append(mobility_data[t].fill_diagonal_(0).unsqueeze(-1).reshape(-1, 1))
    node_future_Y = torch.stack(node_future_Y, dim=0).to(torch.float32)
    edge_future_Y = torch.stack(edge_future_Y, dim=0).to(torch.float32)
    return normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y
    