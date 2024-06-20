import numpy as np
import torch
import random
import argparse
from loguru import logger
import torch.optim as optim
from modules.CDGON import get_CDGON
from lib.Process_data import preprocess_data
from lib.Train_test import train_model, test_model_evaluation, test_model_generalization


parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='Gpu to use')
parser.add_argument('--epochs', type=int, default=200, help='Train epochs')
parser.add_argument('--his_length', type=int, default=1, help='Length of history snapshots(initial snapshot) of input, i.e., 1')
parser.add_argument('--train_length', type=float, default=3, help='Length of target snapshots for train')
parser.add_argument('--test_length', type=int, default=1, help='Length of target snapshots for evaluation')
parser.add_argument('--generalization_train_length', type=int, default=4, help='Length of target snapshots for training in generalization')
parser.add_argument('--generalization_test_length', type=int, default=4, help='Length of target snapshots for training in generalization')
parser.add_argument('--time_scale', type=float, default=30, help='Time scale for integration, controlling the interval of intergration in seconds')
parser.add_argument('--method', type=str, default="euler", help='ODE solver, including: rk4,euler')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--embedding_dim', type=int, default=48, help='Embedding dimension')
parser.add_argument('--lamda', type=int, default=100, help='Edge loss weight, controlling the edge prediction weight in loss')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--log', action='store_true', default=True, help='if write log to files')
args = parser.parse_args()


def get_init_day(city):
    if city == 'FL':
        init_day = 2
    elif city == 'GA':
        init_day = 4
    elif city =='SC':
        init_day = 5
    return init_day

def main(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda:'+str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        logger.add(f'./log/CDGON.log')

    options = vars(args)
    if args.log:
        logger.info(options)
    # print(options)

    cities = ['FL','GA','SC']
    
    # Performance evaluation
    for source_city in cities:
        model = get_CDGON(args, device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y = preprocess_data(
        args, city=source_city, data_length=args.train_length + args.test_length, init_day=get_init_day(source_city))
        train_node_future_Y = node_future_Y[:args.train_length]
        train_edge_future_Y = edge_future_Y[:args.train_length]
        best_model_dict_path = train_model(
            args, device, normal_node_flow, normal_edge_flow, node_x0, edge_x0, train_node_future_Y, train_edge_future_Y, model, optimizer, logger, source_city)
        final_pred_node,final_pred_edge = test_model_evaluation(args, device, best_model_dict_path, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, logger, source_city)
        torch.save(node_future_Y, './results/'+source_city+'_self_node_future_Y.pt')
        torch.save(final_pred_node.to('cpu'), './results/'+source_city+'_self_final_pred_node.pt')
        torch.save(edge_future_Y, './results/'+source_city+'_self_edge_future_Y.pt')
        torch.save(final_pred_edge.to('cpu'), './results/'+source_city+'_self_final_pred_edge.pt')

    # Genralization
    for source_city in cities:
        model = get_CDGON(args, device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y = preprocess_data(
        args, city=source_city, data_length=args.generalization_train_length, init_day=get_init_day(source_city))
        best_model_dict_path = train_model(
            args, device, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, optimizer, logger, source_city)
        for target_city in cities:
            if target_city != source_city:
                normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y = preprocess_data(
                args, city=target_city, data_length=args.generalization_test_length, init_day=get_init_day(target_city))
                final_pred_node = test_model_generalization(args, device, best_model_dict_path, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, logger, target_city)
                node_x0 = node_x0.unsqueeze(0).to('cpu')
                node_future_Y = torch.cat([node_x0, node_future_Y.to('cpu')], dim=0)
                final_pred_node = torch.cat([node_x0, final_pred_node.to('cpu')], dim=0)
                torch.save(node_future_Y, './results/'+ source_city +'_generalization_to_'+target_city+'_node_future_Y.pt')
                torch.save(final_pred_node, './results/'+ source_city +'_generalization_to_'+target_city+'_final_pred_node.pt')


if __name__ == '__main__':
    main(args)
