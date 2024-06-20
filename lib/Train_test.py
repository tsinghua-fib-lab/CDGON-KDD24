from tqdm import tqdm
from lib.Metrics import MAE, RMSE, r_squared, NRMSE
import torch
import os
import pandas as pd


def train_model(args, device, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, optimizer, logger,city):
    
    best_train_loss = float('inf')
    pred_length = len(node_future_Y)
    logger.info(f"time_steps_to_predict = torch.arange{pred_length}/{args.time_scale}")

    train_results = {
                            'Epoch': [],
                            'Total Loss': [],
                            'Node Loss': [],
                            'Node MAE': [],
                            'Node R2': [],
                            'Node NRMSE': [],
                            'Edge Loss': [],
                            'Edge MAE': [],
                            'Edge R2': [],
                            'Edge NRMSE': []
                        }
    
    model.train()
    early_stop = 0
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        time_steps_to_predict = torch.arange(pred_length + 1)/(args.time_scale)
        normal_edge_flow = normal_edge_flow.to(device)
        normal_node_flow = normal_node_flow.to(device)
        node_x0 = node_x0.to(device)
        edge_x0 = edge_x0.to(device)
        node_future_Y = node_future_Y.to(device)
        edge_future_Y = edge_future_Y.to(device)
        time_steps_to_predict = time_steps_to_predict.to(device)
        final_pred_node, final_pred_edge = model(
            normal_node_flow,normal_edge_flow,node_x0, edge_x0, time_steps_to_predict)
        

        node_pred_loss = RMSE(node_future_Y, final_pred_node)
        edge_pred_loss = RMSE(edge_future_Y, final_pred_edge)
        
        loss = node_pred_loss + args.lamda * edge_pred_loss 
        node_pred_mae = MAE(node_future_Y, final_pred_node)
        edge_pred_mae = MAE(edge_future_Y, final_pred_edge)
        node_pred_r2 = r_squared(node_future_Y, final_pred_node)
        edge_pred_r2 = r_squared(edge_future_Y, final_pred_edge)
        node_pred_nmse = NRMSE(node_future_Y, final_pred_node)
        edge_pred_nmse = NRMSE(edge_future_Y, final_pred_edge)
        loss.backward()
        optimizer.step()

        if loss < best_train_loss:
            best_train_loss = loss
            best_model_dict_path = f'./models/train_in_'+city+'_best_model.pth'
            torch.save(model.state_dict(), best_model_dict_path)
            early_stop = 0
        else:
            early_stop += 1

        train_results['Epoch'].append(epoch)
        train_results['Total Loss'].append(round(loss.item(), 4))
        train_results['Node Loss'].append(round(node_pred_loss.item(), 4))
        train_results['Node MAE'].append(round(node_pred_mae.item(), 4))
        train_results['Node R2'].append(round(node_pred_r2.item(), 4))
        train_results['Node NRMSE'].append(round(node_pred_nmse.item(), 4))

        train_results['Edge Loss'].append(round(edge_pred_loss.item(), 4))
        train_results['Edge MAE'].append(round(edge_pred_mae.item(), 4))
        train_results['Edge R2'].append(round(edge_pred_r2.item(), 4))
        train_results['Edge NRMSE'].append(round(edge_pred_nmse.item(), 4))

    train_df = pd.DataFrame(train_results)
    model_train_result_path = f'./results/train_result_in_'+city+'.csv'
    train_df.to_csv(model_train_result_path, index=False)

    return best_model_dict_path

def test_model_evaluation(args, device, best_model_dict_path, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, logger,city):

    model.load_state_dict(torch.load(best_model_dict_path))
    model.eval()
    pred_length = len(node_future_Y)
    with torch.no_grad():
        time_steps_to_predict = torch.arange(pred_length + 1)/(args.time_scale)
        normal_edge_flow = normal_edge_flow.to(device)
        normal_node_flow = normal_node_flow.to(device)
        node_x0 = node_x0.to(device)
        edge_x0 = edge_x0.to(device)
        node_future_Y = node_future_Y.to(device)
        edge_future_Y = edge_future_Y.to(device)
        time_steps_to_predict = time_steps_to_predict.to(device)
        final_pred_node, final_pred_edge = model(
            normal_node_flow, normal_edge_flow, node_x0, edge_x0, time_steps_to_predict)

        final_pred_node_test = final_pred_node[args.train_length:
                                          args.train_length + args.test_length]
        final_pred_edge_test = final_pred_edge[args.train_length:
                                          args.train_length + args.test_length]
        node_future_Y = node_future_Y[args.train_length:
                                      args.train_length + args.test_length]
        edge_future_Y = edge_future_Y[args.train_length:
                                      args.train_length + args.test_length]

        node_pred_loss = RMSE(node_future_Y, final_pred_node_test)
        edge_pred_loss = RMSE(edge_future_Y, final_pred_edge_test)
        node_pred_mae = MAE(node_future_Y, final_pred_node_test)
        edge_pred_mae = MAE(edge_future_Y, final_pred_edge_test)
        node_pred_r2 = r_squared(node_future_Y, final_pred_node_test)
        edge_pred_r2 = r_squared(edge_future_Y, final_pred_edge_test)
        node_pred_nrmse = NRMSE(node_future_Y, final_pred_node_test)
        edge_pred_nrmse = NRMSE(edge_future_Y, final_pred_edge_test)


        print(f'Temporal Test in {city} Node Loss: {node_pred_loss:.4f}, Test Node MAE: {node_pred_mae:.4f}, Test Node R2: {node_pred_r2:.4f}, Test Node NRMSE: {node_pred_nrmse:.4f}')

        print(f'Temporal Test in {city} Edge Loss: {edge_pred_loss:.4f}, Test Edge MAE: {edge_pred_mae:.4f}, Test Edge R2: {edge_pred_r2:.4f}, Test Edge NRMSE: {edge_pred_nrmse:.4f}')
        
    return final_pred_node,final_pred_edge


def test_model_generalization(args, device, best_model_dict_path, normal_node_flow, normal_edge_flow, node_x0, edge_x0, node_future_Y, edge_future_Y, model, logger, city):

    model.load_state_dict(torch.load(best_model_dict_path))
    model.eval()
    pred_length = len(node_future_Y)
    with torch.no_grad():
        time_steps_to_predict = torch.arange(pred_length + 1)/(args.time_scale)
        normal_edge_flow = normal_edge_flow.to(device)
        normal_node_flow = normal_node_flow.to(device)
        node_x0 = node_x0.to(device)
        edge_x0 = edge_x0.to(device)
        node_future_Y = node_future_Y.to(device)
        edge_future_Y = edge_future_Y.to(device)
        time_steps_to_predict = time_steps_to_predict.to(device)
        final_pred_node, final_pred_edge = model(
            normal_node_flow,normal_edge_flow,node_x0, edge_x0, time_steps_to_predict)
        
        node_pred_loss = RMSE(node_future_Y, final_pred_node)
        edge_pred_loss = RMSE(edge_future_Y, final_pred_edge)
        node_pred_mae = MAE(node_future_Y, final_pred_node)
        edge_pred_mae = MAE(edge_future_Y, final_pred_edge)
        node_pred_r2 = r_squared(node_future_Y, final_pred_node)
        edge_pred_r2 = r_squared(edge_future_Y, final_pred_edge)
        node_pred_nrmse = NRMSE(node_future_Y, final_pred_node)
        edge_pred_nrmse = NRMSE(edge_future_Y, final_pred_edge)

        
        print(f'Transfer Test in {city} Node Loss: {node_pred_loss:.4f}, Test Node MAE: {node_pred_mae:.4f}, Test Node R2: {node_pred_r2:.4f}, Test Node NRMSE: {node_pred_nrmse:.4f}')
        
        print(f'Transfer Test in {city} Edge Loss: {edge_pred_loss:.4f}, Test Edge MAE: {edge_pred_mae:.4f}, Test Edge R2: {edge_pred_r2:.4f}, Test Edge NRMSE: {edge_pred_nrmse:.4f}')

        
    return final_pred_node


