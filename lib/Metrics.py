import torch

def MAE(y_true, y_pred):
    absolute_error = torch.abs(y_true - y_pred)
    return torch.mean(absolute_error)


def r_squared(y_true, y_pred):
    mean_observed = torch.mean(y_true)
    total_variation = torch.sum((y_true - mean_observed)**2)
    unexplained_variation = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (unexplained_variation / total_variation)
    return r2


def RMSE(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2)
    rmse = torch.sqrt(mse)
    return rmse


def NRMSE(y_true, y_pred):
    rmse_model = RMSE(y_true, y_pred)
    rmse_baseline = RMSE(y_true, torch.full_like(y_true, torch.mean(y_true)))
    nrmse_value = rmse_model / rmse_baseline
    return nrmse_value
