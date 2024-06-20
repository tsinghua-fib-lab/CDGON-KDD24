import torch

def Euler_Solver(ODEfunc, initial_embedding_t0, time_steps_to_predict):
    time_grid = time_steps_to_predict
    solution = torch.empty(len(time_grid), *initial_embedding_t0.shape,
                           dtype=initial_embedding_t0.dtype, device=initial_embedding_t0.device)
    solution[0] = initial_embedding_t0
    j = 1
    y0 = initial_embedding_t0

    edge_indicators = []
    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
        dt = t1 - t0
        f0,edge_indicator = ODEfunc(t0, y0)

        edge_indicators.append(edge_indicator)
        dy, f0 = dt * f0, f0
        y1 = y0 + dy
        assert (not torch.isnan(f0).any())

        while j < len(time_steps_to_predict) and t1 >= time_steps_to_predict[j]:
            if time_steps_to_predict[j] == t0:
                solution[j] = y0
            elif time_steps_to_predict[j] == t1:
                solution[j] = y1
            else:
                slope = (time_steps_to_predict[j] - t0) / (t1 - t0)
                solution[j] = y0 + slope * (y1 - y0)
            j += 1
        y0 = y1
    assert (not torch.isnan(solution).any())
    edge_indicators = torch.stack(edge_indicators,dim=0)

    return solution[1:],edge_indicators


def rk4_solver(ODEfunc, initial_embedding_t0, time_steps_to_predict):
    time_grid = time_steps_to_predict
    solution = torch.empty(len(time_grid), *initial_embedding_t0.shape,
                           dtype=initial_embedding_t0.dtype, device=initial_embedding_t0.device)
    solution[0] = initial_embedding_t0
    j = 1
    y0 = initial_embedding_t0

    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
        dt = t1 - t0
        f0 = ODEfunc(t0, y0)
        
        k1 = f0
        if k1 is None:
            k1 = ODEfunc(t0, y0)
        k2 = ODEfunc(t0 + dt * 1/3, y0 + dt * k1 * 1/3)
        k3 = ODEfunc(t0 + dt * 1/3, y0 +
                            dt * (k2 - k1 * 1/3))
        k4 = ODEfunc(t1, y0 + dt * (k1 - k2 + k3))
        dy, f0 = (k1 + 3 * (k2 + k3) + k4) * dt * 0.125, f0
        y1 = y0 + dy

        while j < len(time_steps_to_predict) and t1 >= time_steps_to_predict[j]:
            if time_steps_to_predict[j] == t0:
                solution[j] = y0
            elif time_steps_to_predict[j] == t1:
                solution[j] = y1
            else:
                slope = (time_steps_to_predict[j] - t0) / (t1 - t0)
                solution[j] =  y0 + slope * (y1 - y0)
            j += 1
        y0 = y1
    return solution