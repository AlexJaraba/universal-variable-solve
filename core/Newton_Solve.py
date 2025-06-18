import numpy as np

def Newton_Solver(tol, function, d_function, x0, max_iter=100, verbose=False):
    x = x0
    for i in range(max_iter):
        f = function(x)
        df = d_function(x)
        if df == 0 or not np.isfinite(df):
            if verbose:
                print(f"Iteration {i}: df is non-finite or zero at x = {x}")
            break

        dx = -f / df

        if not np.isfinite(dx):
            if verbose:
                print(f"Iteration {i}: dx is non-finite at x = {x}, f = {f}, df = {df}")
            break
        x += dx
        if abs(dx) < tol:
            return x
    return x