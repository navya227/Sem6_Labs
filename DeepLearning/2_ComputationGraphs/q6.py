import torch
import math

def forward_pass(x, y, z):
    a = 2 * x
    b = torch.sin(y)
    c = a / b
    d = c * z
    e = torch.log(d + 1)
    f = torch.tanh(e)
    return f, a, b, c, d, e

def analytical_gradient(x, y, z):
    a = 2 * x
    b = math.sin(y)
    c = a / b
    d = c * z
    e = math.log(d + 1)
    f = math.tanh(e)
    df_de = 1 - f**2
    de_dd = 1 / (d + 1)
    dd_dc = z
    dc_db = -a / b**2
    db_dy = math.cos(y)
    return df_de * de_dd * dd_dc * dc_db * db_dy

def verify_gradient():
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(0.5, requires_grad=True)
    z = torch.tensor(2.0, requires_grad=True)

    f, a, b, c, d, e = forward_pass(x, y, z)
    f.backward()

    print("PyTorch gradient w.r.t y:", y.grad.item())
    print("Analytical gradient w.r.t y:", analytical_gradient(x.item(), y.item(), z.item()))
    print(f"a={a.item()}, b={b.item()}, c={c.item()}, d={d.item()}, e={e.item()}, f={f.item()}")

verify_gradient()
