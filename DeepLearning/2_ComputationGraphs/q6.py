import torch


def forward_pass(x, y):
    a = 2 * x
    b = torch.sin(y)
    c = a / b
    d = c + 1
    e = torch.log(d)
    f = torch.tanh(e)
    return f, a, b, c, d, e


def verify_gradient():
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(0.5, requires_grad=True)

    f, a, b, c, d, e = forward_pass(x, y)
    f.backward()

    print("Gradient with respect to y:", y.grad)
    print("Intermediate values:")
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")


verify_gradient()
