import torch
import torch.autograd as autograd


def manifold_cons(pred, target):
    """
    L_cons = || Î(u; Θ) - I_obs(u) ||_2^2
    """
    return torch.mean((pred - target) ** 2)


def manifold_reg(M_pred, coords, alpha=10.0):
    """
    L_prior = mean( exp(-α * ||∇_s M̂||_2) * |∂M̂/∂t| )    (Eq. 21)

    """
    grad_outputs = torch.ones_like(M_pred)
    grads = autograd.grad(
        outputs=M_pred,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]                                       

    spatial_grad = grads[..., :3]             
    temporal_grad = grads[..., 3]             

    spatial_norm = torch.sqrt((spatial_grad ** 2).sum(dim=-1) + 1e-8)  
    temporal_abs = torch.abs(temporal_grad)                             

    weight = torch.exp(-alpha * spatial_norm)                           
    return torch.mean(weight * temporal_abs)