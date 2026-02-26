import torch.nn.functional as F


def p_xt_g_x1(X1, E1, t, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    # Debug prints to inspect shapes and category counts
 
  
    # Stelle sicher, dass t_time immer [B,1,1] ist, egal was t enthält
    if t.ndim > 1 and t.size(-1) > 1:
        # z.B. t enthält [num_atoms, logp], dann nur den ersten Wert nehmen
        t_time = t[:, 0].unsqueeze(-1).unsqueeze(-1)
    else:
        # normaler Fall: t ist [B,1]
        t_time = t.squeeze(-1)[:, None, None]
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()
    t_time = t_time.to(device)
    try:
        
        Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    except RuntimeError as e:
        # Prüfe die Fehlermeldung genau
       
        print("⚠️ Detected X1 vs limit_dist mismatch on dim 3!")
        print("DEBUG p_xt_g_x1 REAL SHAPES:")
        print("X1_onehot:", X1_onehot.shape)
        print("limit_dist.X:", limit_dist.X.shape)
        print("t_time:", t_time.shape)
        raise RuntimeError(
            f"RuntimeError in p_xt_g_x1: {e}. Check if X1 has the correct number of categories matching limit_dist.X."
        ) from e
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
        (Et.sum(-1) - 1).abs() < 1e-4
    ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)


def dt_p_xt_g_x1(X1, E1, limit_dist):
    # x1 (B, D)
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    dX = X1_onehot - limit_dist.X[None, None, :]
    dE = E1_onehot - limit_dist.E[None, None, None, :]

    assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

    return dX, dE
