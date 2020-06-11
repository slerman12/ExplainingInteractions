def TaylorCAM(x, hessian, batch_dim=0, entities_dim_1=1, entities_dim_2=2, attributes_dim_1=3, attributes_dim_2=4):
    assert x.shape[0] == hessian.shape[batch_dim]
    assert x.shape[1] == hessian.shape[entities_dim_1] == hessian.shape[entities_dim_2]
    assert x.shape[2] == hessian.shape[attributes_dim_1] == hessian.shape[attributes_dim_2]
    hessian = hessian.permute(batch_dim, entities_dim_1, entities_dim_2, attributes_dim_1, attributes_dim_2)
    hess_sum_k = hessian.sum(-1)
    x_i_hess_sum_k = x[:, :, None, :] * hess_sum_k
    ie = x_i_hess_sum_k.sum(-1).pow(2)
    ie += ie.permute(0, 2, 1)
    return ie


def TaylorCAM_basic(x, hessian, batch_dim=0, entities_dim_1=1, entities_dim_2=2, attributes_dim_1=3, attributes_dim_2=4):
    assert x.shape[0] == hessian.shape[batch_dim]
    assert x.shape[1] == hessian.shape[entities_dim_1] == hessian.shape[entities_dim_2]
    assert x.shape[2] == hessian.shape[attributes_dim_1] == hessian.shape[attributes_dim_2]
    hessian = hessian.permute(batch_dim, entities_dim_1, entities_dim_2, attributes_dim_1, attributes_dim_2)
    hess_sum_km = hessian.sum(1).sum(-1)
    # batch, entities, attributes -> batch, entities, entities, attributes
    hess_jl = hess_sum_km.unsqueeze(1).repeat(1, hessian.shape[1], 1, 1)
    x_il = x.unsqueeze(2).repeat(1, 1, hessian.shape[2], 1)
    x_il_hess_jl = x_il[:, :, :, :] * hess_jl
    ie = x_il_hess_jl.sum(-1).pow(2)
    ie += ie.permute(0, 2, 1)
    return ie
