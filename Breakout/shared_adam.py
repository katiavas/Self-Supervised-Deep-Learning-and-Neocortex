import torch as T


# optimising our parameters
# params = default parameters


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas,
                                         eps=eps, weight_decay=weight_decay)
        # self.defaults['seed_base'] = seed
        # Iterate over our parameter groups
        for group in self.param_groups:
            # iterate over our parameters in each group
            for p in group['params']:
                state = self.state[p]
                # self.state.set_seed()
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()














