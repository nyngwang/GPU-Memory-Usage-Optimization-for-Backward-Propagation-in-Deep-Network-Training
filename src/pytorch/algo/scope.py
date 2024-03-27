INF = 99999999


class AlgoScope:
    def __init__(
        self,
        specs_name,
        cost_layer,
        cmdargs,
    ):
        self.specs_name = specs_name
        self.specs_length = len(cost_layer)
        self.n = self.specs_length

        self.cmdargs = cmdargs # algorithm might need more info from the cmdargs.

        # used by the algorithm.

        self.cost_layer = [0] + cost_layer.copy()
        self.t_cache = [INF] * (self.specs_length+1)
        self.idx_last_checkpoint = [0] * (self.specs_length+1)

        # constants.
        self.INF = INF


    def reset(self):
        self.t_cache = [self.INF] * (self.specs_length+1)
        self.idx_last_checkpoint = [0] * (self.specs_length+1)

