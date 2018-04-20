from core.layer import Layer


class System(Layer):
    layers = []

    def __init__(self, params):
        self.params = params

        # fill first layer
        self.layers[0] = Layer(self.params)
        self.layers[0].layer_num = 0   
        self.layers[0].state = None
        self.layers[0].dens_matrix = None
        self.layers[0].channels_num = None
        self.layers[0].channels_mapping = None

    def process(self):
        for i in self.params.layers_num:
            self.layers[i+1] = Layer(self.params)

            pass
        pass
    pass
