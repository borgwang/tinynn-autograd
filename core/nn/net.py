class Net:
    def forward(self, x):
        raise NotImplementedError

    # TODO: parameters managment for general net
    def get_parameters(self):
        return [layer.params for layer in self.layers]

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def zero_grad(self):
        for param in self.get_parameters():
            for p in param.values():
                if p is not None: p.zero_grad()

class SequentialNet(Net):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
