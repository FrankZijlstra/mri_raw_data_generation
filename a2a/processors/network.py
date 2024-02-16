from .processor import UnaryProcessor

class Network(UnaryProcessor):
    def __init__(self, network, device='cuda:0'):
        super().__init__()
        self.network = network
        self.device = device
        self.network.to(self.device)

    def call_torch(self, image, allow_in_place=False):
        return self.network(image)
