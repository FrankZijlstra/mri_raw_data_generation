class DataLoader:
    def __init__(self):
        pass
    
    def get_num_channels(self):
        return 0
    
    def get_valid_indices(self):
        return None
    
    def load(self, index):
        return {}, {}
    
    def load_dataset(self, indices):
        data = []
        attr = []
        for index in indices:
            d,a = self.load(index)
            data.append(d)
            attr.append(a)
        return data, attr
