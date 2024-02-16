import random

class ReplayBuffer:
    def __init__(self, buffer_size, replace_probability=0.5):
        self.buffer_size = buffer_size
        self.replace_probability = replace_probability
        self.buffer = []
        # self.times = []

    def query_image(self, image):
        idx = random.randint(0, len(self.buffer) - 1)
        res = self.buffer[idx]
        if type(image) is list:
            self.buffer[idx] = [x.detach().clone() for x in image]
        else:
            self.buffer[idx] = image.detach().clone()
        # self.times[idx] = time.time()
        return res

    def query(self, batch):
        if type(batch) is list:
            if self.buffer_size == 0:
                return batch
            
            batch_clone = [x.clone() for x in batch]
            
            for b in range(batch[0].shape[0]):
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append([x[b] for x in batch_clone])
                    # self.times.append(time.time())
                elif random.uniform(0,1) < self.replace_probability:
                    res = self.query_image([x[b] for x in batch_clone])
                    for i in range(len(batch)):
                        batch_clone[i][b] = res[i]
            return batch_clone
        else:
            if self.buffer_size == 0:
                return batch
    
            batch_clone = batch.clone()
    
            for b in range(batch.shape[0]):
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append(batch_clone[b])
                    # self.times.append(time.time())
                elif random.uniform(0,1) < self.replace_probability:
                    batch_clone[b] = self.query_image(batch_clone[b])
            
            return batch_clone

