import numpy as np 

class Dataloader:
    def __init__(self, x, y, batch_size, shuffle=True, transform=None):
        if transform is not None:
            self.x, self.y = transform(x,y)
        else:
            self.x, self.y = x,y
        
        self.batch_size = batch_size
        self.size = len(x) // batch_size

        if shuffle is True:
            perm = np.array(range(self.size))
            x = x[perm]
            y = y[perm]

    def __getitem__(self,index):
        B = self.batch_size
        if B*(index+1) >= len(self.x):
            return self.x[B*index:], self.y[B*index:]
        else:
            return self.x[B*index:B*(index+1)], self.y[B*index:B*(index+1)]

    def __len__(self):
        return self.size