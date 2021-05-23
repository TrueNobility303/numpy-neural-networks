import scipy.io 
import numpy as np 
from matplotlib import pyplot as plt 

class Module():
    def __init__(self):
        self.x = 0
        self.y = 0

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        pass

    def backward(self,g):
        pass

class Sigmoid(Module):
    def forward(self,x):
        self.x = x 
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self,g):
        return g * self.y * (1-self.y) 

class Normal2one(Module):
    def forward(self,x):
        self.x = x 
        s = x.sum(axis=1).reshape(-1,1)
        self.y = x / s
        return self.y 

    def backward(self,g):
        s = x.sum(axis=1).reshape(-1,1)
        return g * (s-self.x) / np.square(s)

class Linear(Module):
    def __init__(self,in_num,out_num,lr=0.01, momentum=0,penalty=0.1):
        super().__init__()
        self.W = np.random.randn(in_num,out_num) / np.sqrt(in_num + out_num)
        self.b = np.random.randn(1,out_num)
        self.W_last = np.array(self.W) 
        self.b_last = np.array(self.b)

        self.lr = lr
        self.momentum = momentum
        self.penalty = penalty

    def forward(self,x):
        self.x = x 
        self.y = np.matmul(x,self.W) + self.b
        return self.y 

    def backward(self,g):
        dx = np.matmul(g, self.W.T)
        dW = np.matmul(self.x.T, g) 
        db = g

        dW += self.penalty * self.W / np.linalg.norm(self.W,"fro")
        db += self.penalty * self.b / np.linalg.norm(self.b,"fro")

        self.W = self.W  - self.lr * dW + self.momentum * (self.W - self.W_last)
        self.b = self.b  - self.lr * db + self.momentum * (self.b - self.b_last)
        self.W_last = np.array(self.W)
        self.b_last = np.array(self.b)
        return dx

class LinaerWithoutBias(Module):
    def __init__(self,in_num,out_num,lr=0.01, momentum=0,penalty=0.1):
        super().__init__()
        self.W = np.random.randn(in_num,out_num) / np.sqrt(in_num + out_num)
        self.W_last = np.array(self.W) 

        self.lr = lr
        self.momentum = momentum
        self.penalty = penalty

    def forward(self,x):
        self.x = x 
        self.y = np.matmul(x,self.W) 
        return self.y 

    def backward(self,g):
        dx = np.matmul(g, self.W.T)
        dW = np.matmul(self.x.T, g) 
        dW += self.penalty * self.W / np.linalg.norm(self.W,"fro")

        self.W = self.W  - self.lr * dW + self.momentum * (self.W - self.W_last)
        self.W_last = np.array(self.W)
        return dx

class Relu(Module):
    def forward(self,x):
        self.x = x
        z = np.zeros(x.shape)
        self.y = np.stack([x,z],axis=0).max(axis=0)
        return self.y
    def backward(self,g):
        g[self.y<0] = 0
        return g

class LinearSequnce(Module):
    def __init__(self,lst,lr=0.001,momentum=0,penalty=0):
        super().__init__()
        self.lst = lst
        self.layers = []

        in_num = lst[0]
        for i in range(1,len(lst)):
            out_num = lst[i]
            li = Linear(in_num,out_num,lr,momentum,penalty)
            ri = Relu()
            self.layers.append(li)
            self.layers.append(ri)
            in_num = out_num
    
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,g):
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        return g

def padding(x,pad):
    y = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),"constant" )     
    return y

def point_conv(a, W, b):
    z = np.sum(np.sum(a * W) + b)
    return z 

class Conv(Module):
    def __init__(self,Cin,Cout,K,P,S,lr=0.01,momentum=0):
        super().__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.P = P 
        self.K = K
        self.S = S
        self.lr = lr
        self.momentum = momentum
        
        self.W = np.random.randn(K,K,Cin,Cout) / np.sqrt(K*K*Cin)
        self.b = np.random.randn(1,1,1,Cout)
        self.W_last = np.array(self.W)
        self.b_last = np.array(self.b)

    def forward(self,x):
        self.x = x
        B, H, W, Cin = x.shape
        K, K, Cin, Cout = self.W.shape
        P,K,S = self.P, self.K, self.S
        Hout =  (H - K + 2*P) // S + 1
        Wout =  (W - K + 2*P) // S + 1
    
        y = np.zeros((B,Hout,Wout,Cout))
        xpad = padding(x,P)
        
        for i in range(B):                                                         
            for h in range(Hout):                           
                for w in range(Wout):                       
                    for c in range(Cout):                   
                        up = h * S         
                        down = up + K      
                        left = w * S        
                        right = left + K     
                        xwindow = np.expand_dims(xpad[i, up:down, left:right, :],0)
                        y[i, h, w, c] = point_conv(xwindow, self.W[:, : , :, c], self.b[0,0,0,c])
        self.y = y
        return self.y

    def backward(self,dy):
        B, H, W, Cin = self.x.shape
        P,K,S = self.P, self.K, self.S
        B,Hout,Wout,Cout = dy.shape
        xpad = padding(self.x, P)
        dx = np.zeros(self.x.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        dxpad = np.zeros(xpad.shape)
        
        for i in range(B):
            for h in range(Hout):
                for w in range(Wout):
                    for c in range(Cout):
                        up = h * S         
                        down = up + K      
                        left = w * S        
                        right = left + K 
                        xwindow = xpad[i, up:down, left:right, :]
                        
                        dxpad[i, up:down, left:right, :] += self.W[:,:,:,c] * dy[i, h, w, c]
                        dW[:,:,:,c] += xwindow * dy[i,h,w,c]
                        #print(dW[:,:,:,c])
                        db[0,0,0,c] += dy[i,h,w,c]

            dx[i,:,:,:] = dxpad[i, P:-P, P:-P, :]
        
        dW = (dW*S*S) / (H*W*Cin)  
        db = (db*S*S) / (H*W*Cin)
        dx = dx / (K*K)
        self.W = self.W  - self.lr * dW + self.momentum * (self.W - self.W_last)
        self.b = self.b  - self.lr * db + self.momentum * (self.b - self.b_last)
        self.W_last = np.array(self.W)
        self.b_last = np.array(self.b)

        return dx

class LossModule:
    def __init__(self):
        super().__init__()
        self.l = 0
        self.x = 0
        self.y = 0
    def forloss(self,x,y):
        pass
    def backloss(self):
        pass

class MSELoss(LossModule):
    def forloss(self,x,y):
        self.x = x
        self.y = y 
        self.l = np.linalg.norm(x-y,"fro")
        return self.l 
    def backloss(self):
        return (self.x - self.y) / self.l

class LogSoftmaxLoss(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forloss(self,x,y):
        self.x, self.y = x,y
        s = np.exp(x)
        s = s / np.sum(s)
        l = np.multiply(-np.log(s),y).sum()
        self.s, self.l = s,l 
        return self.l 

    def backloss(self):
        return self.s - self.y 


class Dropout(Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.mask = 0
        self.p = p

    def fordrop(self,x,traning):
        if traning is True:
            self.mask = (np.random.random(x.shape) < self.p) / self.p 
            self.y = x * self.mask
        else:
            self.y = x
        return self.y 

    def backward(self,g):
        return self.mask * g

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        self.x  = x
        self.y = x.reshape(x.shape[0],-1)
        return self.y 

    def backward(self,g):
        g = g.reshape(self.x.shape)
        return g

if __name__ == "__main__":
    pass 