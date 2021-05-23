import numpy as np 
from matplotlib import pyplot as plt 
import scipy.io 

def oneHot(y):
    a = np.zeros((y.shape[0],10))
    for i in range(len(y)):
        a[i,y[i,0]] = 1
    return a

def normalize(x):
    x = x / 255
    x = x - np.mean(x, axis=1).reshape(-1,1)
    x / np.std(x, axis=1).reshape(-1,1)
    return x

def myTransform(x,y):
    x = normalize(x)
    y = oneHot(y)
    return x,y

def biInterpolate(x,i,j):
    rows,cols = x.shape 
    up = int(np.floor(i))
    down = int(np.ceil(i))
    left = int(np.floor(j))
    right = int(np.ceil(j))
    if up<0 or left<0 or down>=rows or right>=cols:
        return 0
    u,v = i-up, j-left 
    y = u*v*x[up,left] + u*(1-v)*x[up,right] + (1-u)*v*x[down,left] + (1-u)*(1-v)*x[down,right]
    return y 

def shift(x,dx,dy):
    y = np.empty(x.shape)
    n,m = x.shape 
    for i in range(n):
        for j in range(m):
            y[i,j] = biInterpolate(x,i+dx,j+dy)
    return y 

def resize(x,NH,NW):
    H,W = x.shape 
    y = np.empty(x.shape)
    ch,cw = H/2, W/2
    for h in range(H):
        for w in range(W):
            preh = ch + H / NH * (h-ch) 
            prew = cw + W / NW * (w-cw) 
            y[h,w] = biInterpolate(x,preh,prew)
    return y 

def rotate(x,angle):
    H,W = x.shape 
    y = np.empty(x.shape)
    ch,cw = H/2, W/2
    #print(cw)
    for h in range(H):
        for w in range(W):
            a = np.pi/2 if w==cw else np.arctan((ch-h)/(w-cw))
            if w < cw:
                a = a + np.pi
            r = np.sqrt((h-ch)**2 + (w-cw)**2)
            na = a + angle
            preh = ch - r * np.sin(na)  
            prew = cw + r * np.cos(na)  
            y[h,w] = biInterpolate(x,preh,prew)
    return y 

if __name__ == "__main__":
    data = scipy.io.loadmat("./digits.mat")
    x = np.array(data["X"])
    x = x[0].reshape(16,16)
    #plt.imshow(x)
    #plt.show()
    y = rotate(x,0.5)
    plt.imshow(y)
    plt.show()


