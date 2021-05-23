from clstorch import * 
from transform import myTransform
from dataloader import Dataloader
import matplotlib.pyplot as plt 
import tqdm 

data = scipy.io.loadmat("./digits.mat")
Xtest,ytest = np.array(data["Xtest"]), np.array(data["ytest"]-1)
X,y = np.array(data["X"]), np.array(data["y"]-1)
Xvalid, yvalid = np.array(data["Xvalid"]), np.array(data["yvalid"]-1)
dataLoader = Dataloader(X,y,batch_size=32,transform=myTransform)
testLoader = Dataloader(Xtest,ytest,batch_size=32,transform=myTransform)
validLoader = Dataloader(Xvalid,yvalid,batch_size=32,transform=myTransform)

class Net():
    def __init__(self):
        self.fc = LinearSequnce([256,64,10],lr=0.01,momentum=0.9,penalty=0)

        #self.fc1 = Linear(256,64,lr=0.01,momentum=0.9)
        #self.activate1 = Relu()

        #self.dropout1 = Dropout()
        #self.fc2 = Linear(64,10,lr=0.01,momentum=0.9)
        self.criterion = LogSoftmaxLoss()

    def forward(self,x,traning):
        x = self.fc(x)
        #x = self.activate1(x)
        #x = self.dropout1.fordrop(x,traning)
        #x = self.fc2(x)
        return x

    def backward(self,g):
        #g = self.fc2.backward(g)
        #g = self.dropout1.backward(g)
        #g = self.activate1.backward(g)
        g = self.fc.backward(g)
        return g

    def train(self, dataLoader, validLoader):
        testloss = []
        validloss = []
        for e in range(1000):
            loss = 0
            for data in dataLoader:
                x,y = data
                if len(x) == 0:
                   break
                sample = np.random.randint(len(x))
                x,y = x[sample].reshape(1,-1), y[sample].reshape(1,-1)

                pred = self.forward(x,traning=True)
                loss += self.criterion.forloss(pred,y)

                g = self.criterion.backloss()
                _ = self.backward(g)

            loss /= len(dataLoader)
            if e%100 == 0:
                for layer in self.fc.layers:
                    if(isinstance(layer,Linear)):
                        layer.lr /= 2
                #self.fc.lr /= 2 
                #self.fc2.lr /= 2
                acc = self.test(validLoader)
                print('epoch',e, "loss", loss,'acc',acc)
        
            testloss.append(self.valid(dataLoader))
            validloss.append(self.valid(validLoader))

        plt.plot(testloss)
        plt.plot(validloss)
        plt.legend(['train','valid'])
        plt.savefig('./dump/loss-valid.png')
        
    def test(self,dataLoader):
        correct = 0
        total = 0
        for data in dataLoader:
            x,y = data
            if len(x) == 0:
                break
            prob = self.forward(x,traning=False)
            pred = np.argmax(prob,axis=1)
            label = np.argmax(y,axis=1)
            correct += np.equal(pred,label).sum() 
            total += len(pred)
        accucacy = correct / total
        return accucacy
    
    def valid(self,dataLoader):
        loss = 0
        for data in dataLoader:
            x,y = data
            if len(x) == 0:
                break
            pred = self.forward(x,traning=False)
            loss += self.criterion.forloss(pred,y)
        loss /= len(dataLoader)
        return loss 

model = Net()
model.train(dataLoader, validLoader)
acc = model.test(testLoader)
print('final',acc)

