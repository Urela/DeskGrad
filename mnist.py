import numpy as np
from tensor import Tensor
from tqdm import trange

def fetch_mnist():
    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                dat = f.read()
        else:
            with open(fp, "wb") as f:
                dat = requests.get(url).content
                f.write(dat)
        return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test

def layer_init_uniform(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
    return ret.astype(np.float32)

class Model:
    def __init__(self):
        self.fc1 = Tensor(layer_init_uniform(784, 200))
        self.fc2 = Tensor(layer_init_uniform(200, 25))
        self.fc3 = Tensor(layer_init_uniform(25, 10))

    def forward(self,x):
        x = x.matmul(self.fc1).relu()
        x = x.matmul(self.fc2).relu()
        x = x.matmul(self.fc3).relu()#softmax()
        return x

def train(model, X_train, Y_train, optimizer, BS,steps):
    #label = Tensor(np.array(Y_train[0]))
    #ypred = model.forward( Tensor(X_train[0].reshape(-1,784)) )
    #loss = (ypred - label)  #**2
    pass

X_train, Y_train, X_test, Y_test = fetch_mnist()

np.random.seed(1337)
model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.001)
train(model, X_train, Y_train, optimizer, BS=69, steps=1)
#for p in model.parameters(): p.realize()

