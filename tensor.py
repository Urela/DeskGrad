# Inspired by https://github.com/avramdj/minigrad/tree/d3bc5b40d4d2b4646c0bf0bff4a81e51891eea9b
import numpy as np
from numbers import Number
from functools import partialmethod
from typing import Union, Iterable, Optional

class Function:
  def __init__(self, *tensor: 'Tensor'):
    self.parents = tensor
    self.saved_tensors = []

  def forward(self, *args, **kwargs):   
    raise NotImplementedError( "forward() is not implementd")

  def backward(self, *args, **kwargs): 
    raise NotImplementedError("backward() is not implementd")
 
### #COM BACK TO 
def unbroadcast(a, shape):
  if shape == (1,):
    return a.sum().reshape((1,))
  axdiff = len(a.shape) - len(shape)
  if axdiff <= 0:
    return a
  return a.sum(axis=tuple(range(axdiff)))

class Add(Function):
  def __init__(self, a, b):
    self.a, self.b = a, b
  def forward(self): 
    return self.a + self.b
  def backward(self):
    return [unbroadcast(grad, self.a.shape), unbroadcast(grad, self.b.shape)]

class MatMul(Function):
  def __init__(self, a, b):
    self.a, self.b = a, b
  def forward(self):
    return np.matmul(self.a , self.b)
 
  def backward(self, grad):
    da = np.matmul(grad, self.b.swapaxes(-2,-1) )
    db = np.matmul(self.a.swapaxes(-2,-1),  grad)
    return [unbroadcast(da, self.a.shape), unbroadcast(db, self.b.shape)]

class ReLU(Function):
  def __init__(self, a): 
    self.a = a
  def forward(self):
    return (self.a >= 0) * self.a

  def backward(self, grad):
    da = (self.a >= 0) * grad
    return [ unbroadcast(da, self.a.shape) ]

class Softmax(Function):
  def __init__(self, a): 
    self.a = a
  def forward(self):
    e_x = np.exp(self.a - np.max(self.a)) # shift values
    return  e_x / np.sum(e_x, axis=1) #[:, np.newaxis]

class Sum(Function):
  def __init__(self, a, axis=None):
    self.a, self.axis = a, axis
  def forward(self):
    return np.sum(self.a , axis=self.axis, keepdims=True)

################
class Tensor:
  def __init__(self, data: Union[Number, list, np.ndarray], requires_grad=True):

    if isinstance(data, Number):
      data = [data]
    if isinstance(data, list):
      data = np.array(data)
    if isinstance(data, np.ndarray):
      self.data = data
    else: raise Exception(f"Can't make Tensor object for {data}")

    self.grad = None
    self.requires_grad = requires_grad
    self._parents  = []
    self._function = None
    #self.ctx = None

  # ------------- backward and Graph ------------- 
  def backward(self):
    visited = set()
    topo_nodes : list[Tensor] = []

    def topo_sort(node: Tensor):
      if node not in visited:
        visited.add(node)
        for p in node._parents:
          topo_sort(p)
        topo_nodes.append(node)
    topo_sort(self)
    for v in reversed(topo_nodes):
      if not v.grad:
        self.grad = v.ones_like(v)
        self.requires_grad = False
      if not v._function: continue

      nabula = v._function.backward(v.grad.data)
      for grad, parent in zip( nabula, v._parents):
          if parent.requires_grad:
            parent.grad = Tensor(parent.grad.data if parent.grad else 0)
    pass

  # ------------- Class Methods ------------- 

  @classmethod
  def ones(cls, shape: Union[int, Iterable[int]], requires_grad=True):
    return cls(np.ones(shape), requires_grad=requires_grad)

  @classmethod
  def zeros(cls, shape: Union[int, Iterable[int]], requires_grad=True):
    return cls(np.zeros(shape),  requires_grad=requires_grad)

  @classmethod
  def ones_like(cls, a, requires_grad=True):
    a = a.data if isinstance(a, Tensor) else a
    return cls(np.ones_like(a, dtype=np.float32), requires_grad)

  # ------------- Types of Operations on Tensor ------------- 

  @staticmethod
  def unary_ops(func:Function.__class__, a: 'Tensor', alpha=None):
    func = func(a.data) if alpha is None else func(a.data, alpha)
    res = Tensor(func.forward(), requires_grad=a.requires_grad )
    res._parents = [a]
    res._function = func
    return res

  @staticmethod
  def binary_ops(func:Function.__class__, a: 'Tensor', b: 'Tensor'):
    if not isinstance(a, Tensor):
      a = Tensor(np.full(b.shape, a, dtype=np.float32), requires_grad=False)
    if not isinstance(b, Tensor):
      b = Tensor(np.full(a.shape, b, dtype=np.float32), requires_grad=False)
    func = func(a.data, b.data)
    res = Tensor(func.forward(), requires_grad=(a.requires_grad or b.requires_grad))
    res._parents = [a,b]
    res._function = func
    return res


  # ------------- Operations on Tensor ------------- 

  def sum(self, axis=None):
    return self.unary_ops(Sum, self, alpha=axis)
  
  def relu(self, axis=None):
    return self.unary_ops(ReLU, self)

  def softmax(self, axis=None):
    return self.unary_ops(Softmax, self)

  def matmul(self, x):
    return self.binary_ops(MatMul, self, x)  

  def __add__(self, x):
    return self.binary_ops(Add, self, x)

  def __str__(self):
    return self.data.__str__()
   
################
if __name__ == '__main__':
  x = Tensor.ones((3,1))
  y = Tensor([[1.0,0,-2.0]])
  z = y.matmul(x).relu()
  z.backward()
  print(x,'\n',y,'\n\n',z)

  #print(x.grad)  # dz/dx
  #print(y.grad)  # dz/dy
