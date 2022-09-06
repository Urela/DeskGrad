#Inspired by https://github.com/avramdj/minigrad/blob/d3bc5b40d4d2b4646c0bf0bff4a81e51891eea9b/src/minigrad/engine/tensor.py  
import numpy as np
from numbers import Number
from typing import Union, Iterable, Optional

class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    pass
  def forward(self, *args, **kwargs): raise NotImplementedError
  def backward(self, *args, **kwargs): raise NotImplementedError

### #COM BACK TO 
def unbroadcast(a, shape):
  if shape == (1,):
    return a.sum().reshape((1,))
  axdiff = len(a.shape) - len(shape)
  if axdiff <= 0:
    return a
  return a.sum(axis=tuple(range(axdiff)))

# mul by negative one and add to subtract
class Add(Function):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  def forward(self):
    return self.x + self.y
  def backward(self, grad):
    return [unbroadcast(grad, self.x.shape), unbroadcast(grad, self.y.shape)]

# mul by negative one to divide
class Mul(Function):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  def forward(self):
    return self.x * self.y

class MatMul(Function):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  def forward(self):
    return np.matmul(self.y, self.x)

  def backward(self, grad):
    dx = np.matmul(grad, self.y.swapaxes_(-2,-1) )
    dy = np.matmul(self.y.swapaxes_(-2,-1),  grad)
    return [unbroadcast(dx, self.x.shape), unbroadcast(dy, self.y.shape)]

class Pow(Function):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  def forward(self):
    return self.x.pow(self.y)

  def backward(self, grad):
    dx = self.x.pow(self.y - 1.0) * self.y * grad
    return [unbroadcast(dx, self.x.shape)]

class Tensor:
  def __init__(self, data, requires_grad=True):

    if isinstance(data, Number):
      data = [data]
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray): 
      self.data = data
    else: raise Exception(f"Can't create Tensor from {data}")

    self.grad : Optional[Tensor] = None
    self.requires_grad = requires_grad
    self._function = None
    self._parents = []

  def backward(self):
    visited = set()
    topo_nodes : list[Tensor] = []

    # move out of this function
    def build_topo(node: Tensor):
      if node not in visited:
        visited.add(node) 
        for p in node._parents:
          build_topo(p)
        topo_nodes.append(node)

    print(topo_nodes)
    build_topo(self)
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

  ############### Properties
  @property
  def shape(self):
    return self.data.shape

  def __str__(self):
    return self.data.__str__()

  def __repr__(self):
    return f"<Tensor {self.data.shape}>"

  ############### Class methods
  @classmethod
  def zeros(cls, shape: Union[int, Iterable[int]], requires_grad=True):
    return cls(np.zeros(shape), requires_grad=requires_grad)

  @classmethod
  def ones(cls, shape: Union[int, Iterable[int]], requires_grad=True):
    return cls(np.ones(shape), requires_grad=requires_grad)
  
  @classmethod
  def rand(cls, shape: Union[int, Iterable[int]], requires_grad=True):
    return cls(np.random.rand(*shape), requires_grad=requires_grad)

  @classmethod
  def ones_like(cls, a, requires_grad=True):
    a = a.data if isinstance(a, Tensor) else a
    return cls(np.ones_like(a, dtype=np.float32), requires_grad)

  @classmethod
  def zeros_like(cls, a, requires_grad=True):
    a = a.data if isinstance(a, Tensor) else a
    return cls(np.zeros_like(a, dtype=np.float32), requires_grad)

  ############### Type of operations we 
  @staticmethod
  def _binary_ops(A, B, func:Function.__class__):
    print(A,B)
    if isinstance(A, Number):
      A = Tensor(np.full(B.shape, A, dtype=np.float32), requires_grad=True )
    if isinstance(B, Number):
      B = Tensor(np.full(A.shape, B, dtype=np.float32), requires_grad=True )
    func = func(A.data, B.data)
    res = Tensor(func.forward(), requires_grad=(A.requires_grad or B.requires_grad))
    #####
    res._function = func
    res._parents = [A, B]
    return res

  @staticmethod
  def _unary_op(A, func:Function.__class__):
    func = func(A.data)
    res = Tensor(func.forward(), requires_grad=A.requires_grad)
    if not requires_grad:
      return res
    res._function = func
    res._parents = [a]
    return res

  @staticmethod
  def _unary_param_op(A, alpha, func:Function.__class__):
    func = func(A.data, alpha)
    res = Tensor(func.forward(), requires_grad=A.requires_grad)
    if not requires_grad:
      return res
    res._function = func
    res._parents = [a]
    return res

  def __pow__(self, alpha, modulo=None):
    return self._unary_param_op(self, alpha, Pow)

  def matmul(self, x):
    return self._binary_ops(self, x, MatMul)

  def __add__(self, x):
    return self._binary_ops(self, x, Add)

  def __radd__(self, x): 
    return self + x

  def __mul__(self, x):
    return self._binary_ops(self, x, Mul) 

  def __rmul__(self, x): 
    return self * x

  def __neg__(self): 
    return self * -1

  def __sub__(self, x):   
    return self._binary_ops(self, -1*x, Add)

  def __rsub__(self, x):
    return x + (-self)


  #def __pow__(self, alpha, modulo=None):
  #  return self._unary_param_op(self, alpha, Pow)


if __name__ == '__main__':
  B = Tensor.zeros((3,3))
  C = Tensor.ones((3,3))
  D = Tensor.rand((3,3))
  A = Tensor.ones_like( D )
  E = A+B+C+D 
  E = A*B*C*D
  E = A-B-C-D

  E = Tensor.zeros((3,3)) +4
  E += sum( Tensor.ones((3,3)) for _ in range(3) )
  E.backward()

  print(E)


