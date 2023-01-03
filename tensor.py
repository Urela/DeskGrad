# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# inspired by https://github.com/geohot/tinygrad/blob/4fb97b8de0e210cc37786ffdafc5dc0e990df593/tinygrad/mlops.py
import numpy as np
import inspect, functools, importlib, itertools

from typing import Union, Iterable, Optional
from functools import partialmethod

class Tensor(object):
    def __init__(self, data, requires_grad=None):

        if isinstance(data, (list, float, int)):
            data = np.array(data, dtype=np.float16)
        if isinstance(data, np.ndarray): 
            self.data = data
        else: raise Exception(f"Can't create Tensor from {data} of type {type(data)}")

        self.grad = None
        self._ctx : Optional[Function] = None
        self.requires_grad : Optional[bool] = requires_grad

    def __repr__(self): return f"<Tensor {self.data} with grad {self.grad}>"

    def backward(self):
        visited, topo_nodes = set(), []
        def toposort(node: Tensor):
            if node._ctx:
                if node not in visited:
                    visited.add(node)
                    for child in node._ctx.parents:
                        toposort(child)
                    topo_nodes.append(node)
            pass
        toposort(self)
        self.grad = Tensor.ones(*self.shape, requires_grad=False)

        for v in reversed(topo_nodes):
            if not any(x.requires_grad for x in v._ctx.parents):
                continue

            grads = v._ctx.backward(v.grad)

            #nabula = v._function.backward(v.grad.data)
            #for grad, parent in zip( nabula, v._parents):
            #    if parent.requires_grad:
            #        parent.grad = Tensor(parent.grad.data if parent.grad else 0)
            pass



    # --------- properties --------- 
    @property
    def shape(self): return self.data.shape

    def numpy(self): 
        return np.array(self.data)
    # ---------  data creation types --------- 
    @classmethod
    def zeros(cls, *shape, **kwargs): 
        return cls(np.zeros(shape, dtype=np.float16), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs): 
        return cls(np.ones(shape, dtype=np.float16), **kwargs)
  
    @classmethod
    def rand(cls, *shape, **kwargs): 
        return cls(np.random.randn(*shape).astype(np.float16), **kwargs)

    # --------- operations --------- 
    #@staticmethod
    #def broadcast(fxn, x: "Tensor"):
    #    #x = Tensor([x], device=None, requires_grad=False) if not isinstance(x, Tensor) else x
    #    return Tensor(fxn(x.data), requires_grad=False)

    ##def sum(self): return Tensor.broadcast(np.sum(), self)
    def sum(self): 
        return Tensor(np.array(np.sum(self.data)), self.requires_grad)

    def relu(self): 
        out = np.array(np.maximum(self.data,0))
        return Tensor(out, self.requires_grad)
    """
    def softmax(self):
        e_x = np.exp(self.a - np.max(self.a)) # shift values
        out = e_x / np.sum(e_x, axis=1) #[:, np.newaxis]
        return Tensor(out, self.requires_grad)
    """

class Function:
    def __init__(self, *tensors:Tensor):
        self.parents = tensors
        self.saved_tensors : List[Tensor] = []

        self.needs_input_grad = [t.requires_grad for t in self.parents]
        self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)

    def forward(self, *args, **kwargs):   
        raise NotImplementedError( "forward() is not implementd")

    def backward(self, *args, **kwargs): 
        raise NotImplementedError("backward() is not implementd")
    
    def save_for_backward(self, *x): self.saved_tensors.extend(x)

    @classmethod
    def apply(cls, *x:Tensor, **kwargs):
        ctx = cls(*x)
        #TODO: how to everything A TENSOR
        #x = [Tensor(t, requires_grad=ctx.requires_grad) if not isinstance(t,Tensor) else t for t in x]
        ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs), requires_grad=ctx.requires_grad)
        if ctx.requires_grad :#and not Tensor.no_grad:
            ret._ctx = ctx    # used by autograd engine
        return ret

# --------------- basic operations --------------- 

class Add(Function):
    def forward(self, x, y):
        return np.add(x,y)
    def backward(self, grad_output):
        return grad_output, grad_output

class Sub(Function):
    def forward(self, x, y): 
        return np.subtract(x,y)

    # TODO
    def backward(self, grad_output): 
        return grad_output, -grad_output

class Mul(Function):
    def forward(self, x, y): 
        self.save_for_backward(x, y)
        return np.multiply(x,y)
    def backward(self, grad_output): 
        x,y = self.saved_tensors
        return y*grad_output, x*grad_output


class Pow(Function):
    def forward(self, x, y):
        out = np.power(x,y)
        self.save_for_backward(x, y, out)
        return out
    def backward(self, grad_output):
        x,y,out = self.save_for_backward(x, y, out)
        grad_x = grad_output * y * (out/x)
        grad_y = grad_output * log(x) * out
        return grad_x, grad_y


class Matmul(Function):
    def forward(self, x, y): 
        self.save_for_backward(x, y)
        return np.matmul(x, y)

    def backward(self, grad):
        x, w = self.saved_tensors
        grad_x = np.matmul(grad.data, w.T)
        grad_w = np.matmul(grad.data.T, x).T
        return grad_x, grad_w

class Truediv(Function):
    def forward(self, x, y): 
        return np.true_divide(x,y)

#for name in ['Expand', 'Permute', 'Reshape', 'Add', 'Sub', 'Mul']:
#for name in [Expand, Permute, Reshape, Add, Sub, Mul]:
    #register(str(name).lower(),name)

def register(name:str, fxn:Function):
    setattr(Tensor, "_"+name if hasattr(Tensor, name) else name, functools.partialmethod(fxn.apply))

    #setattr(Tensor, "_"+name if hasattr(Tensor, name) else name, functools.partialmethod(fxn.apply,fxn))
    #setattr(Tensor, name, partialmethod(fxn.apply, fxn))

register('add',Add)
register('sub',Sub)
register('mul',Mul)
register('pow',Pow)
register('matmul',Matmul)
register('truediv',Truediv)

#eprint(inspect.getmembers( Tensor))

# register the operators
def register_op(name, fxn):
    setattr(Tensor, f"__{name}__", fxn)
    setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
    setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))
for name in ['add', 'sub', 'mul', 'pow', 'matmul', 'truediv']:
    register_op(name, getattr(Tensor, name))
    #register(name, getattr(Tensor, name))

if __name__ == '__main__':
    x = Tensor.ones(3,1)
    x = Tensor.rand(3,1)
    y = Tensor([[1.0,0,-2.0]])
    z = x.matmul(y).sum()

    z.backward()
    print(x,'\n',y,'\n\n',z)


