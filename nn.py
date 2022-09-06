from tensor import Tensor

class Layer:
  def __init__(self, input_dim: int, output_dim: int):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.weights = Tensor.rand((input_dim, output_dim), requires_grad=True)
    self.bias = Tensor.ones((output_dim, ), requires_grad=True)

  def forward(self, x):
    res = self.weights.matmul(x) + self.bias
    pass
  def backward(self):
    pass
 
if __name__ =="__main__":

  layer = Layer(10, 11)
  layer.forward( Tensor([1]*10) ) 

