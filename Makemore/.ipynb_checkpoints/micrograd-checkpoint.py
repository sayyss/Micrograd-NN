import math
import numpy as np
import matplotlib.pyplot as plt
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self._prev = set(_children) # _children -> tuple, set of tuples
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
            
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        
        return out

    def __mul__(self, other):

        if not isinstance(other, Value):
            other = Value(other)
            
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        
        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward
        
        return out
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad
            
        out._backward = _backward
        return out
        
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ), "log")
    
        def _backward():
          self.grad += (1/x) * out.grad
    
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Neuron: 

    def __init__(self, nin):
        # w -> randomly generated weights
        # b -> bias
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self, x):
        # w * x + b
        # sum(iterations, start(start at bias instead of 0))
        #print(len(x))
        #print("weights:",len(self.w))
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()

        return out
        
    def parameters(self):
        return self.w + [self.b]
        
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # calls n(x) for each neuron in layer -> putting x into each neuron and storing results
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        params = []
        for neuron in self.neurons:
            local_params = neuron.parameters()
            params.extend(local_params)
        return params
        
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            #print(sz[i], sz[i+1])
            self.layers.append(Layer(sz[i], sz[i+1]))
            # i becomes num of input, i+1 is number of output

        #self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def parameters(self):
        params = []
        for layer in self.layers:
            layer_params = layer.parameters()
            params.extend(layer_params)
        
        return params
        
    def softmax(self, x):

        counts = [logit.exp() for logit in x]
        denominator = sum(counts)
        out = [c / denominator for c in counts]
        return out
        
    def cross_entropy_loss(self, x, y):

        losses = []
        for j in range(len(x)):

            loss = 0
            for i in range(len(x[j])):
                loss += y[j][i] * x[j][i].log()
            losses.append(-loss)
            
        return losses

    def __call__(self, x, activation=None):

        outputs = []
        for i in range(len(x)):
            input_i = x[i]
            for layer in self.layers:
                input_i = layer(input_i)
    
            if activation == "softmax":
                input_i = self.softmax(input_i)
                
            outputs.append(input_i)
        return outputs
    
        
    
        