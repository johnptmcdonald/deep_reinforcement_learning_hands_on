
import torch
import numpy as np

# Can pass scalar values
# e.g. this creates a 3*2 tensor
a = torch.FloatTensor(3,2)

print(a)
a.zero_()
print(a)

# Or can pass iterables
# e.g. this creates a 2*3 tensor
b = torch.FloatTensor([[1,2,3],[3,2,1]])
print(b)


# or can pass it a np ndarray
n = np.zeros(shape=(3,2), dtype=np.float32) # numpy automatically creates a 64 bit float, which is unnecessary. 32 bit is fine
c = torch.tensor(n)
print(n)

n = np.zeros(shape=(3,2))
d = torch.tensor(n, dtype=torch.float32) # can also pass in the 32 bit requirement to pytorch instead

print('----')

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
v_sum = v1 + v2
v_res = (v_sum*2).sum()
print('v_res', v_res)

print(v1.is_leaf, v2.is_leaf)

v_res.backward()
print('v1.grad', v1.grad)

print('-----')
print('-----')

x = torch.tensor([1.5], requires_grad=True)
y = x*x - 3*x + 3

y.backward()
print(x.grad)

