import numpy as np
x=[1,2,3]
x=np.broadcast_to(x,(1,3)).reshape(3,1)
print(x,x.shape)