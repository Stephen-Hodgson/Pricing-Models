from Underlyings import Stock
from Random_Processes import BrownianMotion
import numpy as np
import matplotlib.pyplot as plt
import Rate_Models as rm
import Volatility_Models as vm
import Underlyings
from scipy.stats import norm   
from util_functions import d_plus, d_minus, BSM_call, BSM_put

#currently assumes constant R and sigma
class UpAndOutCall():
    def __init__(self, Underlying, Rate_Model,T,K,B):
        self.Underlying=Underlying
        self.R=Rate_Model
        self.T=T
        self.K=K
        self.B=B
        self.knockout=np.any(self.Underlying.get_vals()>B)
        self.vals=np.array([])
        
        if self.knockout:
            self.knockout_index=np.where(self.Underlying.get_vals()>B)[0][0]
            self.mask=np.append(np.zeros(self.knockout_index)+1,np.zeros(len(self.Underlying.get_ts())-self.knockout_index))
        else:
            self.knockout_index=np.inf
            self.mask=1


        #self.knockouttime

    def get_ts(self):
        return self.Underlying.get_ts()
    
    def get_vals(self):
        return self.vals
    
    def BSM_model(self):
        #def find_barrier():
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        T=self.T
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        R=self.R.mean(t,T)
        N=norm.cdf
        K=self.K
        B=self.B
        tau=T-t
        #vals= x*(N(d_plus(x,t,T,vol,R,K))-N(d_plus(x,t,T,vol,R,B)))-np.exp(-R*tau)*K*(N(d_minus(x,t,T,vol,R,K))-N(d_minus(x,t,T,vol,R,B)))-B*((x/B)**(-2*R/(vol**2)))*(N(d_plus(B**2,t,T,vol,R,K*x))-N(d_plus(B,t,T,vol,R,x)))+np.exp(-R*tau)*K*((x/B)**((-2*R/(vol**2))+1))*(N(d_minus(B**2,t,T,vol,R,K*x))-N(d_minus(B,t,T,vol,R,x)))
        vals=UpAndOutCall.calc_value(x,R,vol,t,T,K,B)
        vals=vals*self.mask
        self.vals= vals

    @staticmethod
    def calc_value(x,R,vol,t,T,K,B):
        N=norm.cdf
        tau=T-t
        vals= x*(N(d_plus(x,t,T,vol,R,K))-N(d_plus(x,t,T,vol,R,B)))-np.exp(-R*tau)*K*(N(d_minus(x,t,T,vol,R,K))-N(d_minus(x,t,T,vol,R,B)))-B*((x/B)**(-2*R/(vol**2)))*(N(d_plus(B**2,t,T,vol,R,K*x))-N(d_plus(B,t,T,vol,R,x)))+np.exp(-R*tau)*K*((x/B)**((-2*R/(vol**2))+1))*(N(d_minus(B**2,t,T,vol,R,K*x))-N(d_minus(B,t,T,vol,R,x)))
        return vals
    
    def plot_x(R,vol,t,T,K,B):
        x=np.linspace(0.001,B*1.1,1000)
        N=norm.cdf
        tau=T-t
        vals=UpAndOutCall.calc_value(x,R,vol,t,T,K,B)
        vals=vals*(x<B)
        plt.plot(x,vals)


runThis=False
if __name__=="__main__" and runThis:
    drift=1
    vol=vm.ConstantVol(1)
    R=rm.Constant_Rate(1)
    ts=np.linspace(0,10.01,1001)
    W=BrownianMotion.from_ts(ts=ts,W_0=0,drift=0)
    S=Stock(S0=1,drift=drift,vol=vol,W=W)

    B=20
    K=5
    T=10

    v=UpAndOutCall(S,R,T,K,B)
    v.BSM_model()
    plt.subplot(211)
    plt.plot(S.get_ts(),S.get_vals())
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.subplot(212)
    plt.plot(v.get_ts(),v.get_vals())
    plt.show()

runThis=True
if __name__ == "__main__" and runThis:
    ts=[0,3,6,9,9.3,9.6,9.9,10,10.0099]
    for t in ts:
        UpAndOutCall.plot_x(1,1,t,10.01,5,20)
    plt.show()