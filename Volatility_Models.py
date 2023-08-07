from abc import ABC, abstractmethod
from Random_Processes import BrownianMotion
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class VolModel(ABC):
  
    @abstractmethod
    
    def RMS(self):
        pass

class ConstantVol(VolModel):
    def __init__(self,vol):
        self.vol=vol

    def get_vol(self):
        return self.vol

    def voldW(self,W):
        return self.get_vol()*W.get_Ws()
    
    def vol2_dt(self,W):
        return self.get_vol()**2*W.get_ts()
    
    def RMS(self,t0,t1):
        return self.get_vol()
    
    def mean(self,t0,t1):
        return self.get_vol()
    
    def calc_vol(self,ts):
        return self.get_vol()
    def precalculate(self,t0,t1):
            pass
    
class NonRandomVol(VolModel):
    def __init__(self,fun):
        """
        Create a non-random, time varying volatility model.

        Args:
            fun (function): A lambda function vol(t) that takes a numpy array of size (n,) and outputs a numpy array.
        """
        self.fun=fun
        self.vol=None
        self.ts=None
        self.RMSvol=None
        self.meanvol=None
        
    def get_fun(self):
        """
        Get the volatility function.

        Returns:
            function: Volatility function.
        """
        return self.fun
    
    def calc_vol(self,t):
        """
        Calculate the volatility sigma(t) for a given value of t or numpy array of ts.

        Args:
            t (float or numpy.ndarray): Time value(s).

        Returns:
            numpy.ndarray: Array of volatility values.
        """
        return self.fun(t)

    def voldW(self,W):
        """
        Calculate an array of the Ito integral of sigma(t)*dW(t) from 0 to t.

        Args:
            W (BrownianMotion): Brownian motion process.

        Returns:
            numpy.ndarray: Array of integrated values.
        """

        ts=W.get_ts()
        n=len(ts)
        if n==0:
            return np.array([])
        vol=self.calc_vol(ts)
        sigdW=vol*W.get_dWs()
        return np.append(np.array([0]),np.cumsum(sigdW)[:-1])

        

    
    def RMS(self,t0,t1):
        """
        Calculate the Riemann integral of sigma(t)**2*dt between t0 and t1.

        Args:
            t0 (numpy.ndarray): Start time.
            t1 (numpy.ndarray): End time.

        Returns:
            numpy.ndarray: Array of integrated values.
        """
        if self.RMSvol:
            return self.RMSvol
        def elementRMS(t,T):
            ans=np.sqrt(integrate.quad(lambda x: self.fun(x)**2,t,T)[0]/(T-t))
            return ans
        return np.vectorize(elementRMS)(t0,t1)
        #res,_=np.sqrt(integrate.quad(lambda x: self.fun(x)**2,t0,t1))/(t1-t0)
        #return res
    
    def vol2_dt(self,W):
        """
        Calculate an array of the integrated value of sigma(t)**2*dt.

        Args:
            W (BrownianMotion): Brownian motion process.

        Returns:
            numpy.ndarray: Array of integrated values.
        """
        ts=W.get_ts()
        n=len(ts)
        if n==0:
            return np.array([])
        def integral(t):
            return integrate.quad(lambda x: self.fun(x)**2,ts[0],t)[0]
        
        res=np.vectorize(integral)(ts)
        return res
    
    def precalculate(self,t0,t1):
        self.meanvol=self.mean(t0,t1)
        self.RMSvol=self.RMS(t0,t1)
        

    def mean(self,t0,t1):
        """
        Calculate the mean volatility between t0 and t1.

        Args:
            t0 (float): Start time.
            t1 (float): End time.

        Returns:
            float: Mean volatility.
        """
        if self.meanvol:
            return self.meanvol
        res,_=integrate.quad(lambda x: self.fun(x),t0,t1)
        return res/(t1-t0)

    def plot_vol_t(self,ts):
        """
        Plot the volatility over time.

        Args:
            ts (numpy.ndarray): Array of time values.
        """
        plt.plot(ts,self.calc_vol(ts))




if __name__=="__main__":
    test=ConstantVol(1)
    ts=np.linspace(0,10,1000)
    W=BrownianMotion.from_ts(ts)
    ts=W.get_ts()
    print(W.get_t(),ts[-1])
    print(len(W.get_ts()),len(ts))
    plt.plot(ts,np.exp(test.voldW(W)-ts*(test.RMS(0,1)**2)/2))
    plt.plot(ts,np.exp(W.get_Ws()*1-W.get_ts()*(1**2)/2),'--')
    plt.show()



