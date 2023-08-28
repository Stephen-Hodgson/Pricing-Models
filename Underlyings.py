import numpy as np
import matplotlib.pyplot as plt
from Random_Processes import BrownianMotion
import Volatility_Models as vm
from abc import ABC, abstractmethod

class Underlying(ABC):
    @abstractmethod
    def get_vals():
        pass
    def get_ts():
        pass


class Stock(Underlying):
    """
    Class representing a stock as an underlying asset.
    """
    def __init__(self,vol=vm.ConstantVol(1), W=None, S0=1,drift=0.,divis=0.,Q=None):
        """
        Initialize a Stock object.

        Args:
            vol (Vol_Model): Volatility model for the stock.
            W (BrownianMotion): Brownian motion for generating stock paths.
            S0 (float): Initial stock price.
            drift (float or np.ndarray): Drift of the stock.
            divis (float or np.ndarray): Dividends.
            Q: Jump process (not used).
        

    
        Returns:
            None
        """
        assert(S0>=0)

        self.S0=S0
        self.drift=drift
        self.vol=vol
        self.divis=divis
        self.W=W

        if isinstance(divis,np.ndarray):
            self.divi_factor=np.cumprod(1-np.append(np.array([0]),divis[:-1]))
        else:
            self.divi_factor=np.exp(-divis*W.get_ts())
        
        if isinstance(drift,np.ndarray):
            self.drift_factor=np.cumprod(1+np.append(np.array([0]),drift[:-1]))
        else:
            self.drift_factor=np.exp(drift*W.get_ts())

        
        if W!=None:
            (t,W,dt,dW,ts)=(W.get_t(),W.get_W(),W.get_dts(),W.get_dWs(),W.get_ts())
            sigmaW=vol.voldW(self.W)
            vol2dt=vol.vol2_dt(self.W)
            #RMS=vol.RMS(0,t)
            self.vals=S0*np.exp(drift*ts+sigmaW-vol2dt/2)
            self.vals=S0*np.exp(sigmaW-vol2dt/2)*self.divi_factor*self.drift_factor
            if len(self.vals)!=0:
                self.S=self.vals[-1]
            else:
                RMS=vol.RMS(0,t)
                self.S=S0*np.exp((drift-divis)*t+W*RMS-t*(RMS**2)/2)

            
        else:
            W=BrownianMotion()
            self.vals=np.array([])
            self.S=S0
    
    def get_ts(self):
        return self.W.get_ts()
    def get_t(self):
        return self.W.get_t()
    def get_S(self):
        return self.S
    def get_vals(self):
        """
        Get the values of the stock.

        Returns:
            np.ndarray: Array of stock values.
        """
        return self.vals
    
    def plot_S_t(self):
        """
        Plot the stock values over time.
        """
        plt.plot(self.get_ts(),self.get_vals())
        plt.xlabel("t (a.u.)")
        plt.ylabel("Stock Price")

    @classmethod
    def Constant_Params(cls,vol=1,W=None, S0=1, drift=0,divis=0, Q=None):
        """
        Create a Stock instance with constant parameters.

        Args:
            vol (float): Constant volatility.
            W (BrownianMotion): Brownian motion for generating stock paths.
            S0 (float): Initial stock price.
            drift (float): Drift of the stock.
            divis (float): Dividends.
            Q: Jump process (not used).

        Returns:
            Stock: A Stock instance with specified parameters.
        """
        volmodel=vm.ConstantVol(vol)
        S=Stock(vol=volmodel, W=W, S0=S0,drift=drift,divis=divis, Q=None,)
        return S
