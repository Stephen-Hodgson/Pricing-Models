from Underlyings import Stock
from Random_Processes import BrownianMotion
import numpy as np
import matplotlib.pyplot as plt
import Rate_Models as rm
import Volatility_Models as vm
import Underlyings
from scipy.stats import norm   
from util_functions import d_plus, d_minus, BSM_call, BSM_put



class Vanilla:
    """
    Class representing vanilla options.
    """
    def __init__(self, Underlying, Rate_Model):
        """
        Initialize a Vanilla option.

        Args:
            Underlying (Stock): Underlying stock object.
            Rate_Model (RateModel): Interest rate model.

        Returns:
            None
        """
        self.Underlying=Underlying
        self.R=Rate_Model
    
    def get_ts(self):
        return self.Underlying.get_ts()

    
    def forward(self,T,K):
        return (self.Underlying.get_vals()-self.R.zc_bond(self.Underlying.get_ts(),T)*K)
    
    def call(self,T,K):
        """
        Calculate the call option value using Black-Scholes formula.

        Args:
            T (float): Time to expiration.
            K (float): Strike price.

        Returns:
            np.ndarray: Array of call option values.
        """
        #divifactor=self.Underlying.divi_factor[-1]/self.Underlying.divi_factor
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        R=self.R.mean(t,T)
        return BSM_call(x,t,T,vol,R,K)
    
    def put(self,T,K):
        """
        Calculate the put option value using Black-Scholes formula.

        Args:
            T (float): Time to expiration.
            K (float): Strike price.

        Returns:
            np.ndarray: Array of put option values.
        """
        #divifactor=self.Underlying.divi_factor[-1]/self.Underlying.divi_factor
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        R=self.R.mean(t,T)
        return BSM_put(x,t,T,vol,R,K)
    
    def plot_derivative_values(self,T,K):
        plt.subplot(121)
        plt.plot(ts,S.get_vals())
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.subplot(122)
        plt.plot(ts,V.forward(T,K))
        plt.plot(ts,V.call(T,K))
        plt.plot(ts,V.put(T,K))
        plt.plot(ts,V.call(T,K)-V.put(T,K),'--')
        plt.xlabel('Time')
        plt.ylabel('Derivative Values')
        plt.legend(['foward','call','put','c-p'])

    @staticmethod
    def plot_call_x(Rate_Model,Vol_Model,t,T,K):
        """
        Plot call option values against stock prices.

        Args:
            Rate_Model (RateModel): Interest rate model.
            Vol_Model (Vol_Model): Volatility model.
            t (float): Current time.
            T (float): Time to expiration.
            K (float): Strike price.

        Returns:
            None
        """
        x=np.linspace(K/3,3*K,1000)
        vol=Vol_Model.RMS(t,T)
        R=Rate_Model.mean(t,T)
        c=BSM_call(x,t,T,vol,R,K)
        plt.plot(x,c)
        plt.xlabel('Stock Price')
        plt.ylabel('Call Value')

    def plot_put_x(Rate_Model,Vol_Model,t,T,K):
        """
        Plot put option values against stock prices.

        Args:
            Rate_Model (RateModel): Interest rate model.
            Vol_Model (Vol_Model): Volatility model.
            t (float): Current time.
            T (float): Time to expiration.
            K (float): Strike price.

        Returns:
            None
        """
        x=np.linspace(K/3,3*K,1000)
        vol=Vol_Model.RMS(t,T)
        R=Rate_Model.mean(t,T)
        c=BSM_put(x,t,T,vol,R,K)
        plt.plot(x,c)
        plt.xlabel('Stock Price')
        plt.ylabel('Put Value')


#Plot derivative prices for a particular Stock
runThis=True
if __name__ == "__main__" and runThis:
    ts=np.linspace(0,10.01,1000)

    def volfun(t):
        return np.sin(t)**2
    
    def rfun(t):
        return np.cos(t)**4
    
    vol=vm.NonRandomVol(volfun)
    R=rm.NonRandomRate(rfun)
    W=BrownianMotion.from_ts(ts=ts,W_0=0,drift=0)
    S=Stock(S0=1,drift=0.45,vol=vol,W=W)
    V=Vanilla(S,R)
    ts=ts[:-1]
    T=10
    K=20
    V.plot_derivative_values(T,K)
    plt.show()

#Plot BSM model option prices at various times to maturity and interest rates
runThis=False
if __name__ == "__main__" and runThis:
    def volfun(t):
        return np.sin(t)**2
    
    def rfun(t):
        return np.cos(t)**4
    
    vol=vm.NonRandomVol(volfun)
    R=rm.NonRandomRate(rfun)
    ts=np.array([1,2,3,4,5])
    T=5.01
    K=20
    for t in ts:
        Vanilla.plot_call_x(Rate_Model=R,Vol_Model=vol,t=t,T=T,K=K)
    plt.legend(["tau="+str(round(T-t,2)) for t in ts])
    plt.show()
    rates=[rm.Constant_Rate(x) for x in range(5)]
    t=4.95
    for rate in rates:
        Vanilla.plot_put_x(Rate_Model=rate,Vol_Model=vol,t=t,T=T,K=K)
    plt.legend(["rate = "+str(rate.get_r()) for rate in rates])
    plt.show()
