from abc import ABC, abstractmethod
from Random_Processes import BrownianMotion
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class RateModel(ABC):
    """
    Abstract base class for rate models.
    """
    @abstractmethod
    def RMS(self):
        """
        Calculate the root mean square of the rate over the interval [t0, t1].

        Args:
            t0 (float): Start time of the interval.
            t1 (float): End time of the interval.

        Returns:
            float: Root mean square of the rate.
        """
        pass
    def zc_bond(self,t,T):
        pass
    def discount_factor(self,t,T):
        pass
    def forward_price(self,val,t,T):
        pass

class Constant_Rate(RateModel):
    def __init__(self,r):
        self.r=r

    def calc_r(self,ts=[]):
        return self.r
    
    def get_r(self,ts=[]):
        return self.r
    
    def precalculate(self,t0,t1):
        pass

    def RMS(self,t0,t1):
        return self.get_r()
    
    def mean(self,t0,t1):
        return self.get_r()
    
    def zc_bond(self,t,T):
        return np.exp(-self.r*(T-t))

    
class NonRandomRate(RateModel):
    def __init__(self,fun):
        """
        arguments:
        fun: lambda function vol(t) that takes a numpy array and ouputs a numpy array
        t numpy array of ts
        """
        self.fun=fun
        self.r=None
        self.ts=None
        self.RMSr=None
        self.meanr=None
        
    def get_fun(self):
        """
        Get the function used for calculating the interest rate.

        Returns:
            callable: Function for calculating the interest rate.
        """
        return self.fun
    
    def calc_r(self,ts=[]):
        """
        Calculate the interest rate using the provided function.

        Args:
            ts (np.ndarray): Array of time steps.

        Returns:
            np.ndarray: Array of interest rates.
        """
        return self.fun(ts)
    
    def RMS(self,t0,t1):
        """
        Calculate the root mean square of the interest rate over the interval [t0, t1].

        Args:
            t0 (float): Start time of the interval.
            t1 (float): End time of the interval.

        Returns:
            float: Root mean square of the interest rate.
        """
        if self.RMSr:
            return self.RMSr
        res,_=np.sqrt(integrate.quad(lambda x: self.fun(x)**2,t0,t1))/(t1-t0)
        return res
    
    def precalculate(self,t0,t1):
        self.meanr=self.mean(t0,t1)
        self.RMSr=self.RMS(t0,t1)

    def forward_price(self,val,t,T):
        return val/self.zc_bond(t,T)

        

    def mean(self,t0,t1):
        """
        Calculate the mean of the interest rate over the interval [t0, t1].

        Args:
            t0 (np.ndarray): Start times of the interval.
            t1 (np.ndarray): End times of the interval.

        Returns:
            np.ndarray: Array of mean interest rates.
        """
        if self.meanr:
            return self.meanr
        def integral(t,T):
            ans=integrate.quad(lambda x: self.fun(x),t,T)[0]/(T-t)
            return ans
        return np.vectorize(integral)(t0,t1)

    def plot_r_t(self,ts):
        plt.plot(ts,self.calc_r(ts))

    def zc_bond(self,t,T):
        """
        Calculate the zero-coupon bond price using the non-random interest rate.

        Args:
            t (np.ndarray): Array of current times.
            T (float): Maturity time.

        Returns:
            np.ndarray: Array of zero-coupon bond prices.
        """
        def integral(t0):
            ans,_=integrate.quad(lambda x: self.fun(x),t0,T)
            return ans
        res=np.vectorize(integral)(t)
        return np.exp(-res)
    
    def forward_price(self,val,t,T):
        return val/self.zc_bond(t,T)
    
class oneFactorVasicek(RateModel):
    pass



#calculate the bond price for a sinusoidal interest rate
if __name__ == "__main__":
    r=NonRandomRate(lambda x: np.sin(x)**2)
    #r=Constant_Rate(1)
    ts=np.array([x/100 for x in range(1000)])
    plt.plot(ts,r.calc_r(ts))
    plt.plot(ts,r.zc_bond(ts,10))
    print(r.zc_bond(2.5,10),r.zc_bond(np.array([0,2.5,5,7.5,10]),10))
    plt.show()



