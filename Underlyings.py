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
        
def run_monte_carlo_simulation(N,vol,epsilon,ts,drift):
    
    """Here I run a Monte Carlo Simulation to test the model. In particular, I want to check that the drift-free
    geometric brownian motion np.exp(vol*W-t*(vol**2)/2) is a martingale. I run into a problem where the
    Monte Carlo Simulation converges slowly because high stock prices affect the expectation a lot,
    but the probability of observing them in my sample is very small. The variance in the mean is 
    (exp(t*vol**2)-1)/sqrt(N), which for large t, vol can be large. I use importance sampling to resolve this.

    Let x=W(t), let p(x) be the distribution of a drift-free brownian motion and q(x) be a dummy distribution that I
    draw values of W(t) from. To minimize the variance in the montecarlo simulation, I want q(x) to be large when
    p(x)S(x) is large. We know p(x)~N(drift*t,t), using calculus, you can show that max(p(x)f(x))
    is when x=t*vol. This can be done x from q(x)~N(vol*t,t) by introducing a drift=vol to the brownian motion.
    It turns out that this is the 'optimal' distribution to draw on, and reduces the variance to zero. So as to
    not have zero variance, I include a small deviation epsilon and draw from q(x)~N((vol+epsilon)*t,t).
    The expectation is then calculated as sum(p(x_i)S(x_i)/q(x_i))/N. By comparing drift~vol to drift=0, you can see that the Monte Carlo
    simulation is much better and converges to  Mean=1, Variance=exp(t*epsilon**2)-1~t*epsilon**2 for small t.
    """
    t=ts[-1]
    vol.precalculate(0,t)
    RMS_vol=vol.RMS(0,t)
    mu=drift*t

    Sum=0
    SquareSum=0
    n=0
    for i in range(N):
        if i%(N/100)==0:
            print("{}%".format(round(i*100/N),1))
        W=BrownianMotion(drift=drift,t=t)
        S=Stock(vol=vol, W=W, S0=1,drift=drift,divis=0, Q=None,)
        x=W.get_W()
        p=np.exp(-x**2/(2*t))
        q=np.exp(-(x-mu)**2/(2*t))
        Sum+=S.get_S()*p/q
        SquareSum+=(S.get_S()*p/q)**2
        n+=1
    
    mean=(Sum/n)
    var=(SquareSum/n-mean**2)

    print("mean={}, var={}, error in mean={}".format(mean,var,np.sqrt(var/N)))

#Test Constant Volatility
runthis=False
if __name__=='__main__' and runthis:
    vol=1
    #drift=vol+epsilon
    drift=0
    t=10
    W=BrownianMotion(drift=drift,t=0)
    W.update(0.01,n=1000,record_steps=1)
    S=Stock.Constant_Params(vol=vol, W=W, S0=1,drift=drift,divis=0,Q=None)
    S.plot_S_t()
    Ss=np.exp(vol*W.get_Ws()-(W.get_ts()*vol**2)/2)
    plt.plot(W.get_ts(),Ss,'--')
    def always_one(x):
        return x+1-x
    vol=vm.NonRandomVol(always_one)
    S2=Stock(vol=vol,W=W,S0=1, drift=0)
    plt.plot(S2.get_ts(),S2.vals,':')
    plt.show()

#Test Monte Carlo Simulation for Constant Volatility
runthis=False
if __name__=='__main__' and runthis:
    sigma=1
    vol=vm.ConstantVol(sigma)
    epsilon=0.1
    drift=vol.get_vol()+epsilon
    #drift=0
    t=10
    ts=np.linspace(0,10,1000)

    run_monte_carlo_simulation(10000,vol,epsilon,ts,t,drift)

#Compare Non-Random to Constant Volatility
runthis=True
if __name__=='__main__' and runthis:
    sigma=0.5
    ts=np.linspace(0,10,1000)
    W=BrownianMotion.from_ts(ts)
    vol=vm.NonRandomVol(lambda x: sigma*(x<5))
    S=Stock(vol=vol,W=W,S0=1,drift=0)
    S.plot_S_t()
    Ss=np.exp(sigma*W.get_Ws()-W.get_ts()*sigma**2/2)
    plt.plot(W.get_ts(),Ss,'--')
    vol.plot_vol_t(W.get_ts())
    plt.show()

#Test Non-Random Volatility
runthis=False
if __name__=='__main__' and runthis:
    ts=np.linspace(0,10,1000)
    W=BrownianMotion.from_ts(ts)
    def volfun(x):
       return (x<5)*np.sin(x)**2+(x>=5)*np.sin(5)**2*np.exp(5-x)
    vol=vm.NonRandomVol(volfun)
    S=Stock(vol=vol,W=W,S0=1,drift=0)
    S.plot_S_t()
    vol.plot_vol_t(W.get_ts())
    plt.show()

#Monte Carlo Simulation for Non-random time varying volatility
runthis=False
if __name__=='__main__' and runthis:
    ts=np.linspace(0,10,1000)
    def volfun(x):
       return (x<5)*np.sin(x)**2+(x>=5)*np.sin(5)**2*np.exp(5-x)
    vol=vm.NonRandomVol(volfun)
    RMS_vol=vol.RMS(0,ts[-1])
    epsilon=0.2
    drift=RMS_vol+epsilon
    #drift=0
    N=10000
    run_monte_carlo_simulation(N,vol,epsilon,ts,drift)

runthis=False
if __name__=='__main__' and runthis:
    ts=np.linspace(0,10,1000)
    divis=0.5*(ts>0.72) * (ts<0.74)
    #plt.plot(ts,divis)
    #plt.show()
    vol=vm.ConstantVol(1)
    W=BrownianMotion.from_ts(ts)
    S=Stock(vol=vol,W=W,S0=1,drift=0,divis=divis[:-1])
    S2=Stock(vol=vol,W=W,S0=1,drift=0,divis=0)
    S3=Stock(vol=vol,W=W,S0=1,drift=divis[:-1],divis=divis[:-1])
    S.plot_S_t()
    S2.plot_S_t()
    S3.plot_S_t()
    plt.show()