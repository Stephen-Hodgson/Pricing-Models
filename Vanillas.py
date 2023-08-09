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
        Forward_price=self.R.forward_price(x,t,T)
        zc_bond=self.R.zc_bond(t,T)
        return zc_bond*(Forward_price*norm.cdf(d_plus(Forward_price,t,T,vol,0,K))-K*norm.cdf(d_minus(Forward_price,t,T,vol,0,K)))

    
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
        Forward_price=self.R.forward_price(x,t,T)
        zc_bond=self.R.zc_bond(t,T)
        return zc_bond*(-Forward_price*norm.cdf(-d_plus(Forward_price,t,T,vol,0,K))+K*norm.cdf(-d_minus(Forward_price,t,T,vol,0,K)))

    
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

class forward():
    def __init__(self, Underlying, Rate_Model,T,K):
        """
        Initialize a forward.

        Args:
            Underlying (Stock): Underlying stock object.
            Rate_Model (RateModel): Interest rate model.
            T (float): Maturity date
            K: Strike Price

        Returns:
            None
        """
        self.Underlying=Underlying
        self.R=Rate_Model
        self.T=T
        self.K=K
        self.vals=np.array([])

    def get_ts(self):
        return self.Underlying.get_ts()
        
    def get_vals(self):
        return self.vals

    def BSM_model(self):
        self.vals = (self.Underlying.get_vals()-self.R.zc_bond(self.Underlying.get_ts(),self.T)*self.K)

    @staticmethod
    def MC_model(self,t,T,K,vol_model,rate_model,N=10000,time_steps=1000,x_steps=100,divis=0):
        xs=np.linspace(K/3,3*K,x_steps)
        ts=np.linspace(0,T-t,time_steps)
        rates=rate_model.calc_r(ts[:-1])
        means,vars=np.zeros(xs)
        def run_stock(x):
            sum=0
            sum2=0
            for i in range(N):
                S=Stock(vol=vol_model,W=BrownianMotion.from_ts(ts),S0=x,drift=rates,divis=divis)
                sum+=S.get_S()-K
                sum2+=(S.get_S()-K)**2
            mean=sum/N
            var=sum2/N-mean
            return (mean,var)
        means,vars=np.vectorize(run_stock)(xs)
        return (xs,means,vars)
        
        
        
class call():
    def __init__(self, Underlying, Rate_Model,T,K):
        """
        Initialize a call option.

        Args:
            Underlying (Stock): Underlying stock object.
            Rate_Model (RateModel): Interest rate model.
            T (float): Maturity date
            K: Strike Price

        Returns:
            None
        """
        self.Underlying=Underlying
        self.R=Rate_Model
        self.T=T
        self.K=K
        self.vals=np.array([])

    def get_ts(self):
        return self.Underlying.get_ts()
    
    def get_vals(self):
        return self.vals
        
    #Does not yet include divis
    def BSM_model(self):
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        T=self.T
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        Forward_price=self.R.forward_price(x,t,T)
        zc_bond=self.R.zc_bond(t,T)
        R=self.R.mean(t,T)
        self.vals= zc_bond*(Forward_price*norm.cdf(d_plus(Forward_price,t,self.T,vol,0,self.K))-K*norm.cdf(d_minus(Forward_price,t,self.T,vol,0,self.K)))
        #return BSM_call(x,t,self.T,vol,R,self.K)
    
    def delta(self):
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        R=self.R.mean(t,T)
        return norm.cdf(d_plus(x,t,T,vol,R,K))
    
class put():
    def __init__(self, Underlying, Rate_Model,T,K):
        """
        Initialize a put option.

        Args:
            Underlying (Stock): Underlying stock object.
            Rate_Model (RateModel): Interest rate model.
            T (float): Maturity date
            K: Strike Price

        Returns:
            None
        """
        self.Underlying=Underlying
        self.R=Rate_Model
        self.T=T
        self.K=K
        self.vals=np.array([])

    def get_ts(self):
        return self.Underlying.get_ts()
    
    def get_vals(self):
        return self.vals
        
    def BSM_model(self):
        t=self.Underlying.get_ts()
        x=self.Underlying.get_vals()
        T=self.T
        vol=self.Underlying.vol.RMS(self.Underlying.get_ts(),T)
        Forward_price=self.R.forward_price(x,t,T)
        zc_bond=self.R.zc_bond(t,T)
        R=self.R.mean(t,T)
        self.vals = zc_bond*(-Forward_price*norm.cdf(-d_plus(Forward_price,t,self.T,vol,0,self.K))+K*norm.cdf(-d_minus(Forward_price,t,self.T,vol,0,self.K)))
        
        

#Plot derivative prices for a particular Stock
runThis=False
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

runThis=False
if __name__ == "__main__" and runThis:
    ts=np.linspace(0,10.01,1000)
    def volfun(x):
       return (x<5)*np.sin(x)**2+(x>=5)*np.sin(5)**2*np.exp(5-x)
    
    def rfun(t):
        return np.cos(t)**4
    vol=vm.NonRandomVol(volfun)
    R=rm.NonRandomRate(rfun)
    W=BrownianMotion.from_ts(ts=ts,W_0=0,drift=0)
    S=Stock(S0=1,drift=0,vol=vol,W=W)
    T=10
    K=5
    c=call(S,R,T,K)
    p=put(S,R,T,K)
    f=forward(S,R,T,K)
    c.BSM_model()
    p.BSM_model()
    f.BSM_model()
    plt.plot(c.get_ts(),c.get_vals())
    plt.plot(p.get_ts(),p.get_vals())
    plt.plot(f.get_ts(),f.get_vals())
    plt.plot(c.get_ts(),c.get_vals()-p.get_vals(),'--')
    plt.plot(c.get_ts(),c.Underlying.get_vals())
    plt.legend(['c','p','f','c-p','s'])
    plt.show()

