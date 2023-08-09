import numpy as np
from scipy.stats import norm

def d_plus(x,t,T,vol,R,K):
    """
    Calculate d1 for the Black-Scholes-Merton formula.

    Args:
        x (float): Current stock price.
        t (float): Current time.
        T (float): Time to expiration.
        vol (float): Volatility.
        R (float): Risk-free rate.
        K (float): Strike price.

    Returns:
        float: d1 value.
    """
    tau=T-t
    return (1/(vol*np.sqrt(tau)))*(np.log(x/K)+(R+vol**2/2)*tau)

def d_minus(x,t,T,vol,R,K):
    """
    Calculate d2 for the Black-Scholes-Merton formula.

    Args:
        x (float): Current stock price.
        t (float): Current time.
        T (float): Time to expiration.
        vol (float): Volatility.
        R (float): Risk-free rate.
        K (float): Strike price.

    Returns:
        float: d1 value.
    """
    tau=T-t
    return (1/(vol*np.sqrt(tau)))*(np.log(x/K)+(R-vol**2/2)*tau)

def BSM_call(x,t,T,vol,R,K):
    """
    Calculate the call option value using the Black-Scholes-Merton formula.

    Args:
        x (float): Current stock price.
        t (float): Current time.
        T (float): Time to expiration.
        vol (float): Volatility.
        R (float): Risk-free rate.
        K (float): Strike price.

    Returns:
        float: Call option value.
    """
    tau=T-t
    return x*norm.cdf(d_plus(x,t,T,vol,R,K))-np.exp(-R*tau)*K*norm.cdf(d_minus(x,t,T,vol,R,K))

def BSM_put(x,t,T,vol,R,K):
    """
    Calculate the put option value using the Black-Scholes-Merton formula.

    Args:
        x (float): Current stock price.
        t (float): Current time.
        T (float): Time to expiration.
        vol (float): Volatility.
        R (float): Risk-free rate.
        K (float): Strike price.

    Returns:
        float: Call option value.
    """
    tau=T-t
    return np.exp(-R*tau)*K*norm.cdf(-d_minus(x,t,T,vol,R,K))-x*norm.cdf(-d_plus(x,t,T,vol,R,K))



if __name__=="__main__":
    x=1
    t=1
    T=2
    vol=1
    R=0.1
    K=0.5
    print(BSM_call(x,t,T,vol,R,K),BSM_put(x,t,T,vol,R,K),x-np.exp(-R*(T-t))*K,BSM_call(x,t,T,vol,R,K)-BSM_put(x,t,T,vol,R,K))

    