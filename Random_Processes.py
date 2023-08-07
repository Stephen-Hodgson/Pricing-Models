import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion:

    """Represents a Brownian Motion """

    def __init__(self,W_0=0.,t=0.,drift=0,data=None):
        """
        Initializes the BrownianMotion object.

        Args:
            W_0 (float): Initial value of the Brownian Motion.
            t (float): Current time.
            drift (float): Drift rate.
            data (dict): Data dictionary containing recorded values.
        """
        self.t_0=t
        self.t=t
        self.W_0=W_0
        self.W=W_0+np.sqrt(t)*np.random.randn()+drift*t
        self.drift=drift
        if data:
            self.data=data
        else:
            self.data={
                "W" : np.array([]),
                "t" : np.array([]),
                "dW" : np.array([]),
                "dt" : np.array([])
            }

    def get_W(self):
        return self.W
    
    def get_t(self):
        return self.t
    
    def get_data(self):
        return self.data
    
    def get_ts(self):
        return self.data["t"]
    
    def get_Ws(self):
        return self.data["W"]
    
    def get_dWs(self):
        return self.data["dW"]
    
    def get_dts(self):
        return self.data["dt"]

    def update(self, dt, n=1, record_steps=0):
        """
        Updates the Brownian Motion by dt, n times.

        Args:
            dt (float): Time step between each measurement.
            n (int): Number of updates to perform.
            record_steps (int): Records the value of the Brownian Motion every record_steps time steps. 0 means don't record.
        """

        for i in range(n):
            dW=np.sqrt(dt)*np.random.randn()+self.drift*dt
            if record_steps and i%record_steps==0:
                self.data["t"]=np.append(self.data["t"],self.t)
                self.data["W"]=np.append(self.data["W"],self.W)
                self.data["dt"]=np.append(self.data["dt"],dt)
                self.data["dW"]=np.append(self.data["dW"],dW)
            self.t+=dt
            self.W+=dW

    def get_maxW(self):
        """
        Returns the maximum value of the brownian motion recorded.
        """
        return np.max(self.get_Ws())
    
    def get_N(self):
        return len(self.ts)

    #def calculate_stats(self):
    #    drift=(self.W-self.get_Ws[0])/(self.t-self.get_ts[0])

    
    def plot_W_t(self):
        """
        Plots the Brownian Motion over time.
        """
        plt.plot(self.get_ts(),self.get_Ws())
        plt.plot([0,self.t],[0,self.drift*self.t],'k--')
        plt.xlim([0,self.t])
        plt.ylabel("W")
        plt.xlabel("t (a.u.)")
        #plt.show()

    def __str__(self):
        return "t={}, W={}".format(self.t,self.W)
    
    def __repr__(self):
        return "Brownian_motion({},{})".format(self.W,self.t)

    def __len__(self):
        return len(self.get_ts())

    class JumpProcess:
        def __init__(self, lda, f) -> None:
            self.lda=lda
            self.f=f

    @classmethod

    def from_ts(cls,ts,W_0=0,drift=0):
        """
        Initializes a Brownian motion from an array of time values.

        Args:
            ts (numpy.ndarray): Array of time values.
            W_0 (float): Initial value of the Brownian Motion.
            drift (float): Average rate of change per unit time.

        Returns:
            BrownianMotion: Initialized BrownianMotion object.
        """
        W=cls(W_0=W_0,t=0.,drift=drift,data=None)
        n=len(ts)
        for i in range(1,n):
            W.update(ts[i]-ts[i-1], n=1, record_steps=1)
        return W
        

if __name__=="__main__":
    W=BrownianMotion(drift=1)
    W.update(dt=1/512,n=5120,record_steps=10)
    print(len(W))
    W.plot_W_t()
    plt.show()

    drift=10
    t=10
    Ws=np.array([])
    for i in range(10000):
        if i%1000==0:
            print(i)
        W=BrownianMotion(drift=drift,t=t)
        Ws=np.append(Ws,W.get_W())
    print(np.mean(Ws)/t,np.var(Ws)/t)
     

