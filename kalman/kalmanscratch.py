mport numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spop
 
def predict(A,B, Q, u_t,mu_t,sigma_t):
    predicted_mu=A*mu_t + B * u_t
    predicted_sigma = A * sigma_t * A +Q
    return predicted_mu, predicted_sigma 
   
def update(H,R,z,predicted_mu,predicted_sigma):
    residual_mean= z- H * predicted_mu
    residual_covariance = H * predicted_sigma * H + R
    kalman_gain= predicted_sigma * H * 1/(residual_covariance)
    updated_mu = predicted_mu + kalman_gain * residual_mean
    updated_sigma = predicted_sigma - kalman_gain * H * predicted_sigma
            
    return updated_mu, updated_sigma
    
    


class KalF():
    def __init__(self,endog):
        self.endog=endog    
    
    
    
  
    @staticmethod
    def _mse(x,endog,p=0):
        #mu_0 = np.array([0, 0])
        mu_0=x[0]
        #sigma_0 = np.array([[0.1, 0],
                     #[0, 0.1]])
        sigma_0=x[1]
        u_t=x[2]
        A=x[3]
        B=x[4]
        Q=x[5]
        H=x[6]
        R=x[7]
        #u_t = np.array([1, 1]) # we assume constant control input

       # A = np.array([[1, 0],
          #    [0, 1]])
       # B = np.array([[1, 0],
            #  [0, 1]])
      #  Q = np.array([[0.3, 0],
           #   [0, 0.3]])
       # H = np.array([[1, 0],
              #[0, 1]])
      #  R = np.array([[0.75, 0],
              #[0, 0.6]])
        measurement_states = []
        filtered_states = []

# Run KF for each time step
        mu_current = mu_0
        sigma_current = sigma_0
        num_steps=160
        for i in range(num_steps):
            predicted_mu, predicted_sigma = predict(A,B,Q, u_t, mu_current, sigma_current)
            measurement_noise = np.random.normal(0,(R))
            new_measurement = H * endog.iloc[i+1,1] + measurement_noise
            
            mu_current, sigma_current = update(H,R,new_measurement,predicted_mu,predicted_sigma)
            
            measurement_states.append(new_measurement)
            filtered_states.append(mu_current)
            
        measurement_states=np.array(measurement_states)
        filtered_states= np.array(filtered_states)
        mse = np.sum((endog.iloc[0:160,1]- filtered_states)**2)
        if p ==0:
            pass
        else:
            #plt.plot(endog.iloc[0:160,0],endog.iloc[0:160,1],c='r')
            #plt.plot(endog.iloc[0:160,0],filtered_states,c='b')
            trialrun=np.zeros(20)
            for e in range(0,20):
                trialrun[e],_=predict(A,B,Q,u_t,mu_0,sigma_0)
            plt.plot(endog.iloc[160:180,0],endog.iloc[160:180,1],c='r')
            plt.plot(endog.iloc[160:180,0],trialrun,c='b')                    
        return mse
    
    def optimize(self):
        x0=np.random.rand(8)
        opti=spop.minimize(self._mse,x0,args=(self.endog), method='Nelder-mead')
        print(opti.x)
        yur= self._mse(opti.x,self.endog,p=1)
        
    

