import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sdeint
import scipy.special as fcns
import collections




#Define constants
T = 1.0 #Temp
gamma = 1.0 #Viscosity
k = 2.0 #Spring constant
kp = 4.0 #Value of change


num_sims = 1000; #number of simulations
t_i = 0.0; #simulation start time
t_f = 30.0; #simulation final time
t_c = 15.0; #transition time
delta_t = 0.1; #time step size
N = int((t_f-t_i)/delta_t+1); #number of steps

def smooth_theta(t): #Smooth interpolation function between 0 and 1
    return 0.5 * (1 + fcns.erf(t))

B = np.diag([0, np.sqrt(2*gamma*T)]) #noise term in Langevin dynamics

tspan = np.linspace(t_i, t_f, N);

#Integrate the Ito SDE:
#dX = f(X,t) dt + G(X,t) dW 
#Langevin:
#dx = v dt
#dv = (-gamma*v - k(t)*x) dt + \sqrt{2 gamma T} dW

#Define f
def f(x, t): 
    return np.array([[0, 1.0],[-k-kp*smooth_theta(t-t_c), -gamma]]).dot(x)

#Define G
def G(x, t):
    return B

#Initialize result
result = np.zeros((num_sims,N,2));

#Integrate for the number of simulations
for i in range(0,num_sims-1):
    x0 = [np.random.randn(), np.random.randn()];
    result[i] = sdeint.itoint(f, G, x0, tspan);

#Draw a heatmap of the distribution

num_bins = 20; #number of bins in histogram
a = np.ceil(abs(max(np.ndarray.flatten(result[:][:][0]), key=abs))); #largest entry in result
step_size = 2*a/num_bins; #size of bin

def counts(iterable, low, high, bins): #count the number of entries in a range for a given number of uniform bins
    step = (high - low + 0.0) / bins
    dist = collections.Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]

#Initialize freq
freq = np.zeros((num_bins,N));

#Fill freq: freq[j,i] is the fraction of simulations whose position at timestep i*dt is in the range -a+step_size*j < x < -a+step_size*(j+1)
for i in range(0,N-1):
  for j in range(0,num_bins-1):
    result_vec = np.ndarray.flatten(result[:,i,0])
    freq[j,i] = 1./num_sims*counts(result_vec,-a+step_size*j,-a+step_size*(j+1),1)[0]

binspan = np.linspace(-a, a, num_bins)

#Create the plot
plt.contourf(tspan,binspan,freq);
plt.colorbar();
plt.show()
