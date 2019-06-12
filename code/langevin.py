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


num_sims = 1000;
t_i = 0.0;
t_f = 30.0;
t_c = 15.0;
delta_t = 0.1;
N = int((t_f-t_i)/delta_t+1);

def smooth_theta(t): #Smooth interpolation function between 0 and 1
    return 0.5 * (1 + fcns.erf(t))


B = np.diag([0, np.sqrt(2*gamma*T)]) # diagonal, so independent driving Wiener processes

tspan = np.linspace(t_i, t_f, N);

def f(x, t):
    return np.array([[0, 1.0],[-k-kp*smooth_theta(t-t_c), -gamma]]).dot(x)

def G(x, t):
    return B

result = np.zeros((num_sims,N,2));


for i in range(0,num_sims-1):
    x0 = [np.random.randn(), np.random.randn()];
    result[i] = sdeint.itoint(f, G, x0, tspan);


num_bins = 20;
a = np.ceil(abs(max(np.ndarray.flatten(result[:][:][0]), key=abs)));
step_size = 2*a/num_bins;

occurences = np.zeros((num_bins,3));
occurencess = np.array([[0.0, 0.0, 0.0]]);

def counts(iterable, low, high, bins):
    step = (high - low + 0.0) / bins
    dist = collections.Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]



#for i in range(0,N-1):
#  for j in range(0,num_bins-1):
#    occurences[j,0] = delta_t*(i+1);
#    occurences[j,1] = -a+step_size*(j+1/2);
#    result_vec = np.ndarray.flatten(result[:,i,0])
#    occurences[j,2] = 1/num_sims*counts(result_vec,-a+step_size*j,-a+step_size*(j+1),1)[0]
#  occurencess = np.concatenate((occurencess, occurences), axis = 0)

#Better way

freq = np.zeros((num_bins,N));



for i in range(0,N-1):
  for j in range(0,num_bins-1):
    result_vec = np.ndarray.flatten(result[:,i,0])
    freq[j,i] = 1/num_sims*counts(result_vec,-a+step_size*j,-a+step_size*(j+1),1)[0]

binspan = np.linspace(-a, a, num_bins)


plt.contourf(tspan,binspan,freq);
plt.colorbar();
plt.show()
