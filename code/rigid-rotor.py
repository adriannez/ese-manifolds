import numpy as np
import sdeint
import matplotlib.pyplot as plt
import math

# for plotting
from mpl_toolkits import mplot3d
import collections



#Define constants
T = 0.5 #Temp
gamma = 10.0 #Viscosity
E_z = 5.0  #force .. ramped up from 0 to E_z

B = np.sqrt(2*gamma*T)

num_sims = 1000; #number of simulations
t_f = 30.0; #simulation final time
dt = 0.01; #time step size
N = 1 + int(t_f/dt); #number of steps
ts = np.linspace(0, t_f, N);


# ramp of lamba from lambda_0 to lambda_f
lambda_0 = 0
lambda_f = E_z

t_ramp = 15.0; #transition time
Dt_ramp = 1.0; #transition time-scale

def ramp(z):
    "smooth ramp from 0 to 1"
    return 0.5*(1 + np.tanh(z))


thetas = ramp((ts - t_ramp)/Dt_ramp)
lambda_s = (1 - thetas)*lambda_0 + thetas*lambda_f



def f(x, v, lambda_):
    a = np.zeros(2)
    a[0] += -gamma*v[0]+np.cos(x[0])*np.sin(x[0])*(v[1]**2)                  # dissipation
    a[1] += -(gamma+2*np.cos(x[0])/(np.sin(x[0])+dt)*v[0])*v[1] # POTENTIAL TERM in z direction
    return a

def g(x, v, lambda_):
    b = np.zeros(2)
    b += np.sqrt(dt)*B*np.random.normal(size=2) # thermal noise
    return b

def force(x,lambda_):
    c = np.zeros(2)
    c[0] = -lambda_*np.sin(x[0])
    return c

def integrator(x0, v0):
    traj = (x0,)

    x = x0
    v = v0

    for k in range(N):
        t = ts[k]
        lambda_ = lambda_s[k]

        # --- (1) Propogate velocity ---
        v = v + dt*(f(x, v, lambda_)+force(x, lambda_)) + g(x, v, lambda_)     # ramping from lambda_0 to lambda_f
        # --- (2) Propogate position ---
        x = x + dt*v

        if x[0] < 0:
            x[0] = -x[0]
            x[1] = x[1]+np.pi
            v[0] = -v[0]
        elif x[0]>np.pi:
            x[0] = 2*np.pi-x[0]
            x[1] = x[1]-np.pi
            v[0] = -v[0]

        x[1] = math.atan2(np.sin(x[1]),np.cos(x[1]))

        # --- (3) record onto trajectory ---
        traj += (x,)



    return np.array(traj)





# --- MAKES SIMULATIONS (finally!) ---
results = np.zeros([num_sims, N, 2])
inits = np.zeros([num_sims, 4])

for i in range(num_sims):
    print(i)
    # init position confined onto sphere
    # TODO: make rho_0 = rho_eq(t = 0)
    x0 = np.random.randn(2)
    x0[0] += 0.5*np.pi

    # init v0, with no radial component
    v0 = np.random.randn(2)


    traj = integrator(x0, v0)

    # store results
    inits[i, :2] = x0
    inits[i, 2:] = v0
    results[i, :, :] = traj[1:, :]






num_bins = 20; #number of bins in histogram
step_size = np.pi/num_bins; #size of bin

def counts(iterable, low, high, bins): #count the number of entries in a range for a given number of uniform bins
    step = (high - low + 0.0) / bins
    dist = collections.Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]

#Initialize freq
freq = np.zeros((num_bins,N));
freq_eq = np.zeros((num_bins,N)); # equilibrium

binspan = np.linspace(0, np.pi, num_bins)

#Fill freq: freq[j,i] is the fraction of simulations whose position at timestep i*dt is in the range -a+step_size*j < x < -a+step_size*(j+1)
for i in range(0,N):
    for j in range(0,num_bins):
        lambda_ = lambda_s[i]
        freq[j,i] = (1./num_sims)*counts(results[:,i,0], step_size*(j-.5), step_size*(j+0.5),1)[0]
        freq_eq[j,i] = np.exp((lambda_/T)*np.cos(binspan[j])) * np.sin(binspan[j])
    freq_eq[:,i] /= np.sum(freq_eq[:,i])

plt.contourf(ts,binspan,freq_eq);
plt.colorbar();
plt.ylabel('theta (radians)')
plt.xlabel('time (a.u.)')
plt.show()


#Create the plot
plt.contourf(ts,binspan,freq);
plt.colorbar();
plt.ylabel('theta (radians)')
plt.xlabel('time (a.u.)')
plt.show()
