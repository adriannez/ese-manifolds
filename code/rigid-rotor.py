import numpy as np
import sdeint
import matplotlib.pyplot as plt

# for plotting
from mpl_toolkits import mplot3d
import collections



#Define constants
T = 1.0 #Temp
gamma = 1.0 #Viscosity
E_z = 2 # force .. ramped up from 0 to E_z

B = np.sqrt(2*gamma*T)

num_sims = 300; #number of simulations
t_f = 30.0; #simulation final time
dt = 0.1; #time step size
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
    "f for rigid rotor"
    a = np.zeros(3)
    a += -gamma*v                   # dissipation
    a += B*np.random.normal(size=3) # thermal noise
    a[2] += lambda_                 # POTENTIAL TERM in z direction
    return a


def integrator(x0, v0):
    traj = (x0,)

    x = x0
    v = v0

    for k in range(N):
        t = ts[k]
        lambda_ = lambda_s[k]

        # --- (1) Propogate velocity ---
        v = v + dt*f(x, v, lambda_)         # ramping from lambda_0 to lambda_f
        # --- (2) Propogate position ---
        x = x + dt*v

        # --- (3) hack'd x to confine onto sphere, etc. ---
        x_hat = x / np.sqrt(x.dot(x))
        dx = x - x_hat

        v -= (dx / dt)
        x = x_hat

        # --- (4) record onto trajectory ---
        traj += (x,)

    return np.array(traj)





fig = plt.figure()
ax = fig.gca(projection='3d')


# --- MAKES SIMULATIONS (finally!) ---
results = np.zeros([num_sims, N, 3])
inits = np.zeros([num_sims, 6])

for i in range(num_sims):
    print i
    # init position confined onto sphere
    # TODO: make rho_0 = rho_eq(t = 0)
    x0 = np.random.randn(3)
    x0 /= np.sqrt(x0.dot(x0))

    # init v0, with no radial component
    v0 = 5*B*np.random.normal(size=3)
    v0 -= np.dot(v0, x0)*x0


    traj = integrator(x0, v0)
    ax.plot(traj[:,0], traj[:,1], traj[:,2])

    # store results
    inits[i, :3] = x0
    inits[i, 3:] = v0
    results[i, :, :] = traj[1:, :]



plt.show()
#plt.cla()



thetas = np.arccos(results[:, :, 2])


# PLOTS thetas!!!

num_bins = 20; #number of bins in histogram
a = np.pi/2; #largest entry in results
step_size = 2*a/num_bins; #size of bin

def counts(iterable, low, high, bins): #count the number of entries in a range for a given number of uniform bins
    step = (high - low + 0.0) / bins
    dist = collections.Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]

#Initialize freq
freq = np.zeros((num_bins,N));
freq_eq = np.zeros((num_bins,N)); # equilibrium

binspan = np.linspace(0, 2*a, num_bins)

#Fill freq: freq[j,i] is the fraction of simulations whose position at timestep i*dt is in the range -a+step_size*j < x < -a+step_size*(j+1)
for i in range(0,N):
    for j in range(0,num_bins-1):
        lambda_ = lambda_s[i]
        #results_vec = np.ndarray.flatten(thetas[:,i])
        freq[j,i] = (1./num_sims)*counts(thetas[:,i], step_size*(j-.5), step_size*(j+0.5),1)[0]

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


# TODO FIX THIS!
# attempt to calculate free energy
Fs_eq = np.zeros(N)
Fs = np.zeros(N)

for i in range(N):
    Es = -lambda_s[i]*np.cos(binspan)
    sum_eq = 0
    sum = 0
    Fs_eq[i] = np.dot(freq_eq[:,i][freq_eq[:,i] != 0], (Es + T*np.log(freq_eq[:,i]))[freq_eq[:,i] != 0])
    Fs[i] = np.dot(freq[:,i][freq[:,i] != 0], (Es + T*np.log(freq[:,i]))[freq[:,i] != 0])

plt.plot(Fs_eq)
plt.plot(Fs)
plt.show()
