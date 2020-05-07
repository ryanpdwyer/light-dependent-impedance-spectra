import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from scipy.misc import derivative
import datetime
import copy
import json_tricks

class Resistance(object):
    def __init__(self, Rdark, Rlight, tau_L, ton):
        self.Rdark = Rdark
        self.Rlight = Rlight
        self.tau_L = tau_L
        self.ton = ton

    def __call__(self, t):
        return self.Rdark + np.where(t < self.ton, 0,
                                     (self.Rlight - self.Rdark)*(1-np.exp(-(t-self.ton)/self.tau_L))
                                    )

    def __repr__(self):
        return "Resistance(Rdark={self.Rdark}, Rlight={self.Rlight}, tau_L={self.tau_L}, ton={self.ton})".format(self=self)

class Voltage(object):
    def __init__(self, V, tp):
        self.V = V
        self.tp = tp
    
    def __call__(self, t):
        return np.where(t < self.tp, self.V, 0)

    def __repr__(self):
        return "Voltage(V={self.V}, tp={self.tp})".format(self=self)
    

Cz = -0.5e-3/100*2*3.5
Czz = 0.3e-6*4*3.5/0.062



class Sim(object):
    def __init__(self, V, R, f0=0.0625, k0=3.5, Q=26000.0, C=1e-4, Cz=Cz, Czz=Czz):
        self.omega0 = 2*np.pi*f0
        self.f0 = f0
        self.k0 = k0
        self.Q = Q
        self.omega_r2 = 2 * np.pi * f0 / Q
        self.m = k0/(self.omega0**2)
        self.C = C
        self.Cz = Cz
        self.Czz = Czz
        self.DCzz = 2*Cz**2/C
        self.Czz_q = self.Czz - self.DCzz
        self.R = R
        self.V = V
        self.responseReVec = np.vectorize(self.responseRe)

    def to_dict(self):
        return {'omega0': self.omega0, 'k0': self.k0, 'Q': self.Q,
                'C': self.C, 'Cz': self.Cz, 'Czz': self.Czz, 'DCzz': self.DCzz, 'Czz_q': self.Czz_q}
    
    def omegat(self, t):
        return 1/(self.R(t) * self.C)
    
    def Cx(self, x):
        return self.C + x * self.Cz + x**2 * self.Czz**2/2
    
    def Czx(self, x):
        return self.Cz + x * self.Czz
        
    def __call__(self, x, t):
        qt = self.qt(x, t)
        return np.array([
            x[1],
            -self.omega0**2 * x[0] - x[1]*self.omega_r2 + self.Czx(x[0]) / (2*self.Cx(x[0])**2 * self.m) * qt**2,
            -x[2]/(self.R(t)*self.Cx(x[0])) + self.V(t) / self.R(t),
        ])
    
    def qt(self, x, t):
        return x[2]

    def x_eq(self, x, t):
        return self.Cz * self.V(t)**2 / (2* self.k0)
    
    def z(self, x):
        return (x[:, 0] - 1j*x[:, 1] / self.omega0)
    
    def z_eq(self, x, t):
        return (x[:, 0] - self.x_eq(x, t) - 1j*x[:, 1]/self.omega0)
    
    def zLI(self, x, t):
        return self.z_eq(x, t) * np.exp(-1j*self.omega0 * t)
    
    def phase(self, x, t):
        return np.angle(self.zLI(x, t))
    
    def amp(self, x, t):
        return abs(self.zLI(x, t))
    
    def Hdark(self, omega=None):
        if omega is None:
            omega = self.omega0
        R = self.R(-100) # Less than 0
        return 1/(1+1j*R*self.C)

    def Prop(self, t, tau):
        return np.exp(-1*integrate.quad(self.omegat, t-tau, t)[0])
    
    def integrandRe(self, t, tau):
        return self.Prop(t, tau) * self.omegat(t - tau) * np.cos(tau*self.omega0)

    def integrandRe_omega(self, t, tau, omega):
        return self.Prop(t, tau) * self.omegat(t - tau) * np.cos(tau * omega)
    
    def integrandG(self, t, tau):
        return self.Prop(t, tau) * self.omegat(t - tau)
    
    def integrandIm(self, t, tau):
        return self.Prop(t, tau) * self.omegat(t - tau) * np.sin(tau*self.omega0)


    def integrandIm(self, t, tau, omega):
        return self.Prop(t, tau) * self.omegat(t - tau) * np.sin(tau*omega)
    
    def responseRe(self, t, omega=None):
        return self.responseReErr(t, omega)[0]

    def responseReErr(self, t, omega=None):
        if omega is None:
            omega = self.omega0
        t1 = 10 * self.integrandG(t, 0) / self.omegat(t)**2
        if t <= 0:
            return (self.respExactReMinus_omega(t, omega), 0)
        elif t * omega > 10.0: # Check if the integral is oscillatory
            if t1 < t:
                I0, ierr = integrate.quad(lambda tau: self.integrandG(t, tau), 0, t1, weight='cos', wvar=omega)
                I1, ierr1 = integrate.quad(lambda tau: self.integrandG(t, tau), t1, t, weight='cos', wvar=omega)
                I0 += I1
                ierr += ierr1
            else:
                I0, ierr = integrate.quad(lambda tau: self.integrandG(t, tau), 0, t, weight='cos', wvar=omega)
        else:
            if t1 < t:
                I0, ierr = integrate.quad(lambda tau: self.integrandRe_omega(t, tau, omega), 0, t1, points=200)
                I1, ierr1 = integrate.quad(lambda tau: self.integrandRe_omega(t, tau, omega), t1, t, points=200)
                I0 += I1
                ierr += ierr1
            else: 
                I0, ierr = integrate.quad(lambda tau: self.integrandRe_omega(t, tau, omega), 0, t, points=200)
        return (I0 + self.respExactRePlus_omega(t, omega), ierr)

    def responseReSimp(self, t, omega):
        return integrate.quad(lambda tau: self.integrandRe(t, tau), 0, np.infty)
        
        
    def respExactRePlus(self, t):
        expI0 = self.Prop(t, t)
        omega_dark = self.omegat(0)
        omega_0 = self.omega0
        return expI0*omega_dark*(omega_dark*np.cos(t*omega_0) - omega_0*np.sin(t*omega_0))/(omega_0**2 + omega_dark**2)

    def respExactRePlus_omega(self, t, omega):
        expI0 = self.Prop(t, t)
        omega_dark = self.omegat(0)
        return expI0*omega_dark*(omega_dark*np.cos(t*omega) - omega*np.sin(t*omega))/(omega**2 + omega_dark**2)

    
    def respExactReMinus(self, t):
        omega_dark = self.omegat(0)
        omega_0 = self.omega0
        return omega_dark**2/(omega_dark**2 + omega_0**2)

    def respExactReMinus_omega(self, t, omega):
        omega_dark = self.omegat(0)
        return omega_dark**2/(omega_dark**2 + omega**2)

def estimate_dphi(df, i_0, i_f):
    phi_est = 1e-3*(np.cumsum(df) - df[0])
    return (phi_est, (phi_est[i_f] - phi_est[i_0]))






class SimPhaseShift(object):
    def __init__(self, sim, A_phi, V0_frac, t, Ndec_response=4):
        """
        x0 = (A0, phi0)
        """
        self.A0 = A_phi[0]
        self.phi0 = A_phi[1]
        self.sim = sim
        self.V0_frac = V0_frac
        self.t = t
        self.x_eq0 = self.sim.x_eq([0,0, V0_frac*self.sim.V(t[0])*self.sim.C], t[0]) # Given the initial charge...
        self.sol = integrate.odeint(sim, self.x0, t=t)
        self.z = sim.zLI(self.sol, t)
        self.phi = np.unwrap(np.angle(self.z))
        self.t_filt = t_filt = t[15:] 
        self.i0 = i0 = np.argmin(abs(self.t_filt)) 
        self.ip = ip = np.argmin(abs(self.t_filt-self.sim.V.tp))
        self.phi_filt = phi_filt = np.convolve(self.phi, np.ones(16)/16.0, 'valid') # Dependent on using 16 samples / period
        self.df_filt = df_filt = np.gradient(self.phi_filt)/np.gradient(self.t_filt)
        self.t_wide = t_filt[::Ndec_response]
        self.respRePts = self.sim.responseReVec(self.t_wide)
        self.Ht = lambda t: np.interp(t, self.t_wide, self.respRePts)
        
        
        
        self.dphi_act = (phi_filt[ip] - phi_filt[i0])/ (2*np.pi)*1000
        self.phi_filt_mcyc = (phi_filt - phi_filt[0])*1e3/(2*np.pi)
        self.phi_est, self.dphi_est = estimate_dphi(self.df_python, self.i0, self.ip)
        self.error = (self.dphi_est - self.dphi_act)/self.dphi_act
    
    @property
    def df_python(self):
        sim = self.sim
        return -1e6 * sim.f0 / (4*sim.k0) * (sim.Czz_q + sim.DCzz*self.Ht(self.t_filt))*(self.sol[15:, 2]/sim.Cx(self.sol[15:, 0]))**2
    
    @property
    def x0(self):
        return np.array([self.A0*np.cos(self.phi0)+self.x_eq0, -self.A0*np.sin(self.phi0), self.V0_frac*self.sim.V(self.t[0])*self.sim.C])

    def to_dict(self):
        sim = self.sim
        R = sim.R
        V = sim.V
        return {'R': {'Rdark': R.Rdark, 'Rlight': R.Rlight, 'tau_L': R.tau_L, 'ton': R.ton},
                'V': {'V': V.V, 'tp': V.tp},
                'sim': sim.to_dict(),
                't': self.t,
                't_wide': self.t_wide,
                'respRePts': self.respRePts,
                'df_python': self.df_python,
                'df': self.df_filt,
                'phi': self.phi_filt,
                'dphi': self.phi_filt_mcyc,
                'phi_est': self.phi_est,
                'dphi_est': self.dphi_est,
                'error': self.error,
                'z': self.z,
                'sol': self.sol
               }


class SimDC(Sim):
    def __call__(self, x, t):
        return np.array([
            x[1],
            -self.omega0**2 * x[0] - x[1]*self.omega_r2 + self.Czx(x[0]) / (2*self.Cx(x[0])**2 * self.m) * x[2]**2,
            -x[2]/(self.R(t)*self.Cx(x[0])) + self.V(t) / self.R(t),
            -x[3]/(self.R(t)*self.C) + self.V(t)/self.R(t) # Just use the constant C at x = 0
        ])



class SimCs(Sim):
    def __init__(self, V, R, f0=0.0625, k0=3.5, Q=26000.0, C=1e-4, Cz=Cz, Czz=Czz, Cs=1e-4):
        self.omega0 = 2*np.pi*f0
        self.f0 = f0
        self.k0 = k0
        self.Q = Q
        self.omega_r2 = 2 * np.pi * f0 / Q
        self.m = k0/(self.omega0**2)
        self.C = C
        self.Cz = Cz
        self.Czz = Czz
        self.Cs = Cs
        self.DCzz = 2*Cz**2/C
        self.Czz_q = self.Czz - self.DCzz
        self.R = R
        self.V = V
        self.responseReVec = np.vectorize(self.responseReCs)

    def to_dict(self):
        return {'omega0': self.omega0, 'k0': self.k0, 'Q': self.Q,
                'C': self.C, 'Cz': self.Cz, 'Czz': self.Czz, 'DCzz': self.DCzz, 'Czz_q': self.Czz_q, 'Cs': self.Cs}

    def __call__(self, x, t):
        qt = self.qt(x, t)
        return np.array([
            x[1],
            -self.omega0**2 * x[0] - x[1]*self.omega_r2 + self.Czx(x[0]) / (2*self.Cx(x[0])**2 * self.m) * qt**2,
            -x[2]/(self.R(t)*(self.Cx(x[0])+self.Cs)) + self.Cx(x[0]) * self.V(t) / (self.R(t)*(self.Cx(x[0])+self.Cs)),
            -self.omegat(t)*x[3] + self.C*self.omegat(t)*self.V(t) # Just use the constant C at x = 0
        ])

    def omegat(self, t):
        return 1/(self.R(t) * (self.C+self.Cs))

    def qt(self, x, t):
        C = self.Cx(x[0])
        return C/(self.Cs + C) * x[2] + C*self.Cs/(self.Cs + C) * self.V(t)

    def qt0(self, x, t):
        C = self.C
        return C/(self.Cs + C) * x[-1] + C*self.Cs/(self.Cs + C) * self.V(t)

    def responseReCs(self, t, omega=None):
        return self.responseRe(t, omega) * self.C/(self.Cs + self.C) + self.Cs / (self.Cs + self.C)

    def Hdark(self, omega=None):
        if omega is None:
            omega = self.omega0
        R = self.R(-100) # Less than 0
        return (1 + 1j*omega*R*self.Cs) / (1 + 1j*R*(self.C + self.Cs))


class SimPhaseShiftCs(SimPhaseShift):
    def __init__(self, sim, A_phi, V0_frac, t, Ndec_response=4):
        """
        x0 = (A0, phi0)
        """
        self.A0 = A_phi[0]
        self.phi0 = A_phi[1]
        self.sim = sim
        self.V0_frac = V0_frac
        self.t = t
        self.x_eq0 = self.sim.x_eq([0, 0, V0_frac*self.sim.V(t[0])*self.sim.C], t[0]) # Given the initial charge...
        self.sol = integrate.odeint(sim, self.x0, t=t)
        self.z = sim.zLI(self.sol, t)
        self.phi = np.unwrap(np.angle(self.z))
        self.t_filt = t_filt = t[15:] 
        self.i0 = i0 = np.argmin(abs(self.t_filt)) 
        self.ip = ip = np.argmin(abs(self.t_filt-self.sim.V.tp))
        self.phi_filt = phi_filt = np.convolve(self.phi, np.ones(16)/16.0, 'valid') # Dependent on using 16 samples / period
        self.df_filt = df_filt = np.gradient(self.phi_filt)/np.gradient(self.t_filt)
        self.t_wide = t_filt[::Ndec_response]
        self.respRePts = self.sim.responseReVec(self.t_wide)
        self.Ht = lambda tt: np.interp(tt, self.t_wide, self.respRePts)
        
        
        
        self.dphi_act = (phi_filt[ip] - phi_filt[i0])/ (2*np.pi)*1000
        self.phi_filt_mcyc = (phi_filt - phi_filt[0])*1e3/(2*np.pi)
        self.phi_est, self.dphi_est = estimate_dphi(self.df_python, self.i0, self.ip)
        self.error = (self.dphi_est - self.dphi_act)/self.dphi_act



    @property
    def df_python(self):
        sim = self.sim
        return -1e6*sim.f0/(4*sim.k0) * (sim.Czz_q + sim.DCzz*self.Ht(self.t_filt))*(
        np.array([sim.qt0(s, t_) for s, t_ in zip(self.sol[15:], self.t[15:])])/sim.C)**2 # Convert q to V with a constant C term.


    @property
    def x0(self):
        # This formula does not account for the last little bit of oscillating tip charge correctly.
        q0 = self.V0_frac*self.sim.V(self.t[0])*self.sim.C
        Vx = self.sim.Cz * q0 * self.A0 / self.sim.C**2 # Magnitude of the osc voltage
        dq = np.real(self.sim.C * Vx * self.sim.Hdark() * np.exp(1j * self.phi0))
        return np.array([self.A0*np.cos(self.phi0)+self.x_eq0, -self.A0*np.sin(self.phi0), q0+dq, q0 ])

class SimPhaseShiftDC(SimPhaseShift):
    @property
    def df_python(self):
        sim = self.sim
        return -1e6*sim.f0/(4*sim.k0) * (sim.Czz_q + sim.DCzz*self.Ht(self.t_filt))*(self.sol[15:, -1]/sim.C)**2 # Convert q to V with a constant C term.

    @property
    def x0(self):
        # This formula does not account for the last little bit of oscillating tip charge correctly.
        # This is still correct even with the new model
        q0 = self.V0_frac*self.sim.V(self.t[0])*self.sim.C 
        return np.array([self.A0*np.cos(self.phi0)+self.x_eq0, -self.A0*np.sin(self.phi0), q0, q0])


def dictify_sps_container(d):
    new_val = dict(d)
    new_val['val'] = new_val['val'].to_dict() # Replace this object with a dictionary containing relevant data
    return new_val


def dump_sims(sim_list, basename, folder="."):
    d_dictified = [dictify_sps_container(x) for x in sim_list]
    now = datetime.datetime.now().strftime("%y%m%d_%H%M")
    with open(folder+"/"+now+"-"+basename+'.json', 'wb') as fh:
        json_tricks.dump(d_dictified, fh, conv_str_byte=True)


def process_dataset(dataset, func):
    """Calculate things here using the dataset, then return
    a dictionary containing names and values for each 
    calculated quantity."""
    new_dataset = copy.copy(dataset)
    del new_dataset["val"]
    new_dataset.update(func(dataset))
    return new_dataset



