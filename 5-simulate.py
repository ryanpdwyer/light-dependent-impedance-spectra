from tqdm import *
import tdpkefm
from tdpkefm import *
import itertools
import ray
import time
import copy

def work(Rdark, Rlight, tau_Ndec, V0_frac, phase, V, Cz, Cs, A0):
    tau, Ndec = tau_Ndec
    return (V.tp+50.0)/Ndec

def sim_trial(Rdark, Rlight, tau_Ndec, V0_frac, phase, V, Cz, Cs, A0):
    tau, Ndec = tau_Ndec
    t = np.arange(-200, V.tp+50)
    return dict(tau=tau, Rlight=Rlight, Rdark=Rdark, V0_frac=V0_frac, phase=phase, Cs=Cs,
                val=SimPhaseShiftCs(SimCs(V, Resistance(Rdark, Rlight, tau, 0.0), Cz=Cz, Cs=Cs),
                                  (A0, phase),  V0_frac, t, Ndec_response=Ndec))

@ray.remote
def sim_iter(x):
    return sim_trial(*x)


start = time.time()

ray.init()

sim_phase_shifts = []


# Setup the simulations
Rdarks = np.array([1e7])
Rlights = np.array([1e4])
tau_lights = np.array([10])
Ndec_resps = [1]
V0_fracs = np.array([0.7, 0.8, 0.9, 1.0]) # It doesn't make physical sense for V0_frac to be less than Cs/(Cs + Ct)...
phases = np.array([0 ])
Vs = [Voltage(10.0, x) for x in [256.0]]
Czs = -np.array([3.5e-05*0.8])
Css = np.array([0.5e-4])
A0s = np.array([0.050])

trials = list(itertools.product(Rdarks, Rlights, zip(tau_lights, Ndec_resps), V0_fracs, phases, Vs, Czs, Css, A0s))
total = len(trials)


ray_ids = [sim_iter.remote(x) for x in trials]
works = {id: work(*x) for x, id in zip(trials, ray_ids)}


remaining_ids = copy.copy(ray_ids)
sim_phase_shifts = []
pbar = tqdm(total=sum(works.values()), smoothing=0)
while len(remaining_ids): 
    done_id, remaining_ids = ray.wait(remaining_ids)
    sim_phase_shifts.append(ray.get(done_id[0]))
    pbar.update(works[done_id[0]])
    # print("duration =", time.time() - start, "\nresult = ", 1.0 - float(len(remaining_ids)) / len(trials)) 




dump_sims(sim_phase_shifts, "Fig5", "results")

print("\n")
print(time.time() - start)