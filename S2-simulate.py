from tqdm import *
import tdpkefm
from tdpkefm import *
import itertools
import ray
import time
import copy

start = time.time()

ray.init()

sim_phase_shifts = []
# pbar = tqdm(total=total);

Rdarks = np.array([1e5, 1e7])
Rlights = np.array([10])
tau_lights = np.array([10.0, 20, 40, 80, 160])
Ndec_resps = [2, 4, 4, 8, 12]
V0_fracs = np.array([0.7, 1.0])
phases = np.array([0, np.pi])
Vs = [Voltage(10.0, x) for x in [48.0, 128.0, 384.0]]
Czs = -np.array([3.5e-05, 3.5e-05*np.sqrt(2), 3.5e-05*0.8])
A0s = np.array([0.050])

total = len(Rdarks)*len(Rlights)*len(tau_lights)*len(V0_fracs)*len(phases)*len(Vs)*len(Czs)*len(A0s)

def work(Rdark, Rlight, tau_Ndec, V0_frac, phase, V, Cz, A0):
    tau, Ndec = tau_Ndec
    return (V.tp+50.0)/Ndec

def sim_trial(Rdark, Rlight, tau_Ndec, V0_frac, phase, V, Cz, A0):
    tau, Ndec = tau_Ndec
    t = np.arange(-100, V.tp+50)
    return dict(tau=tau, Rlight=Rlight, Rdark=Rdark, V0_frac=V0_frac, phase=phase,
                val=SimPhaseShiftDC(SimDC(V, Resistance(Rdark, Rlight, tau, 0.0), Cz=Cz),
                                  (A0, phase),  V0_frac, t, Ndec_response=Ndec))

@ray.remote
def sim_iter(x):
    return sim_trial(*x)

trials = list(itertools.product(Rdarks, Rlights, zip(tau_lights, Ndec_resps), V0_fracs, phases, Vs, Czs, A0s))[:1019]
ray_ids = [sim_iter.remote(x) for x in trials]
works = {id: work(*x) for x, id in zip(trials, ray_ids)}


remaining_ids = copy.copy(ray_ids)
sim_phase_shifts = []
pbar = tqdm(total=sum(works.values()))
while len(remaining_ids): 
    done_id, remaining_ids = ray.wait(remaining_ids)
    sim_phase_shifts.append(ray.get(done_id[0]))
    pbar.update(works[done_id[0]])
    # print("duration =", time.time() - start, "\nresult = ", 1.0 - float(len(remaining_ids)) / len(trials)) 




dump_sims(sim_phase_shifts, "S2", "results")

print("\n")
print(time.time() - start)