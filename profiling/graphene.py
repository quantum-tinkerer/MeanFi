# %%
import numpy as np
from codes.model import Model
from codes import kwant_examples
from codes.kwant_helper import utils
from pyinstrument import Profiler
import timeit
import memray

# %%
graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()

params = {"U": 0.5, "V": 1.1}
filling = 2
nK = 300

h_int = utils.builder2h_0(int_builder, params)
h_0 = utils.builder2h_0(graphene_builder)
guess = utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))

model = Model(h_0, h_int, filling)


def scf_loop():
    model.mfieldFFT(guess, nK=nK)


# %% Memory profile
with memray.Tracker("memoryProfile.bin"):
    scf_loop()

# %% Time profiler
profiler = Profiler()

profiler.start()
scf_loop()
profiler.stop()
profiler.write_html(path="timeProfile.html")

# %%
number = 1

timeSCF = timeit.timeit(scf_loop, number=number) / number

H = np.random.rand(nK, nK)
H += H.T.conj()
timeDiag = timeit.timeit(lambda: np.linalg.eigh(H), number=number) / number

print(
    f"Single SCF loop takes {timeSCF} whereas a single diagonalization of a corresponding system takes {timeDiag}"
)
