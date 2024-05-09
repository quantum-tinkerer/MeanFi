# %%
import timeit

import memray
import numpy as np
from pyinstrument import Profiler

from meanfi.kwant_helper import kwant_examples, utils
from meanfi.model import Model
from meanfi.tb.utils import guess_tb

# %%
graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()

params = {"U": 0.5, "V": 1.1}
filling = 2
nk = 600

h_int = utils.builder_to_tb(int_builder, params)
h_0 = utils.builder_to_tb(graphene_builder)
norbs = len(list(h_0.values())[0])
guess = guess_tb(frozenset(h_int), norbs)

model = Model(h_0, h_int, filling)


def scf_loop():
    model.mfield(guess, nk=nk)


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
number = 10
time_scf = timeit.timeit(scf_loop, number=number) / number

H = np.random.rand(nk, nk, norbs, norbs).astype(complex)
H += 1j * np.random.rand(nk, nk, norbs, norbs)
H += H.transpose(0, 1, 3, 2).conj()
time_diag = timeit.timeit(lambda: np.linalg.eigh(H), number=number) / number

print(
    f"Single SCF loop takes {time_scf} whereas a single diagonalization of a corresponding system takes {time_diag}"
)
