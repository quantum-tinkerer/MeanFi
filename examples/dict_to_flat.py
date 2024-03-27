# %%
import numpy as np
from codes.kwant_helper import utils
from codes import kwant_examples

# %%
# Example hopping dictionary to use:
graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
tb_model = utils.builder2tb_model(graphene_builder)


# %%
def hop_dict_to_flat(hop_dict):
    sorted_vals = np.array(list(hop_dict.values()))[
        np.lexsort(np.array(list(hop_dict.keys())).T)
    ]
    flat = sorted_vals[..., *np.triu_indices(sorted_vals.shape[-1])].flatten()
    return flat


def flat_to_hop_dict(flat, shape, hop_dict_keys):

    matrix = np.zeros(shape, dtype=complex)
    matrix[..., *np.triu_indices(shape[-1])] = flat.reshape(*shape[:-2], -1)
    indices = np.arange(shape[-1])
    diagonal = matrix[..., indices, indices]
    matrix += np.moveaxis(matrix[-1::-1], -1, -2).conj()
    matrix[..., indices, indices] -= diagonal

    hop_dict_keys = np.array(list(hop_dict_keys))
    sorted_keys = hop_dict_keys[np.lexsort(hop_dict_keys.T)]
    hop_dict = dict(zip(map(tuple, sorted_keys), matrix))
    return hop_dict


# %%
flat = hop_dict_to_flat(tb_model)
shape = (len(tb_model.keys()), *list(tb_model.values())[0].shape)
hop_dict = flat_to_hop_dict(flat, shape, tb_model.keys())

# %%
