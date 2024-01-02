from scipy.spatial import cKDTree
from scipy.cluster.vq import vq, kmeans
import numpy as np


def add_allegro_number_array(ase_atom, n_types = 6):
    # Makes an array of numbers 0 to 5
    # for each atom depending on layer, and atom order in layer

    # atoms in layer 1:
    # for the Mo:0, top_sulfur:1, bottom_sulfur:2

    # atoms in layer 2:
    # for the Mo:3, top_sulfur:4, bottom_sulfur:5

    ''' Use:
    from ase.io import read, write
    atom = read("scf-1014_total.in", format="espresso-in")

    atom = add_allegro_number_array(atom)
    write("MoS2-moire1014.xyz", atom, format="extxyz")
    '''
    
    allegro_number_array = np.zeros(len(ase_atom), dtype=int)

    code_book, _ = kmeans(ase_atom.positions[:, 2], n_types)
    k_ind, _ = vq(ase_atom.positions[:, 2], np.sort(code_book))

    k2a = np.array([2, 0, 1, 5, 3, 4])

    allegro_number_array = k2a[k_ind]

    
    ase_atom.arrays['atom_types'] = allegro_number_array
    
    return ase_atom


# def add_allegro_number_array(ase_atom, nn = 6):
#     # Makes an array of numbers 0 to 5
#     # for each atom depending on layer, and atom order in layer

#     # atoms in layer 1:
#     # for the Mo:0, top_sulfur:1, bottom_sulfur:2

#     # atoms in layer 2:
#     # for the Mo:3, top_sulfur:4, bottom_sulfur:5

#     ''' Use:
#     from ase.io import read, write
#     atom = read("scf-1014_total.in", format="espresso-in")

#     atom = add_allegro_number_array(atom)
#     write("MoS2-moire1014.xyz", atom, format="extxyz")
#     '''

#     allegro_number_array = np.zeros(len(ase_atom), dtype=int)

#     tree = cKDTree(ase_atom.positions[:, :2])
#     dist, ind = tree.query(ase_atom.positions[:, :2], k=14)

#     for i in range(len(ase_atom)):
#         code_book, _ = kmeans(ase_atom.positions[ind[i], 2], 6)

#         k2a = [2, 0, 1, 5, 3, 4]
#         k_ind, _ = vq(ase_atom.positions[ind[i], 2][0], np.sort(code_book))

#         # k_ind[0]
#         allegro_number_array[i] = int(k2a[k_ind[0]])

#     ase_atom.arrays['atom_types'] = allegro_number_array

#     return ase_atom
