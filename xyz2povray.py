#!/usr/bin/env python3
"""Python script converts .xyz geometry file into a Pov-Ray .pov file."""

import argparse
import numpy as np
import numpy.linalg as npl


class Atom:
    """define appearance of atoms as colored spheres"""

    # define the properties for different species
    # + RGB values of the Jmol scheme as referenced by Jmol, see
    #   https://jmol.sourceforge.net/jscolors/ (accessed [2023-04-25 Tue])
    # + average covalent radii as referenced in pm on Wikipedia, see
    #   https://en.wikipedia.org/wiki/Covalent_radius (accessed [2023-04-26 Wed])
    properties = {
        'H': {'mass':   1, 'rad': 0.25, 'rgb': [1.00, 1.00, 1.00], 'r': 31, 'sd':  5},
       'He': {'mass':   4, 'rad': 0.25, 'rgb': [0.85, 1.00, 1.00], 'r': 28, 'sd':  0},
       'Li': {'mass':   6, 'rad': 0.25, 'rgb': [0.80, 0.51, 1.00], 'r':128, 'sd':  7},
       'Be': {'mass':   9, 'rad': 0.25, 'rgb': [0.76, 1.00, 0.00], 'r': 96, 'sd':  3},
        'B': {'mass':  11, 'rad': 0.25, 'rgb': [1.00, 0.71, 0.71], 'r': 84, 'sd':  3},
        'C': {'mass':  12, 'rad': 0.25, 'rgb': [0.56, 0.56, 0.56], 'r': 76, 'sd':  1},
        'N': {'mass':  14, 'rad': 0.25, 'rgb': [0.19, 0.31, 0.97], 'r': 71, 'sd':  1},
        'O': {'mass':  16, 'rad': 0.25, 'rgb': [1.00, 0.05, 0.05], 'r': 66, 'sd':  2},
        'F': {'mass':  19, 'rad': 0.25, 'rgb': [0.56, 0.88, 0.31], 'r': 57, 'sd':  3},
       'Ne': {'mass':  20, 'rad': 0.25, 'rgb': [0.70, 0.89, 0.96], 'r': 58, 'sd':  0},
       'Na': {'mass':  22, 'rad': 0.25, 'rgb': [0.67, 0.36, 0.95], 'r':166, 'sd':  9},
       'Mg': {'mass':  24, 'rad': 0.25, 'rgb': [0.54, 1.00, 0.00], 'r':141, 'sd':  7},
       'Al': {'mass':  27, 'rad': 0.25, 'rgb': [0.75, 0.65, 0.65], 'r':121, 'sd':  4},
       'Si': {'mass':  29, 'rad': 0.25, 'rgb': [0.94, 0.78, 0.63], 'r':111, 'sd':  2},
        'P': {'mass':  31, 'rad': 0.25, 'rgb': [1.00, 0.50, 0.00], 'r':107, 'sd':  3},
        'S': {'mass':  32, 'rad': 0.25, 'rgb': [1.00, 1.00, 0.19], 'r':105, 'sd':  3},
       'Cl': {'mass':  35, 'rad': 0.25, 'rgb': [0.12, 0.94, 0.12], 'r':102, 'sd':  4},
       'Ar': {'mass':  40, 'rad': 0.25, 'rgb': [0.50, 0.82, 0.89], 'r':106, 'sd': 10},
        'K': {'mass':  39, 'rad': 0.25, 'rgb': [0.56, 0.25, 0.83], 'r':203, 'sd': 12},
       'Ca': {'mass':  40, 'rad': 0.25, 'rgb': [0.24, 1.00, 0.00], 'r':176, 'sd': 10},
       'Sc': {'mass':  45, 'rad': 0.25, 'rgb': [0.90, 0.90, 0.90], 'r':170, 'sd':  7},
       'Ti': {'mass':  48, 'rad': 0.25, 'rgb': [0.75, 0.76, 0.78], 'r':160, 'sd':  8},
        'V': {'mass':  51, 'rad': 0.25, 'rgb': [0.65, 0.65, 0.78], 'r':153, 'sd':  8},
       'Cr': {'mass':  52, 'rad': 0.25, 'rgb': [0.54, 0.60, 0.78], 'r':139, 'sd':  5},
       'Mn': {'mass':  55, 'rad': 0.25, 'rgb': [0.61, 0.48, 0.78], 'r':161, 'sd':  8},
       'Fe': {'mass':  56, 'rad': 0.25, 'rgb': [0.88, 0.40, 0.20], 'r':152, 'sd':  6},
       'Co': {'mass':  59, 'rad': 0.25, 'rgb': [0.94, 0.56, 0.62], 'r':150, 'sd':  7},
       'Ni': {'mass':  59, 'rad': 0.25, 'rgb': [0.31, 0.82, 0.31], 'r':124, 'sd':  4},
       'Cu': {'mass':  63, 'rad': 0.25, 'rgb': [0.78, 0.50, 0.20], 'r':132, 'sd':  4},
       'Zn': {'mass':  65, 'rad': 0.25, 'rgb': [0.49, 0.50, 0.69], 'r':122, 'sd':  4},
       'Ga': {'mass':  70, 'rad': 0.25, 'rgb': [0.76, 0.56, 0.56], 'r':122, 'sd':  3},
       'Ge': {'mass':  73, 'rad': 0.25, 'rgb': [0.40, 0.56, 0.56], 'r':120, 'sd':  4},
       'As': {'mass':  75, 'rad': 0.25, 'rgb': [0.74, 0.50, 0.89], 'r':119, 'sd':  4},
       'Se': {'mass':  79, 'rad': 0.25, 'rgb': [1.00, 0.63, 0.00], 'r':120, 'sd':  4},
       'Br': {'mass':  80, 'rad': 0.25, 'rgb': [0.65, 0.16, 0.16], 'r':120, 'sd':  3},
       'Kr': {'mass':  84, 'rad': 0.25, 'rgb': [0.36, 0.72, 0.82], 'r':160, 'sd':  4},
       'Rb': {'mass':  86, 'rad': 0.25, 'rgb': [0.43, 0.18, 0.69], 'r':220, 'sd':  9},
       'Sr': {'mass':  88, 'rad': 0.25, 'rgb': [0.00, 1.00, 0.00], 'r':195, 'sd': 10},
        'Y': {'mass':  89, 'rad': 0.25, 'rgb': [0.58, 1.00, 1.00], 'r':190, 'sd':  7},
       'Zr': {'mass':  91, 'rad': 0.25, 'rgb': [0.58, 0.88, 0.88], 'r':175, 'sd':  7},
       'Nb': {'mass':  93, 'rad': 0.25, 'rgb': [0.45, 0.76, 0.79], 'r':164, 'sd':  6},
       'Mo': {'mass':  96, 'rad': 0.25, 'rgb': [0.32, 0.71, 0.71], 'r':154, 'sd':  5},
       'Tc': {'mass':  97, 'rad': 0.25, 'rgb': [0.23, 0.62, 0.62], 'r':147, 'sd':  7},
       'Ru': {'mass': 101, 'rad': 0.25, 'rgb': [0.14, 0.56, 0.56], 'r':146, 'sd':  7},
       'Rh': {'mass': 103, 'rad': 0.25, 'rgb': [0.04, 0.49, 0.55], 'r':142, 'sd':  7},
       'Pd': {'mass': 106, 'rad': 0.25, 'rgb': [0.00, 0.41, 0.52], 'r':139, 'sd':  6},
       'Ag': {'mass': 108, 'rad': 0.25, 'rgb': [0.75, 0.75, 0.75], 'r':145, 'sd':  5},
       'Cd': {'mass': 112, 'rad': 0.25, 'rgb': [1.00, 0.85, 0.56], 'r':144, 'sd':  9},
       'In': {'mass': 115, 'rad': 0.25, 'rgb': [0.65, 0.46, 0.45], 'r':142, 'sd':  5},
       'Sn': {'mass': 117, 'rad': 0.25, 'rgb': [0.40, 0.50, 0.50], 'r':139, 'sd':  4},
       'Sb': {'mass': 122, 'rad': 0.25, 'rgb': [0.62, 0.39, 0.71], 'r':139, 'sd':  5},
       'Te': {'mass': 128, 'rad': 0.25, 'rgb': [0.83, 0.48, 0.00], 'r':138, 'sd':  4},
        'I': {'mass': 127, 'rad': 0.25, 'rgb': [0.58, 0.00, 0.58], 'r':139, 'sd':  3},
       'Xe': {'mass': 131, 'rad': 0.25, 'rgb': [0.26, 0.62, 0.69], 'r':140, 'sd':  9},
       'Cs': {'mass': 133, 'rad': 0.25, 'rgb': [0.34, 0.09, 0.56], 'r':244, 'sd': 11},
       'Ba': {'mass': 137, 'rad': 0.25, 'rgb': [0.00, 0.79, 0.00], 'r':215, 'sd': 11},
       'La': {'mass': 139, 'rad': 0.25, 'rgb': [0.44, 0.83, 1.00], 'r':207, 'sd':  8},
       'Ce': {'mass': 140, 'rad': 0.25, 'rgb': [1.00, 1.00, 0.78], 'r':204, 'sd':  9},
       'Pr': {'mass': 141, 'rad': 0.25, 'rgb': [0.85, 1.00, 0.78], 'r':203, 'sd':  7},
       'Nd': {'mass': 144, 'rad': 0.25, 'rgb': [0.78, 1.00, 0.78], 'r':201, 'sd':  6},
       'Pm': {'mass': 145, 'rad': 0.25, 'rgb': [0.64, 1.00, 0.78], 'r':199, 'sd':  0},
       'Sm': {'mass': 150, 'rad': 0.25, 'rgb': [0.56, 1.00, 0.78], 'r':198, 'sd':  8},
       'Eu': {'mass': 152, 'rad': 0.25, 'rgb': [0.38, 1.00, 0.78], 'r':198, 'sd':  6},
       'Gd': {'mass': 157, 'rad': 0.25, 'rgb': [0.27, 1.00, 0.78], 'r':196, 'sd':  6},
       'Tb': {'mass': 159, 'rad': 0.25, 'rgb': [0.19, 1.00, 0.78], 'r':194, 'sd':  5},
       'Dy': {'mass': 163, 'rad': 0.25, 'rgb': [0.12, 1.00, 0.78], 'r':192, 'sd':  7},
       'Ho': {'mass': 165, 'rad': 0.25, 'rgb': [0.00, 1.00, 0.61], 'r':192, 'sd':  7},
       'Er': {'mass': 167, 'rad': 0.25, 'rgb': [0.00, 0.90, 0.46], 'r':189, 'sd':  6},
       'Tm': {'mass': 169, 'rad': 0.25, 'rgb': [0.00, 0.83, 0.32], 'r':190, 'sd': 10},
       'Yb': {'mass': 173, 'rad': 0.25, 'rgb': [0.00, 0.75, 0.22], 'r':187, 'sd':  8},
       'Lu': {'mass': 175, 'rad': 0.25, 'rgb': [0.00, 0.67, 0.14], 'r':175, 'sd': 10},
       'Hf': {'mass': 179, 'rad': 0.25, 'rgb': [0.30, 0.76, 1.00], 'r':187, 'sd':  8},
       'Ta': {'mass': 181, 'rad': 0.25, 'rgb': [0.30, 0.65, 1.00], 'r':170, 'sd':  8},
        'W': {'mass': 184, 'rad': 0.25, 'rgb': [0.13, 0.58, 0.84], 'r':162, 'sd':  7},
       'Re': {'mass': 186, 'rad': 0.25, 'rgb': [0.15, 0.49, 0.67], 'r':151, 'sd':  7},
       'Os': {'mass': 190, 'rad': 0.25, 'rgb': [0.15, 0.40, 0.59], 'r':144, 'sd':  4},
       'Ir': {'mass': 192, 'rad': 0.25, 'rgb': [0.09, 0.33, 0.53], 'r':141, 'sd':  6},
       'Pt': {'mass': 195, 'rad': 0.25, 'rgb': [0.82, 0.82, 0.88], 'r':136, 'sd':  5},
       'Au': {'mass': 197, 'rad': 0.25, 'rgb': [1.00, 0.82, 0.14], 'r':136, 'sd':  6},
       'Hg': {'mass': 201, 'rad': 0.25, 'rgb': [0.72, 0.72, 0.82], 'r':132, 'sd':  5},
       'Tl': {'mass': 204, 'rad': 0.25, 'rgb': [0.65, 0.33, 0.30], 'r':145, 'sd':  7},
       'Pb': {'mass': 207, 'rad': 0.25, 'rgb': [0.34, 0.35, 0.38], 'r':146, 'sd':  5},
       'Bi': {'mass': 209, 'rad': 0.25, 'rgb': [0.62, 0.31, 0.71], 'r':148, 'sd':  4},
       'Po': {'mass': 209, 'rad': 0.25, 'rgb': [0.67, 0.36, 0.00], 'r':140, 'sd':  4},
       'At': {'mass': 210, 'rad': 0.25, 'rgb': [0.46, 0.31, 0.27], 'r':150, 'sd':  0},
       'Rn': {'mass': 222, 'rad': 0.25, 'rgb': [0.26, 0.51, 0.59], 'r':150, 'sd':  0},
       'Fr': {'mass': 223, 'rad': 0.25, 'rgb': [0.26, 0.00, 0.40], 'r':260, 'sd':  0},
       'Ra': {'mass': 226, 'rad': 0.25, 'rgb': [0.00, 0.49, 0.00], 'r':221, 'sd':  2},
       'Ac': {'mass': 227, 'rad': 0.25, 'rgb': [0.44, 0.67, 1.00], 'r':215, 'sd':  0},
       'Th': {'mass': 232, 'rad': 0.25, 'rgb': [0.00, 0.73, 1.00], 'r':206, 'sd':  6},
       'Pa': {'mass': 231, 'rad': 0.25, 'rgb': [0.00, 0.63, 1.00], 'r':200, 'sd':  0},
        'U': {'mass': 238, 'rad': 0.25, 'rgb': [0.00, 0.56, 1.00], 'r':197, 'sd':  7},
       'Np': {'mass': 237, 'rad': 0.25, 'rgb': [0.00, 0.50, 1.00], 'r':190, 'sd':  1},
       'Pu': {'mass': 244, 'rad': 0.25, 'rgb': [0.00, 0.42, 1.00], 'r':187, 'sd':  1},
       'Am': {'mass': 243, 'rad': 0.25, 'rgb': [0.33, 0.36, 0.95], 'r':180, 'sd':  6},
       'Cm': {'mass': 247, 'rad': 0.25, 'rgb': [0.47, 0.36, 0.89], 'r':169, 'sd':  3},
#       'Bk': {'mass': 247, 'rad': 0.25, 'rgb': [0.54, 0.31, 0.89]},
#       'Cf': {'mass': 251, 'rad': 0.25, 'rgb': [0.63, 0.21, 0.83]},
#       'Es': {'mass': 252, 'rad': 0.25, 'rgb': [0.70, 0.12, 0.83]},
#       'Fm': {'mass': 257, 'rad': 0.25, 'rgb': [0.70, 0.12, 0.73]},
#       'Md': {'mass': 258, 'rad': 0.25, 'rgb': [0.70, 0.05, 0.65]},
#       'No': {'mass': 259, 'rad': 0.25, 'rgb': [0.74, 0.05, 0.53]},
#       'Lr': {'mass': 260, 'rad': 0.25, 'rgb': [0.78, 0.00, 0.40]},
#       'Rf': {'mass': 261, 'rad': 0.25, 'rgb': [0.80, 0.00, 0.35]},
#       'Db': {'mass': 262, 'rad': 0.25, 'rgb': [0.82, 0.00, 0.31]},
#       'Sg': {'mass': 269, 'rad': 0.25, 'rgb': [0.85, 0.00, 0.27]},
#       'Bh': {'mass': 270, 'rad': 0.25, 'rgb': [0.88, 0.00, 0.22]},
#       'Hs': {'mass': 269, 'rad': 0.25, 'rgb': [0.90, 0.00, 0.18]},
#       'Mt': {'mass': 278, 'rad': 0.25, 'rgb': [0.92, 0.00, 0.15]},
    }

    def __init__(self, species, tag, position=[0, 0, 0], covalent_radius=0.0,
        sd_covalent_radius=0.0):
        self.species = species
        self.tag = tag
        self.position = np.array(position)
        self.covalent_radius = covalent_radius
        self.sd_covalent_radius = sd_covalent_radius

        # set the properties based on the species
        if self.species in Atom.properties:
            prop = Atom.properties[self.species]
            self.mass = prop['mass']
            self.rad = prop['rad']
            self.rgb = prop['rgb']
            self.covalent_radius = prop['r']
            self.sd_covalent_radius = prop['sd']
        else:  # default to hydrogen
            self.mass = 1
            self.rad = 0.25
            self.rgb = [0.75, 0.75, 0.75]
            self.covalent_radius = 31
            self.sd_covalent_radius = 5
            print(f"WARNING: structure contains {self.species}, an atom unknown to the program.")


    def __repr__(self):
        return "%r %r tag: %r pos: %r, %r, %r" % (
            self.species, self.mass, self.tag, self.position[0],
            self.position[1], self.position[2])

    def to_pov(self):
        """assemble atom coordinates and RGB for the .pov file"""
        return "Atom(<{:6.3f},{:6.3f},{:6.3f}>, <{:4.3f}, {:4.3f}, {:4.3f}>, {:4.2f})\n".format(
            self.position[0], self.position[1], self.position[2], self.rgb[0],
            self.rgb[1], self.rgb[2], self.rad)

    def translate(self, vector):
        self.position -= vector


class Bond():
    """define appearance of bonds as struts between atom_a and atom_b"""

    def __init__(self, atom_a, atom_b):
        self.atom_a = atom_a
        self.atom_b = atom_b
        self.ID = str(atom_a.tag) + str(atom_b.tag)

    def to_pov(self):
        halfway_point = (self.atom_a.position -
                         self.atom_b.position) / 2 + self.atom_b.position

        atom_a_cylinder = "Bond(<{:6.3f},{:6.3f},{:6.3f}>, <{:6.3f},{:6.3f},{:6.3f}>, <{:4.3f},{:4.3f},{:4.3f}>, 0.25)\n".format(
            self.atom_a.position[0], self.atom_a.position[1],
            self.atom_a.position[2], halfway_point[0], halfway_point[1],
            halfway_point[2], self.atom_a.rgb[0], self.atom_a.rgb[1],
            self.atom_a.rgb[2])

        atom_b_cylinder = "Bond(<{:6.3f},{:6.3f},{:6.3f}>, <{:6.3f},{:6.3f},{:6.3f}>, <{:4.3f},{:4.3f},{:4.3f}>, 0.25)\n".format(
            self.atom_b.position[0], self.atom_b.position[1],
            self.atom_b.position[2], halfway_point[0], halfway_point[1],
            halfway_point[2], self.atom_b.rgb[0], self.atom_b.rgb[1],
            self.atom_b.rgb[2])
        return atom_a_cylinder + atom_b_cylinder


def get_structure(data):
    """access atomic coordinates"""
    atoms = np.array([])
    with open(data, mode="r", encoding="utf8") as xyz:
        for ii, line in enumerate(xyz):
            line = line.split()
            if len(line) == 4:  #and line[0] == 'C':       #no hydrogen
                atoms = np.append(
                    atoms,
                    Atom(line[0], ii - 2,
                         [float(line[1]),
                          float(line[2]),
                          float(line[3])]))
    return atoms


def get_center_of_mass(molecule):
    """determine the molecule's center of gravity

    Note: this is literally by the atoms' masses, and not only by mere dimension
    of the molecule in 3D."""
    center_of_mass = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    for atom in molecule:
        center_of_mass += atom.mass * atom.position
        total_mass += atom.mass
    center_of_mass /= total_mass

    return np.around(center_of_mass, decimals=2)


def move2origin(molecule, center_of_mass):
    """align molecule's centre of gravity and origin of the coordinate system"""
    for atom in molecule:
        atom.translate(center_of_mass)


def fitPlane(positions):
    G = np.ones((len(positions), 3))

    G[:, 0] = positions[:, 0]  # X
    G[:, 1] = positions[:, 1]  # Y

    Z = positions[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)

    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn

    return normal


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="""This Python script converts a .xyz model into a PovRay
        scene.  Hence for file `example.xyz`, there will be `example.pov for
        an individual frame.  New file `example.ini` (in simultaneous presence
        of `example.pov`) allows to generate a sequence of frames (by call of
        `povray example.ini`) to rotate the molecule around x-axis (Pov-Ray
        coordinate system)."""
    )

    parser.add_argument("source_file",
                        metavar="FILE",
                        help="Input .xyz file about the structure.")
    parser.add_argument("-v", "--verbose",
                        default=False,
                        action='store_true',
                        help="Write an optional detailed report to the CLI.")

    return parser.parse_args()


def plausible_bond(atom1, atom2,report_level=False):
    """ check if two atoms could form a bond

    By comparison of the sum of the corresponding covalent radii with
    the distance derived from data in the .xyz file, this procedure
    shall check if the two are close enough to form any bond (i.e.,
    bond order is irrelevant).  For testing purpose, the output now is
    a print to the screen; the intent for later is the provision of a
    return value."""

    check_value = False
    # computing the sum of the radii:
    observed_distance = abs(npl.norm(atom1.position - atom2.position))
    theoretical_threshold = (atom1.covalent_radius + atom2.covalent_radius) / 100

    # add the sd into the picture
    # one sigma: mean value +/- 68% of the Gaussian distribution
    # two sigma: mean value +/- 95% of the Gaussian distribution
    # three sigma:          +/- 99.7% of the Gaussian distribution
    sum_sd = (atom1.sd_covalent_radius + atom2.sd_covalent_radius) / 100
    ubound_threshold_with_sd = theoretical_threshold + (3* sum_sd)

    # a covalent bound now is set .true. below the limit of ubound
    # (different to xyz2mol, bond order is not of interest here)
    observed_distance = abs(npl.norm(atom1.position - atom2.position))
    if observed_distance <= ubound_threshold_with_sd:
        check_value = True

    report_level = args.verbose
    if report_level:
        print(f"index: {atom1.tag:4} type: {atom1.species:>2}\
     radius: {atom1.covalent_radius:>3} pm sd(radius): {atom1.sd_covalent_radius:3} pm")

        print(f"index: {atom2.tag:4} type: {atom2.species:>2}\
     radius: {atom2.covalent_radius:>3} pm sd(radius): {atom2.sd_covalent_radius:3} pm")

        print(f"{'tabulated threshold (sum of both radii):':47} {theoretical_threshold:7.2f} A")
        print(f"{'sum of both sd_covalent_radii:':47} {sum_sd:7.2f} A")
        print(57*"-")
        print(f"{'upper bound (sum radii + 3 sd of radii):':47} {ubound_threshold_with_sd:7.2f} A")
        print(57*"-")
        print(f"{'Interatomic distance calculated from .xyz file:':47} {observed_distance:7.2f} A")

        if check_value:
            print("This distance is less than the sum of the covalent radii.")
        else:
            print("Warning: This distance is too large for a covalent bond.")
        print("")

    return check_value


if __name__ == '__main__':

    args = get_args()
    input_file = str(args.source_file)
    stem_of_name = input_file.rpartition(".")[0]
    output_pov = ".".join([stem_of_name, "pov"])
    output_ini = ".".join([stem_of_name, "ini"])

    molecule = get_structure(input_file)
    center_of_mass = get_center_of_mass(molecule)
    move2origin(molecule, center_of_mass)

    with open(output_pov, mode="w", encoding="utf8") as povfile:

        positions = np.array([atom.position for atom in molecule])
        normal = fitPlane(
            positions) * 10  #direction from which the camera is looking

        distances = np.array(
            [abs(npl.norm(atom.position)) for atom in molecule])

        visibility_scaling = np.max(distances) + 0.5

        radial = abs(npl.norm(normal))
        polar = np.arccos(normal[2] / radial)
        azimuthal = np.arctan(normal[1] / normal[0])

        l1_radial = radial + 5.0
        l1_azimuthal = azimuthal + np.pi / 180.0 * 30
        l1_polar = polar + np.pi / 180.0 * 30

        l2_radial = radial + 5.0
        l2_azimuthal = azimuthal - np.pi / 180.0 * 30
        l2_polar = polar

        light1 = np.array([
            l1_radial * np.sin(l1_polar) * np.cos(l1_azimuthal),
            l1_radial * np.sin(l1_polar) * np.sin(l1_azimuthal),
            l1_radial * np.cos(l1_polar)
        ])

        light2 = np.array([
            l2_radial * np.sin(l2_polar) * np.cos(l2_azimuthal),
            l2_radial * np.sin(l2_polar) * np.sin(l2_azimuthal),
            l2_radial * np.cos(l2_polar)
        ])

        default_settings = """
    global_settings {ambient_light rgb <0.200, 0.200, 0.200>
             max_trace_level 15}

    background {color rgb <0.8,0.8,0.8>}

    camera {
      orthographic
      location <%r,%r,%r>
      right 16/9 * %r
      up %r
      look_at <0.0,0.0,0.0> }

    light_source {
      <%r,%r,%r>
      color rgb <1, 1, 1>
      fade_distance 71
      fade_power 0
      parallel
      point_at <0,0,0>}

    light_source {
      <%r,%r,%r>
      color rgb <0.05,0.05,0.05>
      fade_distance 71
      fade_power 0
      parallel
      point_at <0,0,0>}

    #default {finish {ambient .8 diffuse 1 specular 1 roughness .005 metallic 0.5}}

    #macro Atom(pos, col, rad)
    sphere {
       pos, rad
       pigment { color rgbt col}}
    #end

    #macro Bond(beginAtom, atom_b, col, rad)
    cylinder {
       beginAtom, atom_b, rad
       pigment { color rgbt col}}
    #end

declare molecule = union {
""" % (normal[0], normal[1], normal[2], visibility_scaling,
           visibility_scaling, light1[0], light1[1], light1[2], light2[0],
           light2[1], light2[2])

        povfile.write(default_settings)

        for atom in molecule:
            povfile.write(atom.to_pov())

        bond_list = [
        ]  #keeps track of the bond halfwaypoints just to avoid double counting
        for atom1 in molecule:
            for atom2 in molecule:
                bond = Bond(atom1, atom2)
                if (atom1 != atom2 and
                        # abs(npl.norm(atom1.position - atom2.position)) <= 1.6
                        plausible_bond(atom1, atom2)
                        and bond.ID not in bond_list
                        and bond.ID[::-1] not in bond_list):
                    bond_list.append(bond.ID)
                    povfile.write(bond.to_pov())
        povfile.write('\n}')

        # declare possibility for a rotation around x in the .pov of scene
        rotation_block_a = """
union{
molecule
rotate <clock*360, 0, 0>
}
        """
        povfile.write(rotation_block_a)

        # Declare a 36 frame rotation around x for file `benzene.ini`
        # To eventually render a sequence of 36 frames, this requires a run
        # `povray benzene.ini` instead of `povray benzene.pov`.  Ascertain the
        # simultaneous presence of `benzene.pov and `benzene.ini` in the very
        # same folder with permission to write records.)
        rotation_block_b = ""
        rotation_block_b += "".join(['Input_File_Name="', output_pov, '"'])
        rotation_block_b +="""
Width = 640
Height = 420
Initial_Frame = 1
Final_Frame = 36
Antialias=on"""

        with open(output_ini, mode="w", encoding="utf8") as newfile:
            newfile.write(rotation_block_b)
