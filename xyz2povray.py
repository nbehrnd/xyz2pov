#!/usr/bin/env python3
"""Python script converts .xyz geometry file into a Pov-Ray .pov file."""

import argparse
import numpy as np
import numpy.linalg as npl


class Atom:
    """define appearance of atoms as colored spheres"""

    def __init__(self, species, tag, position=[0, 0, 0]):
        self.species = species
        self.tag = tag
        self.position = np.array(position)

        if self.species == 'C':
            self.mass = 12
            self.rad = 0.25
            self.rgb = [0.4, 0.4, 0.4]
        else:
            self.mass = 1
            self.rad = 0.25
            self.rgb = [0.75, 0.75, 0.75]

    def __repr__(self):
        return "%r %r tag: %r pos: %r, %r, %r" % (
            self.species, self.mass, self.tag, self.position[0],
            self.position[1], self.position[2])

    def toPOV(self):
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

    def toPOV(self):
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
    atoms = np.array([])
    with open(data) as xyz:
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


def get_center_of_mass(Molecule):
    """determine the molecule's center of gravity

    Note: this is literally by the atoms' masses, and not only by mere dimension
    of the molecule in 3D."""
    center_of_mass = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    for atom in Molecule:
        center_of_mass += atom.mass * atom.position
        total_mass += atom.mass
    center_of_mass /= total_mass

    return np.around(center_of_mass, decimals=2)


def move2origin(Molecule, center_of_mass):
    """align molecule's centre of gravity and origin of the coordinate system"""
    for atom in Molecule:
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

    return parser.parse_args()


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

    background {color rgb <1,1,1>}

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
            povfile.write(atom.toPOV())

        bond_list = [
        ]  #keeps track of the bond halfwaypoints just to avoid double counting
        for atom1 in molecule:
            for atom2 in molecule:
                bond = Bond(atom1, atom2)
                if (atom1 != atom2 and
                        abs(npl.norm(atom1.position - atom2.position)) <= 1.6
                        and bond.ID not in bond_list
                        and bond.ID[::-1] not in bond_list):
                    bond_list.append(bond.ID)
                    povfile.write(bond.toPOV())

        povfile.write('\n}')

        # declare possibilty for a rotation around x in the .pov of scene
        rotation_block_a = """
union{
molecule
rotate <clock*360, 0, 0>
}
        """
        povfile.write(rotation_block_a)

        # declare a 30 frame rotation around x for file `benzene.ini`
        # To eventually render a sequence of 30 frames, this requires a run
        # `povray benzene.ini` instead of `povray benzene.pov`.  Ascertain the
        # simultaneous presence of `benzene.pov and `benzene.ini` in the very
        # same writeable folder.)
        rotation_block_b = ""
        rotation_block_b += "".join(['Input_File_Name="', output_pov, '"'])
        rotation_block_b +="""
Width = 640
Height = 420
Initial_Frame = 1
Final_Frame = 30
Antialias=on"""

        with open(output_ini, mode="w", encoding="utf8") as newfile:
            newfile.write(rotation_block_b)
