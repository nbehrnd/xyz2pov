#!/usr/bin/env python3

import string
import numpy as np
import numpy.linalg as npl
import os
import sys


class Atom:

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
        return "Atom(<%r,%r,%r>, <%r,%r,%r>, %r)\n" % (
            self.position[0], self.position[1], self.position[2], self.rgb[0],
            self.rgb[1], self.rgb[2], self.rad)

    def translate(self, vector):
        self.position -= vector


class Bond():

    def __init__(self, startAtom, endAtom):
        self.startAtom = startAtom
        self.endAtom = endAtom
        self.ID = str(startAtom.tag) + str(endAtom.tag)

    def toPOV(self):
        halfway_point = (self.startAtom.position -
                         self.endAtom.position) / 2 + self.endAtom.position

        startAtom_cylinder = "Bond(<%r,%r,%r>, <%r,%r,%r>, <%r,%r,%r>, 0.25)\n" % (
            self.startAtom.position[0], self.startAtom.position[1],
            self.startAtom.position[2], halfway_point[0], halfway_point[1],
            halfway_point[2], self.startAtom.rgb[0], self.startAtom.rgb[1],
            self.startAtom.rgb[2])

        endAtom_cylinder = "Bond(<%r,%r,%r>, <%r,%r,%r>, <%r,%r,%r>, 0.25)\n" % (
            self.endAtom.position[0], self.endAtom.position[1],
            self.endAtom.position[2], halfway_point[0], halfway_point[1],
            halfway_point[2], self.endAtom.rgb[0], self.endAtom.rgb[1],
            self.endAtom.rgb[2])
        return startAtom_cylinder + endAtom_cylinder


def get_structure(data):
    atoms = np.array([])
    with open(data) as xyz:
        for ii, line in enumerate(xyz):
            line = line.split()
            if len(line) == 4:  #and line[0] == 'C': 			#no hydrogen
                atoms = np.append(
                    atoms,
                    Atom(line[0], ii - 2,
                         [float(line[1]),
                          float(line[2]),
                          float(line[3])]))
    return atoms


def get_CoM(Molecule):

    CoM = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    for atom in Molecule:
        CoM += atom.mass * atom.position
        total_mass += atom.mass
    CoM /= total_mass

    return np.around(CoM, decimals=2)


def move2origin(Molecule, CoM):
    for atom in Molecule:
        atom.translate(CoM)


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


if __name__ == '__main__':

    molecule = get_structure('benzene.xyz')
    CoM = get_CoM(molecule)
    move2origin(molecule, CoM)

    povfile = open('benzene.pov', 'w')

    positions = np.array([atom.position for atom in molecule])
    normal = fitPlane(
        positions) * 10  #direction from which the camera is looking

    distances = np.array([abs(npl.norm(atom.position)) for atom in molecule])

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
global_settings {ambient_light rgb <0.200000002980232, 0.200000002980232, 0.200000002980232> 
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

#macro Bond(beginAtom, endAtom, col, rad)
cylinder {
   beginAtom, endAtom, rad
   pigment { color rgbt col}}
#end

union {
""" % (normal[0], normal[1], normal[2], visibility_scaling, visibility_scaling,
       light1[0], light1[1], light1[2], light2[0], light2[1], light2[2])

    povfile.write(default_settings)

    for atom in molecule:
        povfile.write(atom.toPOV())

    bond_list = [
    ]  #keeps track of the bond halfwaypoints just to avoid double counting
    for atom1 in molecule:
        for atom2 in molecule:
            bond = Bond(atom1, atom2)
            if (atom1 != atom2
                    and abs(npl.norm(atom1.position - atom2.position)) <= 1.6
                    and bond.ID not in bond_list
                    and bond.ID[::-1] not in bond_list):
                bond_list.append(bond.ID)
                povfile.write(bond.toPOV())

    povfile.write('\n}')
    povfile.close()
