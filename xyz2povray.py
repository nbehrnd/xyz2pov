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
			self.rgb = [0.4,0.4,0.4]
		else:
			self.mass = 1
			self.rad = 0.25
			self.rgb = [0.75,0.75,0.75]

	def __repr__(self):
		return "%r %r tag: %r pos: %r, %r, %r" % (
			self.species,
			self.mass,
			self.tag,
			self.position[0],
			self.position[1],
			self.position[2])

	def toPOV(self):
		return "Atom(<%r,%r,%r>, <%r,%r,%r>, %r)\n" % (self.position[0],
													   self.position[1],
													   self.position[2],
													   self.rgb[0],
													   self.rgb[1],
													   self.rgb[2],
													   self.rad)

class Bond():
	def __init__(self, startAtom, endAtom):
		self.startAtom = startAtom
		self.endAtom = endAtom
		self.ID = str(startAtom.tag) + str(endAtom.tag)

	def toPOV(self):
		halfway_point = (self.startAtom.position - self.endAtom.position) / 2 + self.endAtom.position

		startAtom_cylinder = "Bond(<%r,%r,%r>, <%r,%r,%r>, <%r,%r,%r>, 0.25)\n" % (self.startAtom.position[0],
																				 self.startAtom.position[1],
																				 self.startAtom.position[2],
																				 halfway_point[0],
																				 halfway_point[1],
																				 halfway_point[2],
																				 self.startAtom.rgb[0],
																				 self.startAtom.rgb[1],
																				 self.startAtom.rgb[2])

		endAtom_cylinder = "Bond(<%r,%r,%r>, <%r,%r,%r>, <%r,%r,%r>, 0.25)\n" % (self.endAtom.position[0],
																			   self.endAtom.position[1],
																			   self.endAtom.position[2],
																			   halfway_point[0],
																			   halfway_point[1],
																			   halfway_point[2],
																			   self.endAtom.rgb[0],
																			   self.endAtom.rgb[1],
																			   self.endAtom.rgb[2])
		return startAtom_cylinder + endAtom_cylinder


def get_structure(data):
	atoms = np.array([])
	with open(data) as xyz:
		for ii, line in enumerate(xyz):
			line = line.split()
			if len(line) == 4:
				atoms = np.append(atoms, Atom(line[0], ii - 2,
											  [float(line[1]),
											  float(line[2]),
											  float(line[3])]))
	return atoms

def get_CoM(Molecule):

	CoM = np.array([0.0,0.0,0.0])
	total_mass = 0.0

	for atom in Molecule:
		CoM += atom.mass*atom.position
		total_mass += atom.mass
	CoM /= total_mass

	return np.around(CoM, decimals=2)


if __name__ == '__main__':

	molecule = get_structure('benzene.xyz')
	CoM = get_CoM(molecule)
	povfile = open('benzene_test.pov', 'w')

	default_settings = """
global_settings {ambient_light rgb <0.200000002980232, 0.200000002980232, 0.200000002980232> 
				 max_trace_level 15} 

background {color rgb <1,1,1>}

camera {
	perspective
	location <-0.119676273571147, 0.268226323613782, -16.2536219151301>
	angle 40
	right <0.53616984737649, 0.843301580900614, -0.0369369518882763> * 1.77777777777778
	up <0.843582929576705, -0.533775428224727, 0.058750601276018>
	look_at <%r,%r,%r> }

light_source {
	<37.3852768468266, 8.49194382559059, -35.1313725898865>
	color rgb <1, 1, 1>
	fade_distance 71.2594330712784
	fade_power 0
	parallel
	point_at <-37.3852768468266, -8.49194382559059, 35.1313725898865>}

light_source {
	<5.2253807423533, -36.2337908258371, 20.2900381487961>
	color rgb <0.300000011920929, 0.300000011920929, 0.300000011920929>
	fade_distance 71.2594330712784
	fade_power 0
	parallel
	point_at <-5.2253807423533, 36.2337908258371, -20.2900381487961>}

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
""" % (CoM[0], CoM[1], CoM[2])

	povfile.write(default_settings)

	for atom in molecule:
		povfile.write(atom.toPOV())

	bond_list = []	#keeps track of the bond halfwaypoints just to avoid double counting
	for atom1 in molecule:
		for atom2 in molecule:
			bond = Bond(atom1, atom2)
			if(atom1 != atom2 and abs(npl.norm(atom1.position - atom2.position)) <= 2.0 and bond.ID not in bond_list and bond.ID[::-1] not in bond_list):
				bond_list.append(bond.ID)
				povfile.write(bond.toPOV())

	povfile.write('\n}')
	povfile.close()



"""
Auto camera placement:

draw fitplane through molecule
get normal vector
place normal at CoM
(point camera at CoM)

calculatee for atoms furthest away from CoM they are still in view range and increase camera distance till they are


include lighting placement relative to camera

"""