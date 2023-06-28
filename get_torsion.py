#!/usr/bin/env python
# Copyright (c) 2014 Lenna X. Peterson, all rights reserved
# lenna@purdue.edu

import argparse
import logging
import os
import numpy as np
from Bio import PDB

logging.basicConfig(level=logging.DEBUG)

CHIS = dict(
    chi1=dict(
        ARG=['N', 'CA', 'CB', 'CG'],
        ASN=['N', 'CA', 'CB', 'CG'],
        ASP=['N', 'CA', 'CB', 'CG'],
        CYS=['N', 'CA', 'CB', 'SG'],
        GLN=['N', 'CA', 'CB', 'CG'],
        GLU=['N', 'CA', 'CB', 'CG'],
        HIS=['N', 'CA', 'CB', 'CG'],
        ILE=['N', 'CA', 'CB', 'CG1'],
        LEU=['N', 'CA', 'CB', 'CG'],
        LYS=['N', 'CA', 'CB', 'CG'],
        MET=['N', 'CA', 'CB', 'CG'],
        PHE=['N', 'CA', 'CB', 'CG'],
        PRO=['N', 'CA', 'CB', 'CG'],
        SER=['N', 'CA', 'CB', 'OG'],
        THR=['N', 'CA', 'CB', 'OG1'],
        TRP=['N', 'CA', 'CB', 'CG'],
        TYR=['N', 'CA', 'CB', 'CG'],
        VAL=['N', 'CA', 'CB', 'CG1'],
    ),
    altchi1=dict(
        VAL=['N', 'CA', 'CB', 'CG2'],
    ),
    chi2=dict(
        ARG=['CA', 'CB', 'CG', 'CD'],
        ASN=['CA', 'CB', 'CG', 'OD1'],
        ASP=['CA', 'CB', 'CG', 'OD1'],
        GLN=['CA', 'CB', 'CG', 'CD'],
        GLU=['CA', 'CB', 'CG', 'CD'],
        HIS=['CA', 'CB', 'CG', 'ND1'],
        ILE=['CA', 'CB', 'CG1', 'CD1'],
        LEU=['CA', 'CB', 'CG', 'CD1'],
        LYS=['CA', 'CB', 'CG', 'CD'],
        MET=['CA', 'CB', 'CG', 'SD'],
        PHE=['CA', 'CB', 'CG', 'CD1'],
        PRO=['CA', 'CB', 'CG', 'CD'],
        TRP=['CA', 'CB', 'CG', 'CD1'],
        TYR=['CA', 'CB', 'CG', 'CD1'],
    ),
    altchi2=dict(
        ASP=['CA', 'CB', 'CG', 'OD2'],
        LEU=['CA', 'CB', 'CG', 'CD2'],
        PHE=['CA', 'CB', 'CG', 'CD2'],
        TYR=['CA', 'CB', 'CG', 'CD2'],
    ),
    chi3=dict(
        ARG=['CB', 'CG', 'CD', 'NE'],
        GLN=['CB', 'CG', 'CD', 'OE1'],
        GLU=['CB', 'CG', 'CD', 'OE1'],
        LYS=['CB', 'CG', 'CD', 'CE'],
        MET=['CB', 'CG', 'SD', 'CE'],
    ),
    chi4=dict(
        ARG=['CG', 'CD', 'NE', 'CZ'],
        LYS=['CG', 'CD', 'CE', 'NZ'],
    ),
    chi5=dict(
        ARG=['CD', 'NE', 'CZ', 'NH1'],
    ),
)


class GetTorsion(object):
    """
    Calculate side-chain torsion angles (also known as dihedral or chi angles).
    Depends: Biopython (http://www.biopython.org)
    """
    chi_atoms = CHIS

    def __init__(self, chi=(1, 2, 3, 4, 5)):
        """Set parameters and calculate torsion values"""
        chi_names = list()
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in self.chi_atoms.keys():
                chi_names.append(reg_chi)
            else:
                logging.warning("Invalid chi %s", x)
        self.chi_names = chi_names

    def calculate_torsion(self, chain):
        """Calculate side-chain torsion angles for given file"""

        side_chain_xyz = np.zeros((len(chain), 15))

        for i, residues in enumerate(chain):
            # Skip heteroatoms
            res_name = residues.resname
            if residues.id[0] != " " or res_name in ["ALA", "GLY", "UNK"]:
                continue
            for j, chi in enumerate(self.chi_names):
                chi_res = self.chi_atoms[chi]
                if res_name not in chi_res:
                    break
                atom_name = chi_res[res_name][-1]
                if residues.has_id(atom_name):
                    atom_object = residues[atom_name]
                    side_chain_xyz[i, 3*j:(3*j)+3] = atom_object.get_coord()
        return side_chain_xyz
