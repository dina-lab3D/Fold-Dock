from Bio.PDB import *
from Bio import SeqIO
import subprocess as sp
from amino_acids import modres, longer_names
from Bio.PDB.cealign import CEAligner
from Bio.PDB.PDBIO import PDBIO
import os
from abnumber import Chain

AA_DICT = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12,
           "Q": 13, "R": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20} # ,  "-": 21
MODIFIED_AA = modres
THREE_TO_ONE = longer_names
SURFACE = "surfaceResidues"  # surface <pdb>
MAX_SURFACE_VALUE = 300

def is_float(f_string):
    try:
        float(f_string)
        return True
    except ValueError:
        return False

def seq_iterator(fasta_file_path):
    """
    iterates over a fasta file
    :param fasta_file_path: path to fasta file
    :return:yields sequence, name
    """
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        yield seq_record


def is_nb(sequence):
    """
    :param sequence: string
    :return: bool
    """
    return ":" not in sequence


def three_to_one(three_letter_code):
    """
    :param three_letter_code: string (len == 3)
    :return: char
    """
    if three_letter_code in MODIFIED_AA:
        three_letter_code = MODIFIED_AA[three_letter_code]

    if three_letter_code not in THREE_TO_ONE:  # BAD amino acid
        return None
    return THREE_TO_ONE[three_letter_code]


def get_seq_aa(pdb_model, only_ca=False):
    """
    :param pdb: pdb file path
    :param chain_letters: chains to extract (ABC for example)
    :param only_ca: take only residues that have a CA atom
    :return: sequence (string), aa residues
    """
    aa_residues = []
    seq = []
    for chain in pdb_model:
        chain_aa = []
        chain_seq = ""
        for residue in chain:
            aa = residue.get_resname()
            if not three_to_one(aa) or (only_ca and not residue.has_id('CA')):
                continue
            chain_seq += three_to_one(aa)
            chain_aa.append(residue)
        seq.append(chain_seq)
        aa_residues.append(chain_aa)
    return seq, aa_residues


def get_model_with_chains(pdb, chain_letters=None):
    """
    """
    if pdb is None:
        return None
    model = PDBParser(QUIET=True).get_structure(pdb, pdb)[0]
    if chain_letters is not None:
        for chain_id in [chain.get_id() for chain in model]:
            if chain_id not in chain_letters:
                model.detach_child(chain_id)

    # The model doesnt have any of the requested chain ids
    if len(model) == 0:
        raise ValueError("The given antigen pdb {} doesn't have any of the requested chains: {} ".format(pdb, chain_letters))
    return model


def align_pdb_models(model, target, out_file_name):
    """
    """
    num_residues = len(list(model.get_residues()))
    if num_residues < 16:
        align = CEAligner(window_size=4)
    else:
        align = CEAligner()
    align.set_reference(target)
    align.align(model)
    out_pdb = PDBIO()
    out_pdb.set_structure(model)
    out_pdb.save(out_file_name)


def get_var_region(ab_chain_seq):
    if ab_chain_seq:
        try:
            ab_chain_seq = Chain(ab_chain_seq, scheme='imgt').seq
        except Exception as e:
            pass
    return ab_chain_seq


def separate_antibody_chains(antibody_sequence):
    """
    """
    chains_seq = antibody_sequence.upper().split(":")
    heavy_seq, light_seq = chains_seq[0], (None if len(chains_seq) == 1 else chains_seq[1])  # Nb / Ab
    return heavy_seq, light_seq


def get_pdb_surface(pdb_model, surface_executable=SURFACE):
    """
    calculate the antigen surface values
    """
    out_pdb = PDBIO()
    out_pdb.set_structure(pdb_model)
    out_pdb.save("surface.pdb")
    surface_values = str(sp.run("{} surface.pdb".format(surface_executable), shell=True, capture_output=True).stderr.decode("utf-8")).split("\n")[9:-1]
    os.remove("surface.pdb")
    return [float(i)/MAX_SURFACE_VALUE for i in surface_values if is_float(i)]


