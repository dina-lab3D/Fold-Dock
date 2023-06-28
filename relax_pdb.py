from modeller import *
from modeller.automodel import *
from utils import get_model_with_chains, get_seq_aa
import os


def my_automodel(segment_ids):
    """
    """
    class MyAutoModel(automodel):
        def special_patches(self, aln):
            # Rename the chains
            self.rename_segments(segment_ids=segment_ids)
    return MyAutoModel


def make_alignment_file(pdb_model, pdb, pdb_name):
    """
    makes alignment file for modeller
    """
    chains_seq, _ = get_seq_aa(pdb_model)
    chains_seq = "/".join(chains_seq)
    with open("temp_alignment.ali", "w") as ali_file:
        ali_file.write(">P1;{}\n".format(pdb_name))
        ali_file.write("sequence:{}:::::::0.00: 0.00\n".format(pdb_name))
        ali_file.write("{}*\n".format(chains_seq))

    env = environ()
    aln = alignment(env)
    mdl = model(env, file=pdb)
    aln.append_model(mdl, align_codes=pdb, atom_files=pdb)
    aln.append(file="temp_alignment.ali", align_codes=pdb_name)
    aln.align2d()
    aln.write(file="alignment_for_modeller.ali", alignment_format='PIR')


def relax_pdb(pdb):
    """
    reconstruct side chains using modeller
    """
    log.none()
    log.level(output=0, notes=0, warnings=0, errors=0, memory=0)

    pdb_name = pdb.split(".")[0].replace("unrelaxed", "relaxed")
    pdb_model = get_model_with_chains(pdb)

    make_alignment_file(pdb_model, pdb, pdb_name)

    # log.verbose()
    env = environ()

    # directories for input atom files
    env.io.atom_files_directory = ['.', '../atom_files']

    a = my_automodel([chain.get_id() for chain in pdb_model])(env, alnfile='alignment_for_modeller.ali', knowns=pdb, sequence=pdb_name)
    a.starting_model = 1
    a.ending_model = 1
    a.make()

    # clean temp files
    for file in os.listdir(os.getcwd()):
        if file[-3:] in ['001', 'rsr', 'csh', 'ini', 'ali', 'sch']:
            os.remove(file)
    os.rename("{}.B99990001.pdb".format(pdb_name),
              "{}.pdb".format(pdb_name))