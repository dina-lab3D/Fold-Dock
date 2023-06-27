import argparse
import os
import sys
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
from tqdm import tqdm
from utils import seq_iterator, get_antigen_input, get_antibody_input, convert_dock_input_to_score_input, is_nb, matrix_to_pdb_antibody, \
    matrix_to_pdb_antigen, get_model_with_chains, antibody_sequence, get_seq_aa
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.cealign import CEAligner
from Bio.PDB.PDBIO import PDBIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


def make_alignment_file(pdb):
    """
    makes alignment file for modeller
    """
    pdb_name = pdb.split(".")[0]
    pdb_model = get_model_with_chains(pdb)
    chains_seq, _ = get_seq_aa(pdb_model)
    chains_seq = "/".join(chains_seq)
    with open("temp_alignment.ali", "w") as ali_file:
        ali_file.write(">P1;{}_relaxed\n".format(pdb_name))
        ali_file.write("sequence:{}:::::::0.00: 0.00\n".format(pdb_name))
        ali_file.write("{}*\n".format(chains_seq))

    env = environ()
    aln = alignment(env)
    mdl = model(env, file=pdb)
    aln.append_model(mdl, align_codes=pdb, atom_files=pdb)
    aln.append(file="temp_alignment.ali", align_codes="{}_relaxed".format(pdb_name))
    aln.align2d()
    aln.write(file="alignment_for_modeller.ali", alignment_format='PIR')


def relax_pdb(pdb):
    """
    reconstruct side chains using modeller
    """
    log.none()
    log.level(output=0, notes=0, warnings=0, errors=0, memory=0)
    make_alignment_file(pdb)

    # log.verbose()
    env = environ()

    # directories for input atom files
    env.io.atom_files_directory = ['.', '../atom_files']

    a = automodel(env, alnfile='alignment_for_modeller.ali', knowns=pdb,
                  sequence="{}_relaxed".format(pdb_name))
    a.starting_model = 1
    a.ending_model = 1
    a.make()

    # clean temp files
    for file in os.listdir(os.getcwd()):
        if file[-3:] in ['001', 'rsr', 'csh', 'ini', 'ali', 'sch']:
            os.remove(file)
    os.rename("{}_relaxed.B99990001.pdb".format(pdb_name),
              "{}_relaxed.pdb".format(pdb_name))


def add_antigen_lines(ag_seq, ag_pred, ag_model, out_file):
    """
    """
    with open("temp1.pdb", "w") as file:
        matrix_to_pdb_antigen(file, ag_seq, ag_pred)

    pred_ag = PDBParser(QUIET=True).get_structure("temp1.pdb", "temp1.pdb")[0]
    align = CEAligner()
    align.set_reference(pred_ag)
    align.align(ag_model)
    out_pdb = PDBIO()
    out_pdb.set_structure(ag_model)
    out_pdb.save("temp2.pdb")
    with open("temp2.pdb", "r") as file:
        ag_lines = file.readlines()
    out_file.writelines(ag_lines)

    # clean temp files
    os.remove("temp1.pdb")
    os.remove("temp2.pdb")


def make_pdb_files(out_file_name, ab_seq, ab_pred, ag_seq=None, ag_model=None, ag_pred=None, run_modeller=False):
    """
    """
    has_antigen = ag_seq is not None and ag_pdb is not None
    heavy_seq, light_seq = antibody_sequence(ab_seq)

    # write the antibody model to the pdb file
    with open(out_file_name, "w") as file:
        matrix_to_pdb_antibody(file, heavy_seq, light_seq, ab_pred, write_end= not has_antigen)
        # We also did docking
        if has_antigen:
            add_antigen_lines(ag_seq, ag_pred, ag_model, file)

    # run modeller for relaxed models
    if run_modeller:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        relax_pdb(out_file_name)
        sys.stdout = old_stdout  # reset old stdou


def dock_and_fold(ab_sequence, ag_model, ag_seq, ag_input, dock, ab_score,
                  nb_score, topn, run_modeller):
    """
    runs Fold&Dock structure predictions
    """
    ab_input = get_antibody_input(ab_sequence.seq)
    ab_input = np.array([np.array(ab_input) for _ in range(len(ag_input))])

    # Fold, Dock and Score
    fold_dock_results = dock.predict([ab_input, ag_input])
    ab_score_input, ag_score_input, pred_ab, pred_ag = convert_dock_input_to_score_input(ab_input, ag_input, fold_dock_results)

    if ag_model is not None:
        scores = nb_score.predict([ab_score_input, ag_score_input]) if is_nb(ab_sequence.seq) else ab_score.predict([ab_score_input, ag_score_input])
        scores = -1 * np.array(scores).flatten()
    else:
        scores = np.array([np.nan])
    # get models by rank
    ranks = np.argsort(scores)
    pd.DataFrame({"model": ["{}_rank_{}".format(ab_sequence.id, i+1) for i in range(len(ranks))], "score": scores[ranks]}).to_csv("scores.csv")

    print("")

    # output models for top n results
    for rank, pred_model in enumerate(ranks[:topn]):
        model_file_path = "{}_rank_{}_unrelaxed.pdb".format(ab_sequence.id, rank + 1)
        make_pdb_files(model_file_path, ab_sequence.seq, pred_ab[pred_model], ag_seq=ag_seq, ag_model=ag_model, ag_pred=pred_ag[pred_model], run_modeller=run_modeller)


def dock_and_fold_batch(ab_fasta, ag_pdb, antigen_chains, dock, ab_score,
                        nb_score, topn, run_modeller, out_dir):
    """
    runs Fold&Dock structure predictions for a batch of antibody sequences
    """
    # make input for Fold&Dock
    sequences = []
    for sequence in seq_iterator(ab_fasta):
        sequences.append(sequence)

    # load models
    logging.getLogger('tensorflow').disabled = True
    dock = tf.keras.models.load_model(dock, compile=False)
    ab_score = tf.keras.models.load_model(ab_score, compile=False)
    nb_score = tf.keras.models.load_model(nb_score, compile=False)

    # get antigen input for the network
    ag_model = get_model_with_chains(ag_pdb, antigen_chains)
    ag_seq, ag_input = get_antigen_input(ag_model, known_epitope=None)

    # change to output directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    os.chdir(out_dir)

    for i, sequence in enumerate(sequences):

        if not os.path.exists(sequence.id):
            os.mkdir(sequence.id)
        os.chdir(sequence.id)

        start_ = timer()
        print("Working on sequence {}/{}".format(i, len(sequences)))
        dock_and_fold(sequence, ag_model, ag_seq, ag_input, dock, ab_score, nb_score, topn, run_modeller)
        end_ = timer()
        print("Finished working on sequence {}/{} with id {}, running time: {}".format(i+1, len(sequences), sequence.id, end_-start_))
        os.chdir("..")


def check_input_args():
    if not os.path.exists(args.ab_fasta):
        print("Can't find the given antibody fasta file: '{}', aborting.".format(args.ab_fasta), file=sys.stderr)
        exit(1)
    if not os.path.exists(args.ag_pdb):
        print("Can't find the given antigen pdb file: '{}', aborting.".format(args.ag_pdb), file=sys.stderr)
        exit(1)
    if not args.antigen_chains.isalpha():
        print("Antigen pdb chains should contain only a-z,A-Z characters: '{}', aborting.".format(args.antigen_chains), file=sys.stderr)
        exit(1)
    if not os.path.exists(dock_model) or not os.path.exists(ab_score_model) or not os.path.exists(nb_score_model):
        print("Can't find the trained models, aborting.", file=sys.stderr)
        exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("ab_fasta", help="fasta file with antibody sequences")

    parser.add_argument("-a", "--ag_pdb",
                        help="pdb file with the antigen structure",
                        type=str)
    parser.add_argument("-c", "--antigen_chains",
                        help="Which antigen chains to consider for docking, for example ABC, default: All chains in the pdb file)",
                        type=str)
    parser.add_argument("-o", "--output_dir",
                        help="Directory to put the predicted PDB models, (default: ./Results)",
                        type=str)
    parser.add_argument("-m", "--modeller",
                        help="Side chains reconstruction using modeller (default: False)",
                        action="store_true")
    parser.add_argument("-t", "--topn",
                        help="Number of models to generate for each antibody sequence (0-len(antigen))", default=5,
                        type=int)
    args = parser.parse_args()

    # check arguments
    run_dir_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    dock_model = os.path.join(run_dir_path, 'DockModel')
    ab_score_model = os.path.join(run_dir_path, 'AbScoreModel')
    nb_score_model = os.path.join(run_dir_path, 'NbScoreModel')

    output_directory = args.output_dir if args.output_dir else os.path.join(".", "Results")
    input_ag_pdb = os.path.abspath(args.ag_pdb) if args.ag_pdb else None

    check_input_args()

    if args.modeller:
        from modeller import *
        from modeller.automodel import *

    start = timer()
    dock_and_fold_batch(args.ab_fasta, os.path.abspath(args.ag_pdb), args.antigen_chains, dock_model, ab_score_model,
                        nb_score_model, args.topn, args.modeller, output_directory)
    end = timer()

    print("Fold&Dock ended successfully, models are located in directory:'{}', total time : {}.".format(output_directory, end - start))