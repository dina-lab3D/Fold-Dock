from scipy.spatial.transform import Rotation as R
import numpy as np
import tensorflow as tf
from Bio.Data.PDBData import protein_letters_1to3
from get_torsion import GetTorsion, CHIS
from utils import AA_DICT, get_seq_aa, get_pdb_surface, separate_antibody_chains, SURFACE

MAX_LENGTH_ANTIGEN = 600
CENTER_TRIANGLE = [[-0.826, -0.93966667, -0.09566667], [0.177,0.02833333,-0.53166667],[0.649,0.91133333, 0.62733333]]
ANTIGEN_FEATURE_NUM = len(AA_DICT) + 1 + 1 + 1  # surface column + docking column + contact column
MIN_LENGTH_ANTIGEN = 5
TORSION_CALCULATOR = GetTorsion(chi=[1, 2, 3, 4, 5])
FEATURE_NUM = len(AA_DICT) + 2 + 1 # amino acids + heavy, light columns + transformation column
LIGHT_MAX_LENGTH = 130
HEAVY_MAX_LENGTH = 150
MAX_LENGTH = LIGHT_MAX_LENGTH + HEAVY_MAX_LENGTH
BACKBONE = ["N", "CA", "C", "O", "CB"]
CHIS_KEYS = sorted([chi_name for chi_name in CHIS.keys() if 'alt' not in chi_name])


def get_antigen_one_hot(antigen_seq, antigen_surface):
    """
    get antigen one hot from its sequence
    """
    # turn in to one-hot encoding matrix
    seq_matrix = np.zeros((len(antigen_seq), ANTIGEN_FEATURE_NUM))
    for i in range(len(antigen_seq)):
        seq_matrix[i][AA_DICT[antigen_seq[i]]] = 1
        seq_matrix[i][21] = antigen_surface[i]  # surface column

    return seq_matrix


def calculate_center_transformation(center_residue):
    """
    calculates the transformation that moves this residue center triangle to the origin triangle
    """
    translation = np.array([np.average(center_residue[:,i]) for i in range(3)])
    rotation, rmsd = R.align_vectors(CENTER_TRIANGLE, center_residue-translation)
    return rotation, translation


def move_to_global_frame(antigen_patches_xyz, antigen_patches_side_chains, patches_centers):
    """
    moves the antigen input to the reference frame
    """
    centered_antigen_patches = []
    centered_antigen_side_chains = []
    patch_mask = np.array(antigen_patches_xyz) != 0.0
    side_chain_mask = np.array(antigen_patches_side_chains) != 0.0

    for center, patch, mask, side_chains_patch, side_mask in zip(patches_centers, antigen_patches_xyz, patch_mask, antigen_patches_side_chains, side_chain_mask):
        rotation, translation = calculate_center_transformation(center)

        patch = np.reshape(patch, newshape=(patch.shape[0] * 5, 3))
        side_chains_patch = np.reshape(side_chains_patch, newshape=(side_chains_patch.shape[0] * 5, 3))

        patch = rotation.apply(patch - translation)
        side_chains_patch = rotation.apply(side_chains_patch - translation)

        patch = np.reshape(patch, newshape=(patch.shape[0] // 5, 15)) * mask
        side_chains_patch = np.reshape(side_chains_patch, newshape=(side_chains_patch.shape[0] // 5, 15)) * side_mask

        pad_patch = np.zeros((patch.shape[0] + 1, patch.shape[1]))
        pad_side_chains_patch = np.zeros((side_chains_patch.shape[0] + 1, side_chains_patch.shape[1]))

        pad_patch[:-1,:] = patch
        pad_side_chains_patch[:-1,:] = side_chains_patch

        centered_antigen_patches.append(pad_patch)
        centered_antigen_side_chains.append(pad_side_chains_patch)

    return centered_antigen_patches, centered_antigen_side_chains


def get_antigen_xyz(antigen_seq, antigen_residues):
    """
    get the antigen xyz coordinates
    """
    ag_xyz_matrix = np.zeros((len(antigen_seq), 15))

    # get the heavy coordinates
    for i in range(len(antigen_seq)):
        for j, atom in enumerate(BACKBONE):
            if antigen_seq[i] != "-":
                if antigen_residues[i].has_id(atom):
                    ag_xyz_matrix[i][3*j:3*j+3] = antigen_residues[i][atom].get_coord()

    return ag_xyz_matrix


def get_antibody_one_hot(heavy_seq, light_seq=None):
    """
    get antibody one hot from heavy and light chains sequences
    """
    if len(heavy_seq) > HEAVY_MAX_LENGTH:
        raise ValueError("Heavy chain is too long: {}, we support heavy chains of up to {} amino acids".format(len(heavy_seq), HEAVY_MAX_LENGTH))
    # pad the sequence with '-'
    heavy_padding = (HEAVY_MAX_LENGTH - (len(heavy_seq)))
    # turn in to one-hot encoding matrix
    seq_matrix = np.zeros((MAX_LENGTH + 1, FEATURE_NUM))
    for i in range(len(heavy_seq)):
        seq_matrix[i][AA_DICT[heavy_seq[i]]] = 1
        seq_matrix[i][21] = 1  # heavy column

    if light_seq is not None:
        if len(light_seq) > LIGHT_MAX_LENGTH:
            raise ValueError("Light chain is too long: {}, we support light chains of up to {} amino acids".format(len(light_seq), LIGHT_MAX_LENGTH))
        for i in range(len(light_seq)):
            seq_matrix[i+len(heavy_seq)+ heavy_padding][AA_DICT[light_seq[i]]] = 1
            seq_matrix[i+len(heavy_seq)+ heavy_padding ][22] = 1  # light column
        seq_matrix[MAX_LENGTH][23] = 1  # light chain transformation
    return seq_matrix


def get_antigen_one_hot_xyz(antigen_model, surface_executable=SURFACE, known_epitope=None):
    """
    get antigen one hot and xyz coordinates
    """
    antigen_seq_list, antigen_residues_list = get_seq_aa(antigen_model, only_ca=True)
    
    antigen_seq = "".join(antigen_seq_list)
    antigen_residues = []
    for chain_aa in antigen_residues_list:
        antigen_residues += chain_aa

    if len(antigen_seq) > MAX_LENGTH_ANTIGEN or len(antigen_seq) < MIN_LENGTH_ANTIGEN:
        raise ValueError(f"The Input antigen should have between 5 to 600 amino acids, the provided antigen has {len(antigen_seq)} amino acids.")
    
    if known_epitope is not None:
        assert len(known_epitope) == len(antigen_residues)

    antigen_surface = get_pdb_surface(antigen_model, surface_executable=surface_executable)
    if len(antigen_seq) != len(antigen_surface):
        raise ValueError(f"Antigen sequence length and surface don't match!: {len(antigen_seq)}, {len(antigen_surface)}")

    antigen_xyz = get_antigen_xyz(antigen_seq, antigen_residues)
    antigen_one_hot = get_antigen_one_hot(antigen_seq, antigen_surface)
    antigen_side_chain_xyz = TORSION_CALCULATOR.calculate_torsion(antigen_residues)

    patches_centers = []
    patches_xyz = []
    patches_side_chain_xyz = []
    patches_one_hot = []

    for aa in range(len(antigen_seq)):
        if antigen_surface[aa] > 0.0:  # the residue is in the surface

            # the residue is in the known epitope
            if known_epitope is not None and not known_epitope[aa]:
                continue

            # the residue has all of the required atoms for moving to the gloal reference frame
            if (antigen_residues[aa].has_id('N') and antigen_residues[aa].has_id('CA') and antigen_residues[aa].has_id('C')):

                patch_center = np.array([list(antigen_residues[aa][atom].get_coord()) for atom in ['N', 'CA', 'C']])

                patch_xyz = np.zeros((MAX_LENGTH_ANTIGEN, 15))
                patch_side_chain = np.zeros((MAX_LENGTH_ANTIGEN, 15))
                patch_one_hot = np.zeros((MAX_LENGTH_ANTIGEN + 1, ANTIGEN_FEATURE_NUM))

                patch_xyz[:len(antigen_seq),:] = np.array(antigen_xyz)
                patch_side_chain[:len(antigen_seq),:] = np.array(antigen_side_chain_xyz)
                patch_one_hot[:len(antigen_seq),:] = np.array(antigen_one_hot)

                patch_one_hot[MAX_LENGTH_ANTIGEN][ANTIGEN_FEATURE_NUM - 2] = 1  # docking column
                patch_one_hot[aa][ANTIGEN_FEATURE_NUM - 1] = 1  # interaction column

                patches_centers.append(patch_center)
                patches_xyz.append(patch_xyz)
                patches_side_chain_xyz.append(patch_side_chain)
                patches_one_hot.append(patch_one_hot)

    return antigen_seq, patches_one_hot, patches_xyz, patches_side_chain_xyz, patches_centers


def get_antigen_input(antigen_model, surface_executable=SURFACE, known_epitope=None):
    """
    get the antigen input for the docking network
    """

    if not antigen_model:  # only folding, no docking
        return None, np.zeros((1, MAX_LENGTH_ANTIGEN + 1, 15 + 15 + ANTIGEN_FEATURE_NUM))


    antigen_seq, patches_one_hot, patches_xyz, patches_side_chain_xyz, patches_centers = get_antigen_one_hot_xyz(antigen_model=antigen_model, surface_executable=surface_executable, known_epitope=known_epitope)
    patches_centered_xyz, patches_centered_sides = move_to_global_frame(patches_xyz, patches_side_chain_xyz, patches_centers)

    antigen_x = [np.concatenate((i, j, k), axis=-1) for i, j, k in zip(patches_one_hot, patches_centered_xyz, patches_centered_sides)]
    return antigen_seq, np.array(antigen_x)


def get_antibody_input(var_heavy_seq, var_light_seq):
    """
    get the antibody input for the docking network
    """
    antibody_x = get_antibody_one_hot(var_heavy_seq, var_light_seq)
    return antibody_x


def get_antibody_xyz_mask(var_heavy_seq, var_light_seq):
    """
    """
    mask = np.zeros((1, MAX_LENGTH, 30))
    cb_s = BACKBONE.index("CB") * 3

    for hl_seq, padding in zip([var_heavy_seq, var_light_seq], [0, HEAVY_MAX_LENGTH]):
        if hl_seq is None:
            continue
        for i, aa in enumerate(hl_seq):
            if aa == 'G':  # GLY aa missing CB atom
                mask[0, i+padding, 0:cb_s] = 1
            else:
                mask[0, i+padding, 0:cb_s+3] = 1
            three_letter_code = protein_letters_1to3[aa] if aa != "X" else "UNK"
            for j, chi in enumerate(CHIS_KEYS):
                if three_letter_code in CHIS[chi]:
                    chi_s = 15 + (j * 3)
                    mask[0, i+padding, chi_s:chi_s+3] = 1
    return mask


def convert_dock_input_to_score_input(var_heavy_seq, var_light_seq, ab_dock_input, at_dock_input, dock_output):
    """
    """
    antibody_xyz_mask = get_antibody_xyz_mask(var_heavy_seq, var_light_seq) 
    antigen_xyz_mask = at_dock_input[:, :-1, 24:] != 0.0

    pred_ab_ = dock_output["light_orientation"] * tf.cast(antibody_xyz_mask, dtype=tf.float32)
    pred_ag_ = dock_output["docking"] * tf.cast(antigen_xyz_mask, dtype=tf.float32)

    predicted_antibody = np.pad(pred_ab_,[[0,0],[0,1],[0,0]], mode='constant', constant_values=0.0)
    predicted_antigen = np.pad(pred_ag_,[[0,0],[0,1],[0,0]], mode='constant', constant_values=0.0)

    ab_score_input = np.concatenate([ab_dock_input, predicted_antibody], axis=-1)
    ag_score_input = np.concatenate([at_dock_input[:,:,:24], predicted_antigen], axis=-1)

    return ab_score_input, ag_score_input, pred_ab_, pred_ag_
