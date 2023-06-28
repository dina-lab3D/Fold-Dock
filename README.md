# Fold&Dock

<p align="center"><img src="https://drive.google.com/uc?id=1FUTKK5IZPNxNvi-aHA5vcQ0Pe_ba4B6h" width="1000" /></p>

<p align="center"><img src="https://drive.google.com/uc?id=1xWMqarIhJb2IbBMqOnuS72a0a7eUgbw9" width="1000" />
<p align="center"><img src="http://i.stack.imgur.com/SBv4T.gif" alt="this slowpoke moves"  width="250" />


Fold&Dock - a rapid modeling tool for antibody-antigen and nanobody-antigen complexes. 

for citations, please cite our paper: [End to end accurate and high throughput modeling of antibody antigen complexes](https://www.mlsb.io/papers_2022/End_to_end_accurate_and_high_throughput_modeling_of_antibody_antigen_complexes.pdf)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dina-lab3D/Fold-Dock/blob/main/Fold_Dock.ipynb)

How to run Fold&Dock from google Colaboratory:

    1. Open the Colab notebook (Fold_Dock.ipynb, link above).
    2. Select protein type (Nb/mAb heavy chain or TCR VB).
    3. Select input type (sequence (String) or path to a fasta file)
    4. Provide a Nb sequence/fasta (NanoNet will preduce a model for each entry in the fasta file).
    5. Select whether or not you want to reconstruct the side chains using modeller (requires license - https://salilab.org/modeller/).
    6. Press the 'Run all' option.

How to run Fold&Dock locally:

    1. Clone the git repository : git clone "https://github.com/dina-lab3D/NanoNet"
    2. Make sure you have the following libraries installed in your environment:

            - numpy
            - tensorflow (2.4.0 or higher)
            - Bio (1.8.0 or higher)
            - modeller (optional, only if you want to reconstruct the side chains using modeller, requires license - https://salilab.org/modeller/)

    3. Run the following command (with python 3):

            python NanoNet.py <fasta file path>

            this will produce a backbone + cb pdb named '<record name>_nanonet_backbone_cb.pdb' for each record in the fasta file.

            options:

                    -s : write all the models into a single PDB file, separated with MODEL and ENDMDL (reduces running time when predicting many structures), default is False.
                    -o <output directory> : path to a directory to put the generated models in, default is './NanoNetResults'
                    -m : run side chains reconstruction using modeller, default is False. Output it to a pdb file named '<record name>_nanonet_full_relaxed.pdb'
                    -c <path to Scwrl4 executable>: run side chains reconstruction using scwrl, default is False. Output it to a pdb file named '<record name>_nanonet_full.pdb'
                    -t : use this parameter for TCR V-beta modeling, default is False

Running times for 1,000 structures on a single standard CPU: 

only backbone + Cb - less than 15 seconds (For better preformance use GPU and cuda).
backbone + SCWRL - about 20 minutes. 
backbone + Modeller - about 80 minutes.
