# Fold&Dock

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dina-lab3D/Fold-Dock/blob/main/Fold_Dock.ipynb)

<p align="center"><img src="https://github.com/dina-lab3D/Fold-Dock/blob/main/Images/FoldDock_architecture.png" width="900" /></p>

**Fold&Dock** - a rapid modeling tool for antibody-antigen and nanobody-antigen complexes. 

for citations, please cite our paper: [End to end accurate and high throughput modeling of antibody antigen complexes](https://www.mlsb.io/papers_2022/End_to_end_accurate_and_high_throughput_modeling_of_antibody_antigen_complexes.pdf)



## How to run Fold&Dock from google Colaboratory:

    1. Open the Colab notebook (Fold_Dock.ipynb, link above).
    
    2. Input antibody sequence
    
        - Select antibody input type (a fasta file or an antibody sequence). 
          Separate the light and heavy chains with ':' in your input sequence.
          To model multiple antibody sequences for a given antigen in a single run upload a fasta file with multiple antibody sequences.
        - Run this cell to upload your fasta file (if chosen this option).
          
    3. Input antigen structure
    
        - If you want to preform docking to a given antigen structure in addition to antibody folding, select the option 'do_docking'.
        - If you want to preform docking only for specific chains of the PDB file, specify them in the format 'ABC' for chains A,B,C.
        - Run this cell to upload your antigen pdb file (if chosen the option 'do_docking')
        
    4. Advanced settings
    
        - Select the number of best scoring complexes to create PDB files for (with an antigen this value can be between 0-len(antigen), 
          without an antigen this value can be either 0 or 1)
        - You have the option to relax the structures and reconstruct the side chains using MODELLER. 
          To do so you need a licese key which  can be obtained from here: https://salilab.org/modeller/
        - You have the option to visualize the best scoring model and select the verbose of the program.
        
    5. Saving options
    
        - You can select the output directory and whether or not you want to save the results to your drive.
        
    6. Run the other cells without changes.


## How to run Fold&Dock locally:

    1. Clone the git repository : git clone "https://github.com/dina-lab3D/Fold-Dock"
    2. Make sure you have the following libraries installed in your environment:
    
            - timeit
            - logging
            - argparse
            - pandas
            - subprocess
            - scipy
            - numpy
            - abnumber
            - tensorflow (2.4.0 or higher)
            - Bio (1.8.0 or higher)
            - modeller (optional, only if you want to reconstruct the side chains using modeller, requires license - https://salilab.org/modeller/)

    3. Run the following command (with python 3):

            python fold_dock.py <antibody fasta file path>

            options:

                    -a <antigen_pdb>: pdb file with the antigen structure for docking.
                    -c <antigen_chains>: which antigen chains to consider for docking, for example ABC, (default: All chains in the given antigen_pdb file)
                    -o <output directory> : path to a directory to put the generated models in, default is './Results'
                    -m : run side chains reconstruction using modeller on the structures, default is False. 
                    -t <top_n>: number of models to generate for each antibody sequence (0-len(antigen)), default is 5.
                    -v <verbose>: whether or not to print the program progress, default is 1 (print). for a quiet run use -v 0.
                    

## Approximate running times (creating PDB files top 5 models): 

### single antibody sequence (minutes)

|               | without modeller | with modeller  |
| ------------- |:----------------:| :--------------:|
| CPU           |      3-5         |      8-10      |
| GPU           |      1-2         |      3-7       |

### 100 antibody sequences (hours): 

|               | without modeller | with modeller  |
| ------------- |:----------------:| :--------------:|
| CPU           | 3                |      12         |
| GPU           | 0.33-0.5         |      7          |


<p align="center"><img src="https://github.com/dina-lab3D/Fold-Dock/blob/main/Images/FoldDock_movie.gif" width="700" /></p>


