# ImaPEp: Antibody-Antigen Binding Probability Prediction
ImaPEp is a computational tool for predicting the binding probability between antibody paratopes and antigen epitopes. This repository provides the necessary scripts for training models and performing predictions on antibody-antigen complexes using structural data.

## Dependencies

ImaPEp has been tested on Linux with the following Python packages:
- PyTorch == 1.13.0
- TorchVision == 0.14.0
- Numpy == 1.25.2
- abnumber
- tqdm
- matplotlib == 3.7.2
- Biopython == 1.78
- Scikit-learn == 1.3.0
- Pillow == 10.0.1

## Usage
To use ImaPEp, first clone the repository and update your environment variables:

```bash
cd /your/path/
git clone https://github.com/3BioCompBio/ImaPEp.git 
export PYTHONPATH="/your/path/ImaPEp/:$PYTHONPATH"
```

To make the changes permanent, add the export command to your ~/.bashrc file.

### Training a Model from Scratch
To train ImaPEp from scratch using your dataset, run the following command:

```bash
python3 train.py --train-dir=train_resi --model-dir=models
```

This will train ten cross-validated models using the dataset in `train_resi` and save the trained model parameters in the `models` directory.

### Predicting Antibody-Antigen Binding
ImaPEp supports two modes for prediction: 

1\. Single-File Mode

For predicting the binding affinity of a single antibody-antigen complex, provide a PDB structure as input:

```bash
python3 predict.py --sample-id=my_pdb --input=my_pdb.pdb --chains=HLBC
```

Here, the `--chains` argument specifies the relevant protein chains in the PDB file:

* The first character represents the antibody heavy chain ID (e.g., `H`).

* The second character represents the antibody light chain ID (e.g., `L`).

* The remaining characters correspond to the antigen chains (e.g., `BC`).

2\. Batch mode

For processing multiple antibody-antigen complexes in one go, place all input PDB files into a designated folder and create a job.txt file containing the details for each complex (one per line in the format `pdb_id,chains`). Then, execute:

```bash
python3 predict.py --input=/job/folder/
```

The `chains` column in the `job.txt` file follows the same format as in Single-File Mode, specifying the heavy chain, light chain, and antigen chains.

## Notes
* The antigen should be a protein with at least 50 residues.

* The paratope and epitope should each contain at least three residues.

* All input PDB files must have no missing atoms. If residues with missing atoms exist, consider removing them or adding the missing atoms (recommended).

* In rare cases, the CDR regions may not be successfully parsed, causing the prediction to terminate.
