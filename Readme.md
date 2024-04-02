## Dependencies

Our model has been tested on Linux using the following Python packages: 
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
To train ImaPEp from scratch, one can execute:
```
python3 train.py --train-dir=train_resi --model-dir=models
```
After the process, ten cross validated models are trained with the dataset in `train_resi` and their parameters are saved under directory `model_dir`.

Prediction of binding affinity between a given paratope and epitope can be performed in two ways: 

- Single-file mode. The user provides the PDB structure of an antibody-antigen complex. For example,
    ```
    python3 predict.py --sample-id=1a2y --input=1a2y.pdb --chains=BAC
    ```
- Batch mode. There is also an option for scoring a large amount of inputted structures in one click. The user is required to assign a folder to the `--input` option which has to contain all the inputted PDB files and a file named job.txt. For each Ab-Ag complex to be scored, one line in the format "pdb_id,chains" is included in job.txt file. For example:
    ```
    python3 predict.py --input=/job/folder
    ```
The `chains` option is used to specify the protein chains of interest in which the first character must be the identifier (ID) of the antibody heavy chain, followed by the antibody light chain and antigen chains. For example, in "HLAB" the "H" and "L" represent antibody heavy and light chain's IDs and the "A" and "B" correspond to two antigen chains.

## Advice
Note that in principle, the antigen should be a protein containing at least 50 residues, and the paratope and epitope should contain no fewer than three residues individually. All input PDB files must contain no residues with missing atoms. Consider deleting the problematic residues or adding the missing atoms (more recommended). Though with low probability, sometimes the CDRs cannot be successfully parsed, in which case the scoring procedure will be terminated.