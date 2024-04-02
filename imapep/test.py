r"""
To run the script:

1. Install miniconda and create a virtual environment with Python 3.9. 
Install the following packages:
    - pytorch==1.13.0
    - torchvision==0.14.0
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
    - tqdm
    - biopython
    - abnumber
PyTorch and TorchVision can be installed using:
```
    conda install pytorch==1.13.0 torchvision==0.14.0 cpuonly -c pytorch
```
Abnumber can be installed using:
```
    conda install abnumber -c bioconda
```

2. Put the imapep package into the environment variables, by adding 
the following line to your .bashrc file. Suppose you put the 
`imapep` folder (the one directly containing the .py files) under 
a driectory named `/home/3bio`:
```
    export PYTHONPATH=$PYTHONPATH:/home/3bio
``` 

3. All the code used for test is in test.py, i.e. the current file.
The snippet of "Generate image" should generate two images for the
paratope and epitope. Respectively use APIs ended with "residue/atom" 
to generate per-residue/per-atom images.

4. The snippet of "Score with existing model"
should arrive at a score of 0.7836 for per-residue mode.

Note: Only use the provided PDB under `imapep` directory. This is a PDB
without missing atoms (processed using PDBFixer). Using a PDB downloaded 
from RCSB PDB may bring an error.
"""

from pathlib import Path

from imapep.data import parse_pdb
from imapep.imgutils import *
from imapep.ml import *
from torchvision.io import write_png

# Generate image
# Change the 2nd param to the path on your machine
complex_ = parse_pdb("1a14_HL_N", "/data1/dongli/ImaPEp/ImaPEp/data/test/1a14_HL_N.pdb", "HLN")
metadict = complex_.write_json(None)
kws_para, kws_epi = get_img_metaparams_resi(metadict)
img_para = draw_interface_resi(**kws_para).to(torch.uint8)
img_epi = draw_interface_resi(**kws_epi).to(torch.uint8)
# Save the generated images
write_png(img_para, "/data1/dongli/ImaPEp/ImaPEp/data/image/1a14_HL_N_para.png")
write_png(img_epi, "/data1/dongli/ImaPEp/ImaPEp/data/image/1a14_HL_N_epi.png")


# Score with existing model
cfg = {"conv_in": 6, "conv_out": 32, "fc_dim": 25*25*32*2, "dropout": 0.75}
model = CNN(**cfg).eval()
model_dir = Path("/data1/dongli/ImaPEp/ImaPEp/data/model/models_100x100")  # Change the folder of saved models
data = torch.cat([
    torch.div(img_para[:,50:-50,50:-50], 255), 
    torch.div(img_epi[:,50:-50,50:-50], 255)
]).unsqueeze(0)
# print(data.shape)
score = get_result_on_existing_model(model, data, list(model_dir.iterdir()), "cpu")
print(score)