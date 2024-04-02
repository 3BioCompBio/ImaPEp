import os
import sys
import time

import click
from imapep.data import parse_pdb
from imapep.imgutils import *
from imapep.ml import *


def _pdb_to_tensor(sample_id, input, chains, channel_mode, shape=100):
    input_ = str(input)
    complex_ = parse_pdb(sample_id, input_, chains)
    if complex_ is None:
        return None
    else:
        kw_para, kw_epi = get_img_metaparams(complex_.write_json(None), "resi")  # No json is output
        img_para = draw_interface_resi(**kw_para, img_size=(shape,shape)) / 255
        img_epi = draw_interface_resi(**kw_epi, img_size=(shape,shape)) / 255
    
    if channel_mode != "OOO":
        img_channels_para = []
        img_channels_epi = []
        for i in range(len(channel_mode)):
            x = channel_mode[i]
            if x == "O":
                img_channel_para = img_para[[i],...]
                img_channel_epi = img_epi[[i],...]
            elif x == "X":
                img_channel_para = torch.zeros_like(img_para[[i],...])
                img_channel_epi = torch.zeros_like(img_epi[[i],...])
            img_channels_para.append(img_channel_para)
            img_channels_epi.append(img_channel_epi)
        img_para = torch.cat(img_channels_para)
        img_epi = torch.cat(img_channels_epi)
    
    data = torch.cat([img_para, img_epi]).to(torch.float32)
    return data.unsqueeze(0)


@click.command()
@click.option("--sample-id", default="xxxx", help="Name of the input")
@click.option("--input", "-i", required=True, help="Path to a PDB file")
@click.option("--chains", help="Ab heavy chain, Ab light chain and antigen chain(s)")
@click.option("--channel-mode", default="OOO", required=True)
@click.option("--shape", default=100)
@click.option("--model-dir", help="Directory that exclusively contains pickled model parameters")
@click.option("--device", default="cuda:0")
def predict(
        sample_id, 
        input, 
        chains, 
        channel_mode,
        shape, 
        model_dir, 
        device
):
    input_ = Path(input)
    # Prepare data for prediction
    if input_.is_dir():
        tbl = pd.read_csv(input_/"job.txt", header=None)
        all_ids = tbl.iloc[:,0].to_list()
        id_to_score = {id_: None for id_ in all_ids}
        valid_ids = []  # PDBs that ImaPEp can score, others to be scored as 0
        all_data = []
        for i in tqdm.tqdm(list(tbl.index)):
            sample_id = tbl.iloc[i,0]
            pdb_file = str(input_/f"{sample_id}.pdb")
            chains = tbl.iloc[i,1]
            data = _pdb_to_tensor(sample_id, pdb_file, chains, channel_mode, shape)
            if data is None:
                click.echo(f"{sample_id}: PDB Parsing Error")
                continue
            else:
                valid_ids.append(sample_id)
            all_data.append(data)
        data = torch.cat(all_data)
    elif input_.is_file():
        data = _pdb_to_tensor(
            sample_id, 
            input_, 
            chains, 
            channel_mode, 
            shape
        )
        if data is None:
            click.echo(f"{sample_id}: PDB Parsing Error")
            return
    else:
        click.echo("Input error")
        return

    # Prediction
    model_args = {
        "conv_in": 6, 
        "conv_out": 32, 
        "fc_dim": 2*32*(shape//4)**2, 
        "dropout": 0.75, 
        "bias": True
    }
    model = CNN(**model_args).to(device).eval()
    if model_dir is not None:
        model_files = list(Path(model_dir).iterdir())
    else:
        if shape == 100:
            model_files = list(Path(f"{os.path.dirname(__file__)}/data/model/models_100x100").iterdir())
        elif shape == 64:
            model_files = list()
        else:
            click.echo("Should specify a directory containing model parameter files")

    # To avoid a GPU memory overflow, use a batch of 128 samples each time
    all_scores = []
    num_samples = data.shape[0]
    slicing_indices = (np.arange(num_samples//128+1)*128).tolist() + [num_samples]
    for idx1, idx2 in zip(slicing_indices[:-1], slicing_indices[1:]):
        print(idx1, idx2)
        sliced = data[idx1:idx2, ...]
        all_scores.append(get_result_on_existing_model(model, sliced, model_files, device))  # [B] np.array
        del sliced
    scores = np.concatenate(all_scores)
    
    if scores.shape[0] == 1:  # single mode
        output = f"{sample_id},{round(scores[0], 4)}"
    elif scores.shape[0] > 1:
        assert len(valid_ids) == scores.shape[0]
        valid_id_to_score = {valid_ids[i]: round(scores[i], 4) for i in range(len(scores))}
        id_to_score.update(valid_id_to_score)
        output = "\n".join([f"{id_},{score}" for id_, score in id_to_score.items()])
    
    with open(f"{os.getcwd()}/scores_{int(time.time())}.txt", "w") as fp:
        fp.write(output)


if __name__ == "__main__":
    predict()