import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.patches import Circle
from PIL import Image
from torch import tensor

from imapep.data import ImmuneComplex

AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

AMINO_ACID_RADII = {
    "A": 2.3, "C": 2.5, "D": 2.8, "E": 3.05, "F": 3.4,
    "G": 1.9, "H": 3.1, "I": 3.09, "K": 3.15, "L": 3.15,
    "M": 3.1, "N": 2.85, "P": 2.8, "Q": 3.05, "R": 3.16,
    "S": 2.4, "T": 2.8, "V": 2.91, "W": 3.6, "Y": 3.45
}

# HB-donor,HB-acc,polarizability,charge,hydrophobicity,radius
AMINO_ACID_PROPERTIES = [
    [0.0,   0.0,    1.1,    6.0,    1.8,    2.3],  # A 
    [0.0,   0.0,    2.7,    5.07,   2.5,    2.5],  # C 
    [0.0,   1.0,    3.0,    2.77,   -3.5,   2.8],  # D
    [0.0,   1.0,    4.1,    3.22,   -3.5,   3.05], # E
    [0.0,   0.0,    8.0,    5.48,   2.8,    3.4],  # F
    [0.0,   0.0,    0.03,   5.97,   -0.4,   1.9],  # G
    [1.0,   1.0,    6.3,    7.95,   -3.2,   3.1],  # H
    [0.0,   0.0,    4.3,    6.02,   4.5,    3.09], # I
    [1.0,   1.0,    5.2,    9.74,   -3.9,   3.15], # K
    [0.0,   0.0,    4.2,    5.98,   3.8,    3.15], # L
    [0.0,   0.0,    5.1,    5.74,   1.9,    3.1],  # M
    [1.0,   1.0,    3.7,    5.41,   -3.5,   2.85], # N
    [0.0,   0.0,    4.3,    6.3,    -1.6,   2.8],  # P
    [1.0,   1.0,    4.8,    5.65,   -3.5,   3.05], # Q
    [1.0,   1.0,    8.5,    10.76,  -4.5,   3.16], # R
    [1.0,   1.0,    1.6,    5.68,   -0.8,   2.4],  # S
    [1.0,   1.0,    2.7,    5.6,    -0.7,   2.8],  # T
    [0.0,   0.0,    3.2,    5.96,   4.2,    2.91], # V
    [1.0,   1.0,    12.1,   5.89,   -0.9,   3.6],  # W
    [1.0,   1.0,    8.8,    5.66,   -1.2,   3.45]  # Y
]

RESIDUE_COLOR = {
    'A': np.array([0.088623046875, 0.404296875, 0.7001953125]), 
    'C': np.array([0.22119140625, 0.287841796875, 0.77783203125]), 
    'D': np.array([0.24609375, 0.0, 0.111083984375]), 
    'E': np.array([0.337158203125, 0.05633544921875, 0.111083984375]), 
    'F': np.array([0.66015625, 0.339111328125, 0.81103515625]), 
    'G': np.array([0.0, 0.400390625, 0.45556640625]), 
    'H': np.array([0.51953125, 0.6484375, 0.1444091796875]), 
    'I': np.array([0.353759765625, 0.40673828125, 1.0]), 
    'K': np.array([0.42822265625, 0.87255859375, 0.066650390625]), 
    'L': np.array([0.345458984375, 0.40185546875, 0.92236328125]), 
    'M': np.array([0.420166015625, 0.371826171875, 0.7109375]), 
    'N': np.array([0.303955078125, 0.330322265625, 0.111083984375]), 
    'P': np.array([0.353759765625, 0.44189453125, 0.322265625]), 
    'Q': np.array([0.395263671875, 0.3603515625, 0.111083984375]), 
    'R': np.array([0.70166015625, 1.0, 0.0]), 
    'S': np.array([0.130126953125, 0.3642578125, 0.4111328125]), 
    'T': np.array([0.22119140625, 0.354248046875, 0.422119140625]), 
    'V': np.array([0.2626953125, 0.399169921875, 0.966796875]), 
    'W': np.array([1.0, 0.390380859375, 0.39990234375]), 
    'Y': np.array([0.7265625, 0.36181640625, 0.36669921875])
}

# Feature2
# RESIDUE_COLOR = {
#     'A': np.array([0., 0.5, 0.7001953125]), 
#     'C': np.array([0., 0.5, 0.77783203125]), 
#     'D': np.array([0., 0., 0.111083984375]), 
#     'E': np.array([0., 0., 0.111083984375]), 
#     'F': np.array([1., 0.5, 0.81103515625]), 
#     'G': np.array([0., 0.5, 0.45556640625]), 
#     'H': np.array([0.5, 0.75, 0.1444091796875]), 
#     'I': np.array([0., 0.5, 1.0]), 
#     'K': np.array([0., 1., 0.066650390625]), 
#     'L': np.array([0., 0.5, 0.92236328125]), 
#     'M': np.array([0., 0.5, 0.7109375]), 
#     'N': np.array([0., 0.5, 0.111083984375]), 
#     'P': np.array([0., 0.5, 0.322265625]), 
#     'Q': np.array([0., 0.5, 0.111083984375]), 
#     'R': np.array([0., 1.0, 0.0]), 
#     'S': np.array([0., 0.5, 0.4111328125]), 
#     'T': np.array([0., 0.5, 0.422119140625]), 
#     'V': np.array([0., 0.5, 0.966796875]), 
#     'W': np.array([1., 0.5, 0.39990234375]), 
#     'Y': np.array([1., 0.5, 0.36669921875])
# }

# # Feature3
# RESIDUE_COLOR = {
#     'A': np.array([0., 0.5, 0.375]), 
#     'C': np.array([0., 0.5, 0.81818182]), 
#     'D': np.array([0., 0., 0.]), 
#     'E': np.array([0., 0., 0.14772727]), 
#     'F': np.array([1., 0.5, 0.85227273]), 
#     'G': np.array([0., 0.5, 0.35227273]), 
#     'H': np.array([0.5, 0.75, 0.40909091]), 
#     'I': np.array([0., 0.5, 0.89772727]), 
#     'K': np.array([0., 1., 0.]), 
#     'L': np.array([0., 0.5, 1.]), 
#     'M': np.array([0., 0.5, 0.82954545]), 
#     'N': np.array([0., 0.5, 0.29545455]), 
#     'P': np.array([0., 0.5, 0.10227273]), 
#     'Q': np.array([0., 0.5, 0.03409091]), 
#     'R': np.array([0., 1.0, 0.51136364]), 
#     'S': np.array([0., 0.5, 0.29545455]), 
#     'T': np.array([0., 0.5, 0.13636364]), 
#     'V': np.array([0., 0.5, 0.88636364]), 
#     'W': np.array([1., 0.5, 0.46590909]), 
#     'Y': np.array([1., 0.5, 0.71590909])
# }

# # Feature4 Eisenberg hydrophathy
# RESIDUE_COLOR = {
#     'A': np.array([0., 0.5, 0.8056266]), 
#     'C': np.array([0., 0.5, 0.72122762]), 
#     'D': np.array([0., 0., 0.4168798]), 
#     'E': np.array([0., 0., 0.45780051]), 
#     'F': np.array([1., 0.5, 0.95140665]), 
#     'G': np.array([0., 0.5, 0.76982097]), 
#     'H': np.array([0.5, 0.75, 0.54475703]), 
#     'I': np.array([0., 0.5, 1.]), 
#     'K': np.array([0., 1., 0.26342711]), 
#     'L': np.array([0., 0.5, 0.91815857]), 
#     'M': np.array([0., 0.5, 0.81074169]), 
#     'N': np.array([0., 0.5, 0.44757033]), 
#     'P': np.array([0., 0.5, 0.67774936]), 
#     'Q': np.array([0., 0.5, 0.42966752]), 
#     'R': np.array([0., 1.0, 0.]), 
#     'S': np.array([0., 0.5, 0.60102302]), 
#     'T': np.array([0., 0.5, 0.6342711]), 
#     'V': np.array([0., 0.5, 0.92327366]), 
#     'W': np.array([1., 0.5, 0.85421995]), 
#     'Y': np.array([1., 0.5, 0.71355499])
# }

# # Feature5 Janin hydrophathy
# RESIDUE_COLOR = {
#     'A': np.array([0., 0.5, 0.7777778]), 
#     'C': np.array([0., 0.5, 1.]), 
#     'D': np.array([0., 0., 0.44444444]), 
#     'E': np.array([0., 0., 0.40740741]), 
#     'F': np.array([1., 0.5, 0.85185185]), 
#     'G': np.array([0., 0.5, 0.7777778]), 
#     'H': np.array([0.5, 0.75, 0.62962963]), 
#     'I': np.array([0., 0.5, 0.92592593]), 
#     'K': np.array([0., 1., 0.]), 
#     'L': np.array([0., 0.5, 0.85185185]), 
#     'M': np.array([0., 0.5, 0.81481481]), 
#     'N': np.array([0., 0.5, 0.48148148]), 
#     'P': np.array([0., 0.5, 0.55555556]), 
#     'Q': np.array([0., 0.5, 0.40740741]), 
#     'R': np.array([0., 1.0, 0.14814815]), 
#     'S': np.array([0., 0.5, 0.62962963]), 
#     'T': np.array([0., 0.5, 0.59259259]), 
#     'V': np.array([0., 0.5, 0.88888889]), 
#     'W': np.array([1., 0.5, 0.77777778]), 
#     'Y': np.array([1., 0.5, 0.51851852])
# }

RESIDUE_COLOR_PN = {
    'A': np.array([1, 1, 1]),  
    'C': np.array([1, 1, 1]), 
    'D': np.array([0, 0, 1]), 
    'E': np.array([0, 0, 1]), 
    'F': np.array([0, 1, 0]), 
    'G': np.array([1, 1, 1]), 
    'H': np.array([1, 1, 1]), 
    'I': np.array([1, 1, 1]), 
    'K': np.array([1, 0, 0]), 
    'L': np.array([1, 1, 1]),
    'M': np.array([1, 1, 1]), 
    'N': np.array([1, 1, 1]), 
    'P': np.array([1, 1, 1]), 
    'Q': np.array([1, 1, 1]),  
    'R': np.array([1, 0, 0]), 
    'S': np.array([1, 1, 1]),
    'T': np.array([1, 1, 1]),
    'V': np.array([1, 1, 1]), 
    'W': np.array([0, 1, 0]), 
    'Y': np.array([0, 1, 0])
}


def get_color_code(resitypes, dists, coloring="chem"):
    assert len(resitypes) == len(dists), f"{len(resitypes)}-{len(dists)}"
    residues = [AMINO_ACIDS[resitype] for resitype in resitypes]
    
    if coloring == "chem":
        dists_scaled = torch.sigmoid(dists).tolist()
        colors = [RESIDUE_COLOR[residues[i]]*dists_scaled[i] for i in range(len(resitypes))]
    elif coloring == "chem_nd":
        colors = [RESIDUE_COLOR[residues[i]] for i in range(len(resitypes))]
    elif coloring == "4c":
        dists_scaled = torch.sigmoid(dists).tolist()
        colors = [RESIDUE_COLOR_PN[residues[i]]*dists_scaled[i] for i in range(len(resitypes))]

    return residues, colors


def get_zorder(dists):
    zorders = np.ones(dists.shape[0])
    for i, arg in enumerate(np.argsort(dists)):
        zorders[arg] = i
    return zorders


def get_image_framework(X):
    X_x, X_y = X[:,0].flatten().tolist(), X[:,1].flatten().tolist()  # [N],[N]
    xa = int(min(X_x)//10*10-5)
    xb = int((max(X_x)//10+1)*10+5)
    xs = xb if abs(xa) < xb else abs(xa)
    xtick_begin, xtick_end = -xs, xs
    ya = int(min(X_y)//10*10-5)
    yb = int((max(X_y)//10+1)*10+5)
    ys = yb if abs(ya) < yb else abs(ya)
    ytick_begin, ytick_end = -ys, ys
    scale_x = xtick_end - xtick_begin
    scale_y = ytick_end - ytick_begin
    figsize = scale_x//5, scale_y//5
    return (xtick_begin, xtick_end, ytick_begin, ytick_end), figsize


def get_img_metaparams_resi(metadict: dict, coloring="chem"):
    X = metadict["interface_resi_coords2d"]
    dists = metadict["interface_resi_dists"]
    resitypes = metadict["interface_resitypes"]

    if "antibody_chains" in metadict.keys() or "antigen_chains" in metadict.keys():
        chains_ab = metadict["antibody_chains"]
        chains_ag = metadict["antigen_chains"]

        ab_dists = tensor(dists[0])
        ab_resitypes = torch.cat([tensor(resitypes[chain], dtype=int) for chain in chains_ab]).tolist()
        ag_dists = tensor(dists[1])
        ag_resitypes = torch.cat([tensor(resitypes[chain], dtype=int) for chain in chains_ag]).tolist()

        X_ab = tensor(X[0])
        X_ag = tensor(X[1])
        # Initial image size (cropped afterwards)
        X = torch.cat([X_ab, X_ag], dim=0)
        lims, figsize = get_image_framework(X)

        # Determine the color
        ab_residues, ab_colors = get_color_code(ab_resitypes, ab_dists, coloring)
        ag_residues, ag_colors = get_color_code(ag_resitypes, ag_dists, coloring)

        # Determine the zorder
        ab_zorders = get_zorder(ab_dists)
        ag_zorders = get_zorder(ag_dists)
            
        return ({"residues": ab_residues,
                "coords": X_ab.numpy(), "colors": ab_colors, "zorders": ab_zorders, 
                "figsize": figsize, "lims": lims}, 
                {"residues": ag_residues,
                "coords": X_ag.numpy(), "colors": ag_colors, "zorders": ag_zorders, 
                "figsize": figsize, "lims": lims})
    else:
        X = tensor(X)
        chains = metadict["chains"]
        dists = tensor(dists)
        resitypes = torch.cat([tensor(resitypes[chain], dtype=int) for chain in chains]).tolist()

        lims, figsize = get_image_framework(X)
        residues, colors = get_color_code(resitypes, dists, coloring)
        zorders = get_zorder(dists)
        return {"residues": residues, "coords": X.numpy(), "colors": colors, 
                "zorders": zorders, "figsize": figsize, "lims": lims}
        

def get_img_metaparams_atom(metadict: dict, coloring="chem"):
    X = metadict["interface_atom_coords2d"]
    dists = metadict["interface_atom_dists"]
    resitypes = metadict["interface_atom_resitypes"]

    if "antibody_chains" in metadict and "antigen_chains" in metadict:
        chains_ab = metadict["antibody_chains"]
        chains_ag = metadict["antigen_chains"]

        ab_dists = tensor(dists[0])
        ab_resitypes = torch.cat([tensor(resitypes[chain], dtype=int) for chain in chains_ab]).tolist()
        ag_dists = tensor(dists[1])
        ag_resitypes = torch.cat([tensor(resitypes[chain], dtype=int) for chain in chains_ag]).tolist()

        X_ab = tensor(X[0])
        X_ag = tensor(X[1])
        # Initial image size (cropped afterwards)
        X = torch.cat([X_ab, X_ag], dim=0)
        lims, figsize = get_image_framework(X)

        # Determine the color
        _, ab_colors = get_color_code(ab_resitypes, ab_dists, coloring)
        _, ag_colors = get_color_code(ag_resitypes, ag_dists, coloring)

        # Determine the zorder
        ab_zorders = get_zorder(ab_dists)
        ag_zorders = get_zorder(ag_dists)
            
        return ({"coords": X_ab.numpy(), "colors": ab_colors, "zorders": ab_zorders, 
                "figsize": figsize, "lims": lims},
                {"coords": X_ag.numpy(), "colors": ag_colors, "zorders": ag_zorders, 
                "figsize": figsize, "lims": lims})

    else:
        lims, figsize = get_image_framework(X)
        _, colors = get_color_code(resitypes, dists, coloring)
        zorders = get_zorder(dists)
        return {"coords": X.numpy(), "colors": colors, 
                "zorders": zorders, "figsize": figsize, "lims": lims}


def get_img_metaparams(metadict, mode, coloring="chem"):
    if mode == "resi":
        return get_img_metaparams_resi(metadict, coloring)
    elif mode == "atom":
        return get_img_metaparams_atom(metadict, coloring)


def draw_interface_atom(
        coords, 
        colors, 
        zorders, 
        figsize, 
        lims, 
        dpi=20,
        img_size=(200, 200), 
        fc="black", 
        ec=None, 
        labels=None,
        label_fontsize=3,
        style="RGB"
):
    """Generate an image of the interface on which each residue is
    represented by a solid circle with radius related to the radius 
    of the residue type.

    `residues`, `coords`, `colors` and `zorders` must have the same
    length and the length should be equal to the number of residues
    on the interface.

    Args:
        residues (Sequence[str]): 
            One-letter codes of residues.
        coords (Sequence[Sequence[float]]): 
            Two-Dimensional coordinates of each residue in the image.
        zorders (Sequence[int]): 
            Same as `zorder` in `matplotlib`.
        figsize (Sequence[float]): 
            Same as `figsize` in `matplotlib`.
        dpi (int): 
            Same as `dpi` in `matplotlib`.
        lims (Sequence[float]): 
            Four-element array which is 
            (xlim_left, xlim_right, ylim_top, ylim_bottom).
        img_size (Sequence[float]): 
            Size of the finally output image: (height, width).
        fc (Union[Sequence[float], str], optional): 
            Same as `facecolor` in `matplotlib.Figure`. 
            Defaults to "black".
        labels (Union[Sequence[str], None], optional): 
            Labels put on patch circles. Defaults to None (no labels).

    Returns:
        Tensor: The drawn image.
    """
    assert len(coords) == len(colors) == len(zorders), \
        (f"Lengths of `coords`, `colors` and `zorders` not equal:"
         f" {len(coords)}-{len(colors)}-{len(zorders)}")
    
    plt.figure(figsize=figsize, dpi=dpi, facecolor=fc)
    plt.axis("equal")
    plt.axis("off")
    plt.xlim(lims[0], lims[1])
    plt.ylim(lims[2], lims[3])
    
    for i in range(len(coords)):
        x, y = coords[i][0], coords[i][1]
        color = colors[i]
        # print(color)
        circle = Circle(
            (x, y), 1, color=color, 
            zorder=zorders[i], 
            antialiased=False, ec=ec
        )
        plt.gca().add_patch(circle)
        if labels is not None:
            plt.text(
                x, y, labels[i], 
                zorder=len(zorders)+zorders[i], 
                size=label_fontsize, 
                c="white"
            )
    
    with io.BytesIO() as io_buf:
        plt.savefig(io_buf, format="png")
        io_buf.seek(0)
        tsr = F.pil_to_tensor(Image.open(io_buf).convert(style))
    
    plt.close("all")
    
    tsr = F.center_crop(tsr, img_size)
    
    return tsr


def draw_interface_resi(
        residues,
        coords, 
        colors, 
        zorders, 
        figsize, 
        lims, 
        dpi=20,
        img_size=(200, 200), 
        fc="black", 
        ec=None, 
        labels=None,
        label_fontsize=20,
        style="RGB"
):
    """Generate an image of the interface on which each residue is
    represented by a solid circle with radius related to the radius 
    of the residue type.

    `residues`, `coords`, `colors` and `zorders` must have the same
    length and the length should be equal to the number of residues
    on the interface.

    Args:
        residues (Sequence[str]): 
            One-letter codes of residues.
        coords (Sequence[Sequence[float]]): 
            Two-Dimensional coordinates of each residue in the image.
        zorders (Sequence[int]): 
            Same as `zorder` in `matplotlib`.
        figsize (Sequence[float]): 
            Same as `figsize` in `matplotlib`.
        dpi (int): 
            Same as `dpi` in `matplotlib`.
        lims (Sequence[float]): 
            Four-element array which is 
            (xlim_left, xlim_right, ylim_top, ylim_bottom).
        img_size (Sequence[float]): 
            Size of the finally output image: (height, width).
        fc (Union[Sequence[float], str], optional): 
            Same as `facecolor` in `matplotlib.Figure`. 
            Defaults to "black".
        labels (Union[Sequence[str], None], optional): 
            Labels put on patch circles. Defaults to None (no labels).

    Returns:
        Tensor: The drawn image.
    """
    assert len(residues) == len(coords) == len(colors) == len(zorders), \
        (f"Lengths of `residues`, `coords`, `colors` and `zorders` not equal:"
         f" {len(residues)}-{len(coords)}-{len(colors)}-{len(zorders)}")
    
    plt.figure(figsize=figsize, dpi=dpi, facecolor=fc)
    plt.axis("equal")
    plt.axis("off")
    plt.xlim(lims[0], lims[1])
    plt.ylim(lims[2], lims[3])
    
    for i in range(len(residues)):
        x, y = coords[i][0], coords[i][1]
        residue = residues[i]
        radius = AMINO_ACID_RADII[residue]
        color = colors[i]
        circle = Circle(
            (x, y), radius, 
            color=color, 
            zorder=zorders[i], 
            antialiased=False, 
            ec=ec
        )
        plt.gca().add_patch(circle)
        if labels is not None:
            if len(labels[i]) > 1:
                x = x - 1.5
            plt.text(
                x, y, labels[i], 
                zorder=len(zorders)+zorders[i], 
                size=label_fontsize, 
                c="white"
            )
    
    with io.BytesIO() as io_buf:
        plt.savefig(io_buf, format="png")
        io_buf.seek(0)
        tsr = F.pil_to_tensor(Image.open(io_buf).convert(style))
    
    plt.close("all")
    
    if len(img_size) == 2:
        tsr = F.center_crop(tsr, img_size)
    elif len(img_size) == 4:
        tsr = F.crop(tsr, *img_size)
    
    return tsr