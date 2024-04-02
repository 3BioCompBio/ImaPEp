# -*-coding:utf-8-*-
import warnings

warnings.filterwarnings("ignore")
import json
from typing import *

import abnumber
import torch
from Bio import PDB
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from imapep.utils import *
from torch import tensor

SIDECHAIN_ATOMS = {
    "Ala": ("CB",),
    "Cys": ("CB", "SG"),
    "Asp": ("CB", "CG", "OD1", "OD2"),
    "Glu": ("CB", "CG", "CD", "OE1", "OE2"),
    "Phe": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "Gly": (),
    "His": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "Ile": ("CB", "CG1", "CG2", "CD1"),
    "Lys": ("CB", "CG", "CD", "CE", "NZ"),
    "Leu": ("CB", "CG", "CD1", "CD2"),
    "Met": ("CB", "CG", "SD", "CE"),
    "Asn": ("CB", "CG", "OD1", "ND2"),
    "Pro": ("CB", "CG", "CD"),
    "Gln": ("CB", "CG", "CD", "OE1", "NE2"),
    "Arg": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "Ser": ("CB", "OG"),
    "Thr": ("CB", "OG1", "CG2"),
    "Val": ("CB", "CG1", "CG2"),
    "Trp": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "Tyr": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")
}

RESIDUE_FLAGS = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, 
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}


class Protein(object):
    
    def __init__(self, id_, seqs, chains, atom_coords, atom_CA_coords, atom_CMR_coords, atom_mask, atom_types):
        self.id = id_
        self.seqs = seqs
        self.chains = chains
        self.atom_coords = atom_coords
        self.atom_CA_coords = atom_CA_coords
        self.atom_CMR_coords = atom_CMR_coords
        self.atom_mask = atom_mask
        self.atom_types = atom_types

    def get_interface(self, interf: str):
        r"""For a single protein, the interface can only be specified
        artificially.

        Args:
            interf (str): comma-separated string like "A12,B45"
        """
        chains = self.chains
        
        interface_strings = interf.split(",")
        _interface = {c:[] for c in chains}
        for interf_str in interface_strings:
            _interface[interf_str[0]].append(int(interf_str[1:]))
        for c in _interface:
            _interface[c] = tensor(_interface[c])
        self.interface_resi_indices = _interface

        interf_idx_atom = {c: tensor([], dtype=torch.int32) for c in chains}
        for c in chains:
            idx_resi = _interface[c]
            idx_atom = torch.isin(self.atom_mask[c], idx_resi).int().nonzero().squeeze(-1)
            interf_idx_atom[c] = idx_atom
        self.interface_atom_indices = interf_idx_atom

    def generate_patch(self, mode="atom"):
        r"""
        Run PCA to reduce the 3-D plane into the 2-D one.
        """
        seqs = self.seqs
        
        if mode == "atom":
            interface_indices = self.interface_atom_indices
            coords = self.atom_coords
            atom_masks = self.atom_mask
        elif mode == "resi":
            interface_indices = self.interface_resi_indices
            coords = self.atom_CMR_coords

        interface_coords = {}
        for c in seqs:
            idx = interface_indices[c]
            if idx.nelement():
                X = torch.index_select(coords[c], 0, idx)
            else:
                X = tensor([])
            interface_coords[c] = X

        X = torch.cat([interface_coords[c] for c in seqs])
        _, X_patch, normal = run_pca(X)

        if mode == "atom":
            self.interface_atom_coords2d = X_patch
        elif mode == "resi":
            self.interface_resi_coords2d = X_patch
        
        X_ctr = X.mean(dim=0)
        D_plane = -torch.matmul(X_ctr, normal).unsqueeze(0)
        plane = torch.cat([normal, D_plane])
        self.pca_plane = plane

        interface_resitypes = {}
        for c in seqs:
            idx = interface_indices[c]
            X = interface_coords[c]  # [N,3]
            seq = seqs[c]
            if mode == "atom":
                atom_mask = atom_masks[c]
                # if X.shape[0]:
                resitypes = tensor([RESIDUE_FLAGS[seq[i]] for i in atom_mask])
                resitypes = torch.index_select(resitypes, 0, idx)
                interface_resitypes[c] = resitypes  # [N_sc,3]
                self.interface_atom_resitypes = interface_resitypes
            elif mode == "resi":
                resitypes = tensor([RESIDUE_FLAGS[seq[i]] for i in idx])
                interface_resitypes[c] = resitypes
                self.interface_resitypes = interface_resitypes
    
    def distance_to_pca(self, mode="atom"):
        if not hasattr(self, "pca_plane"):
            raise AttributeError(("Attribute 'pca_plane' not found. Did you run `prot.generate_patch`?"))

        seqs = self.seqs
        pca_plane = self.pca_plane
        if mode == "resi":
            coords = self.atom_CMR_coords
            interface_indices = self.interface_resi_indices
        elif mode == "atom":
            coords = self.atom_coords
            interface_indices = self.interface_atom_indices
        
        interface_coords = {}
        for c in seqs:
            idx = interface_indices[c]
            if idx.nelement():
                X_sc = torch.index_select(coords[c], 0, idx)
            else:
                X_sc = tensor([])
            interface_coords[c] = X_sc
        X_interf = torch.cat([interface_coords[c] for c in interface_coords])

        # Collect coordinates of non-interface residues ("frame")
        all_X_frame = []
        for chain in seqs:
            X_sc = coords[chain]  # [N_sc,3]
            interface_idx = interface_indices[chain]  # [N_sc_inter]
            frame_idx = list(set(range(X_sc.size(0))).difference(set(interface_idx)))  # [N_sc_frame]
            X_frame = torch.index_select(X_sc, 0, tensor(frame_idx))  # [N_sc_frame,3]
            all_X_frame.append(X_frame)
        X_frame = torch.cat(all_X_frame)  # [N_frame,3]

        cylinder1, cylinder2 = build_patch_cylinder(X_interf)
        num_points_in_cyl1 = torch.count_nonzero(cylinder1.is_inside(X_frame)).item()
        num_points_in_cyl2 = torch.count_nonzero(cylinder2.is_inside(X_frame)).item()
        distances = distance_to_plane(X_interf, pca_plane)  # [N_para]
        # This means cylinder1 == negative intercept == interior
        if num_points_in_cyl1 > num_points_in_cyl2:
            ones = torch.ones((X_interf.size(0), 1))  # [N_para,1]
            X_expand = torch.cat([X_interf, ones], dim=1)  # [N_para,4]
            intercepts = torch.matmul(X_expand, pca_plane).squeeze()
            dist_signs = (intercepts > 0).int() * 2 - 1
            distances *= dist_signs
        # This means cylinder2 == positive intercept == interior
        else:
            ones = torch.ones((X_interf.size(0), 1))  # [N_para,1]
            X_expand = torch.cat([X_interf, ones], dim=1)  # [N_para,4]
            intercepts = torch.matmul(X_expand, pca_plane).squeeze()
            dist_signs = (intercepts < 0).int() * 2 - 1
            distances *= dist_signs
        
        if mode == "atom":
            self.interface_atom_dists = distances
        elif mode == "resi":
            self.interface_resi_dists = distances

    def write_json(self, fname):
        metadict = {}
        
        metadict["id"] = self.id
        metadict["seqs"] = self.seqs
        metadict["chains"] = self.chains
        metadict["atom_CA_coords"] = self.atom_CA_coords
        metadict["atom_coords"] = self.atom_coords
        metadict["atom_CMR_coords"] = self.atom_CMR_coords
        metadict["atom_mask"] = self.atom_mask
        metadict["atom_types"] = self.atom_types
        metadict["interface_resi_indices"] = self.interface_resi_indices
        metadict["interface_atom_indices"] = self.interface_atom_indices
        for attr in [
            "interface_resi_coords2d", "interface_resi_dists", "interface_resitypes", 
            "interface_atom_coords2d", "interface_atom_dists", "interface_atom_resitypes"
        ]:
            if hasattr(self, attr):
                metadict[attr] = getattr(self, attr)
        
        metadict = tensor_to_list(metadict)
        if fname is not None:
            with open(fname, "w") as fp:
                json.dump(metadict, fp)
        return metadict

    def load_json(self, metadict):
        self.id = metadict["id"]
        self.seqs = metadict["seqs"]
        # self.antibody_chains = metadict["antibody_chains"]
        # self.antigen_chains = metadict["antigen_chains"]
        # self.cdr_masks = tensor(metadict["cdr_masks"][0]), tensor(metadict["cdr_masks"][1])
        # self.segments = metadict["antibody_segments"]
        # self.cdr_atom_coords = tensor(metadict["CDR_atom_coords"])
        self.atom_CA_coords = {c: tensor(xyz) for c, xyz in metadict["atom_CA_coords"].items()}
        self.atom_coords = {c: tensor(xyz) for c, xyz in metadict["atom_coords"].items()}
        self.atom_mask = {c: tensor(mask) for c, mask in metadict["atom_mask"].items()}
        self.atom_types = metadict["atom_types"]
        self.interface_resi_indices = {c: tensor(idx, dtype=torch.int32) for c, idx in metadict["interface_resi_indices"].items()}
        self.interface_atom_indices = {c: tensor(idx, dtype=torch.int32) for c, idx in metadict["interface_atom_indices"].items()}
        
        attrs = [
            "interface_atom_coords2d", "interface_atom_dists", "interface_atom_resitypes",
            "interface_resi_coords2d", "interface_resi_dists", "interface_resitypes"
        ]
        for attr in attrs:
            if attr in metadict:
                setattr(self, attr, metadict[attr])


class ImmuneComplex(Protein):
    """ Class of antibody-antigen complex structure. """

    def __init__(self, id_, seqs, antibody_chains, antigen_chains, atom_coords, atom_CA_coords, atom_CMR_coords, atom_mask, atom_types):
        assert (set(antibody_chains+antigen_chains).issubset(set(seqs.keys()))), (f"Chain ID not matched: "
             f"{antibody_chains + antigen_chains} vs. {list(seqs.keys())}")
        super().__init__(id_, seqs, antibody_chains+antigen_chains, atom_coords, atom_CA_coords, atom_CMR_coords, atom_mask, atom_types)
        # In a Ab-Ag complex, one should make clear the chains for Ab and Ag
        self.antibody_chains = antibody_chains
        self.antigen_chains = antigen_chains

    @property
    def antigen_length(self):
        """ Get lengths of all antigen chains. Alaways returns a list. """
        lengths_antigen = []
        
        for chain in self.antigen_chains:
            lengths_antigen.append(len(self.seqs[chain]))
        
        return lengths_antigen
    
    @property
    def interface_sizes(self):
        """ Get lengths of paratope and epitope. """
        num_paratope_residue = num_epitope_residue = None
        chains_ab, chains_ag = self.antibody_chains, self.antigen_chains
        
        if chains_ab:
            chain_H, chain_L = chains_ab[0], chains_ab[1]
            paratope_indices_Hchain = self.interface_resi_indices[chain_H]
            paratope_indices_Lchain = self.interface_resi_indices[chain_L]
            num_paratope_residue = len(paratope_indices_Hchain)+len(paratope_indices_Lchain)
        if chains_ag:
            num_epitope_residue = len(sum([self.interface_resi_indices[c].tolist() for c in chains_ag], []))
        
        return num_paratope_residue, num_epitope_residue

    @property    
    def cdr_residues(self):
        segments_H, segments_L = self.segments[0], self.segments[1]
        cdrH1, cdrH2, cdrH3 = segments_H[1], segments_H[3], segments_H[5]
        cdrL1, cdrL2, cdrL3 = segments_L[1], segments_L[3], segments_L[5]
        all_cdr = cdrH1 + cdrH2 + cdrH3 + cdrL1 + cdrL2 + cdrL3
        return list(all_cdr)

    def initialize(self):
        cdrH_mask, cdrL_mask, segments = self.parse_cdr()
        self.cdr_masks = cdrH_mask, cdrL_mask
        self.segments = segments[:7], segments[7:]  # 7==3CDRs+4FRs
        self.parse_cdr_coords()
        self.get_interface()

    def parse_cdr(self, scheme="chothia"):
        seq_H = self.seqs[self.antibody_chains[0]]
        chain_H = abnumber.Chain(sequence=seq_H, scheme=scheme)
        assert chain_H.is_heavy_chain(), f"Not an antibody heavy chain"
        seq_L = self.seqs[self.antibody_chains[1]]
        chain_L = abnumber.Chain(sequence=seq_L, scheme=scheme)
        assert chain_L.is_light_chain(), f"Not an antibody light chain"
        cdr_masks, segments = [], []
        for chain, seq in [(chain_H, seq_H), (chain_L, seq_L)]:
            cdr_mask = torch.zeros(len(seq)).int()
            cursor = 0
            i = 0
            for name in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
                sgm = getattr(chain, f"{name}_seq")
                cdr_mask[cursor:cursor+len(sgm)] = i
                segments.append(sgm)
                cursor += len(sgm)
                i += 1
            cdr_masks.append(cdr_mask)
        return cdr_masks[0], cdr_masks[1], segments

    def parse_cdr_coords(self):
        cdr_masks = self.cdr_masks
        chains_ab = self.antibody_chains
        atom_coords = self.atom_coords
        atom_mask = self.atom_mask

        chain_H, chain_L = chains_ab[0], chains_ab[1]
        cdrH_mask, cdrL_mask = tensor(cdr_masks[0]), tensor(cdr_masks[1])
        atomH_mask, atomL_mask = atom_mask[chain_H], atom_mask[chain_L]
        
        # Get indices of CDR residues out of all residues
        cdrH_idx = torch.isin(cdrH_mask, tensor([1,3,5])).int().nonzero().squeeze(-1)
        cdrL_idx = torch.isin(cdrL_mask, tensor([1,3,5])).int().nonzero().squeeze(-1)
        # Get indices of atoms in CDR out of all atoms
        atom_cdrH_idx = torch.isin(atomH_mask, cdrH_idx).nonzero().squeeze(-1)
        atom_cdrL_idx = torch.isin(atomL_mask, cdrL_idx).nonzero().squeeze(-1)

        # Get coordiantes of atoms in CDR
        X_atomH, X_atomL = atom_coords[chain_H], atom_coords[chain_L]
        X_atom_cdrH = torch.index_select(X_atomH, dim=0, index=atom_cdrH_idx)
        X_atom_cdrL = torch.index_select(X_atomL, dim=0, index=atom_cdrL_idx)
        
        self.cdr_atom_coords = torch.cat([X_atom_cdrH, X_atom_cdrL], dim=0)

    def get_interface(self, interface_k=6):
        chains_ab, chains_ag = self.antibody_chains, self.antigen_chains
        chains = chains_ab + chains_ag
        resi_coords = self.atom_CMR_coords
        atom_masks = self.atom_mask
        chain_H, chain_L = chains_ab[0], chains_ab[1]
        
        # X_ab:[N_ab,3], X_ag:[N_ag,3]
        X_ab = torch.cat([resi_coords[chain_H], resi_coords[chain_L]])
        X_ag = torch.cat([resi_coords[c] for c in chains_ag])

        _interf_idx_resi = {c: tensor([], dtype=torch.int32) for c in chains}

        for c in chains_ab:
            X = resi_coords[c]  # atom coords of a single Ab chain, [N,3]
            dx = X.unsqueeze(1) - X_ag.unsqueeze(0)  # [N,1,3]-[1,N_ag,3]->[N,N_ag,3]
            dists_sq = torch.sum(dx**2, dim=-1)  # [N,N_ag]
            # Get the indices of close residue pairs
            idx_para, _ = (dists_sq<=interface_k**2).int().nonzero(as_tuple=True)
            idx_para = idx_para.unique()
            _interf_idx_resi[c] = idx_para
        
        for c in chains_ag:
            X = resi_coords[c]  # atom coords of a single Ab chain, [N,3]
            dx = X.unsqueeze(1) - X_ab.unsqueeze(0)  # [N,1,3]-[1,N_ab,3]->[N,N_ab,3]
            dists_sq = torch.sum(dx**2, dim=-1)  # [N,N_ab]
            # Get the indices of close residue pairs
            idx_epi, _ = (dists_sq<=interface_k**2).int().nonzero(as_tuple=True)
            idx_epi = idx_epi.unique()
            _interf_idx_resi[c] = idx_epi

        # Check the interface residues
        cdr_center = self.cdr_atom_coords.mean(dim=0)  # [3]
        interf_idx_resi = {c: tensor([], dtype=torch.int32) for c in chains}
        for c in chains:
            idx_resi = _interf_idx_resi[c].tolist()
            unwanted = []
            for i in idx_resi:
                xyz = self.atom_CA_coords[c][i,:]
                if torch.norm(xyz-cdr_center) > 40:
                    unwanted.append(i)
                if not xyz.any():
                    unwanted.append(i)
            # remove all unwanted residues
            for j in unwanted:
                idx_resi.remove(j)
            interf_idx_resi[c] = tensor(idx_resi)

        interf_idx_atom = {c: tensor([], dtype=torch.int32) for c in chains}
        for c in chains_ab + chains_ag:
            idx_resi = interf_idx_resi[c]
            idx_atom = torch.isin(atom_masks[c], idx_resi).int().nonzero().squeeze(-1)
            interf_idx_atom[c] = idx_atom
        
        self.interface_resi_indices = interf_idx_resi
        self.interface_atom_indices = interf_idx_atom

    def generate_patch(self, mode):
        seqs = self.seqs
        chains_ab, chains_ag = self.antibody_chains, self.antigen_chains
        
        if mode == "atom":
            interface_indices = self.interface_atom_indices
            coords = self.atom_coords
            atom_masks = self.atom_mask
        elif mode == "resi":
            interface_indices = self.interface_resi_indices
            coords = self.atom_CMR_coords

        interface_coords = {}
        for c in chains_ab + chains_ag:
            idx = interface_indices[c]
            if idx.nelement():
                X = torch.index_select(coords[c], 0, idx)
            else:
                X = tensor([], dtype=torch.float32)
            interface_coords[c] = X

        X_para = torch.cat([interface_coords[c] for c in chains_ab])
        X_epi = torch.cat([interface_coords[c] for c in chains_ag])

        X_para_patch, X_epi_patch, _, _, normal_para, normal_epi = superpose(X_para, X_epi)
        if mode == "atom":
            self.interface_atom_coords2d = X_para_patch, X_epi_patch
        elif mode == "resi":
            self.interface_resi_coords2d = X_para_patch, X_epi_patch
        
        para_ctr, epi_ctr = X_para.mean(dim=0), X_epi.mean(dim=0)
        D_para_plane = -torch.matmul(para_ctr, normal_para).unsqueeze(0)
        D_epi_plane = -torch.matmul(epi_ctr, normal_epi).unsqueeze(0)
        para_plane = torch.cat([normal_para, D_para_plane])
        epi_plane = torch.cat([normal_epi, D_epi_plane])
        self.pca_plane = para_plane, epi_plane

        interface_resitypes = {}
        for c in chains_ab + chains_ag:
            idx = interface_indices[c]
            X = interface_coords[c]  # [N,3]
            seq = seqs[c]
            if mode == "atom":
                atom_mask = atom_masks[c]
                # if X.shape[0]:
                resitypes = tensor([RESIDUE_FLAGS[seq[i]] for i in atom_mask])
                resitypes = torch.index_select(resitypes, 0, idx)
                interface_resitypes[c] = resitypes  # [N_sc,3]
                self.interface_atom_resitypes = interface_resitypes
            elif mode == "resi":
                resitypes = tensor([RESIDUE_FLAGS[seq[i]] for i in idx])
                interface_resitypes[c] = resitypes
                self.interface_resitypes = interface_resitypes

    def distance_to_pca(self, mode):
        if not hasattr(self, "pca_plane"):
            raise AttributeError(("Attribute 'pca_plane' not found. Did you run `prot.generate_patch`?"))

        para_plane, epi_plane = self.pca_plane
        chains_ab, chains_ag = self.antibody_chains, self.antigen_chains
        if mode == "resi":
            coords = self.atom_CMR_coords
            interface_indices = self.interface_resi_indices
        elif mode == "atom":
            coords = self.atom_coords
            interface_indices = self.interface_atom_indices
        
        interface_coords = {}
        for c in chains_ab + chains_ag:
            idx = interface_indices[c]
            if idx.nelement():
                X = torch.index_select(coords[c], 0, idx)
            else:
                X = tensor([], dtype=torch.float32)
            interface_coords[c] = X
        X_para = torch.cat([interface_coords[c] for c in chains_ab])
        X_epi = torch.cat([interface_coords[c] for c in chains_ag])

        # Collect coordinates of non-interface residues ("frame")
        all_X_ab_frame = []
        for chain in chains_ab:
            X = coords[chain]  # [N_sc,3]
            interface_idx = interface_indices[chain]  # [N_sc_inter]
            frame_idx = list(set(range(X.size(0))).difference(set(interface_idx)))  # [N_sc_frame]
            X_frame = torch.index_select(X, 0, tensor(frame_idx))  # [N_sc_frame,3]
            all_X_ab_frame.append(X_frame)
        X_ab_frame = torch.cat(all_X_ab_frame)  # [N_frame,3]
        all_X_ag_frame = []
        for chain in chains_ag:
            X = coords[chain]  # [N_sc,3]
            interface_idx = interface_indices[chain]  # [N_sc_inter]
            frame_idx = list(
                set(range(X.size(0))).difference(set(interface_idx))
            )  # [N_sc_frame]
            # [N_sc_frame,3]
            X_frame = torch.index_select(X, 0, tensor(frame_idx))
            all_X_ag_frame.append(X_frame)
        X_ag_frame = torch.cat(all_X_ag_frame)  # [N_frame,3]

        all_distances = []
        for X_frame, X, pca_plane in [(X_ab_frame, X_para, para_plane), (X_ag_frame, X_epi, epi_plane)]:
            cylinder1, cylinder2 = build_patch_cylinder(X)
            
            num_points_in_cyl1 = torch.count_nonzero(cylinder1.is_inside(X_frame)).item()
            num_points_in_cyl2 = torch.count_nonzero(cylinder2.is_inside(X_frame)).item()
            
            distances = distance_to_plane(X, pca_plane)  # [N_para]
            # This means cylinder1 == negative intercept == interior
            if num_points_in_cyl1 > num_points_in_cyl2:
                ones = torch.ones((X.size(0), 1))  # [N_para,1]
                X_expand = torch.cat([X, ones], dim=1)  # [N_para,4]
                intercepts = torch.matmul(X_expand, pca_plane).squeeze()
                dist_signs = (intercepts > 0).int() * 2 - 1
                distances *= dist_signs
            # This means cylinder2 == positive intercept == interior
            else:
                ones = torch.ones((X.size(0), 1))  # [N_para,1]
                X_expand = torch.cat([X, ones], dim=1)  # [N_para,4]
                intercepts = torch.matmul(X_expand, pca_plane).squeeze()
                dist_signs = (intercepts < 0).int() * 2 - 1
                distances *= dist_signs
            all_distances.append(distances.tolist())
        
        if mode == "atom":
            self.interface_atom_dists = all_distances[0], all_distances[1]
        elif mode == "resi":
            self.interface_resi_dists = all_distances[0], all_distances[1]

    def write_json(self, fname):
        metadict = super().write_json(None)
        metadict["cdr_masks"] = (self.cdr_masks[0], self.cdr_masks[1])
        metadict["antibody_segments"] = self.segments
        metadict["antibody_chains"] = self.antibody_chains
        metadict["antigen_chains"] = self.antigen_chains
        metadict["CDR_residues"] = self.cdr_residues
        metadict["CDR_atom_coords"] = self.cdr_atom_coords
        metadict = tensor_to_list(metadict)
        if fname is not None:
            with open(fname, "w") as fp:
                json.dump(metadict, fp)
        return metadict


def _check_prot(
        prot: ImmuneComplex,
        min_para_len=3,
        max_para_len=9999999,
        min_epi_len=3,
        max_epi_len=999999,
        min_antigen_len=51
) -> Tuple[bool, str]:
    
    paratope_len, epitope_len = getattr(prot, "interface_sizes")
    ag_len = sum(getattr(prot, "antigen_length"))
    # para_cdr_ratio = getattr(prot, "paratope_cdr_ratio")
    # abr_size = len(getattr(prot, "abr_residues"))

    result = True
    error = []
    
    if ag_len < min_antigen_len:
        result = False
        error.append(f"too short antigen ({ag_len})")
    if paratope_len < min_para_len or paratope_len > max_para_len:
        result = False
        error.append(f"paratope size invalid: ({paratope_len})")
    if epitope_len < min_epi_len or epitope_len > max_epi_len:
        result = False
        error.append(f"epitope size invalid ({epitope_len})")
    
    return result, "OK" if not error else error


def parse_coords(pdb_struct: Structure, chains: str):
    chain_to_atomcoords = {}
    chain_to_atomCAcoords = {}
    chain_to_atomCMRcoords = {}
    chain_to_atommask = {}
    chain_to_atomtypes = {}
    
    for chain in pdb_struct.get_chains():
        # Ignore chains not included in the current complex
        chain_id = chain.get_id()
        
        if chain_id not in chains:
            continue
        
        # Process all atoms in one residue
        all_atom_coords = []  # one chain
        all_atomCA_coords = []
        all_atomCMR_coords = []
        all_atom_mask = []
        all_atom_types = []
        resi_num = 0
        
        for residue in chain.child_list:
            atom_list = residue.child_list  # A list of biopython `Atom`
            atom_list = list(filter(lambda x: x.element!="H" and x.name!="OXT", atom_list))  # only use heavy atoms
            
            resi_name = residue.resname.title()
            if seq1(resi_name) not in RESIDUE_FLAGS:  # Ignore non-AA and non-traditional AA
                continue
            
            atomCA_coords = [atom.coord for atom in atom_list if atom.name == "CA"]
            assert atomCA_coords, f"No CA at chain {chain_id}, residue {residue.id}"
            atomCA_coords = tensor(atomCA_coords[0]).unsqueeze(0)  # [1,3]
            
            atom_coords = [tensor(atom.coord).unsqueeze(0) for atom in atom_list]
            atom_types = [atom.name for atom in atom_list]

            if resi_name == "Gly":
                atomCMR_coords = atomCA_coords
            else:
                sidechain_atoms = [atom for atom in atom_list if atom.name not in ("N", "CA", "C", "O")]
                # assert set([atom.name for atom in sidechain_atoms]) == set(SIDECHAIN_ATOMS[resi_name]), f"Sidechain error: {chain_id}{residue.id[1]} - {set([atom.name for atom in sidechain_atoms])} vs. {set(SIDECHAIN_ATOMS[resi_name])}"
                if sidechain_atoms:
                    atomCMR_coords = torch.cat(
                        [tensor(atom.coord).unsqueeze(0) for atom in sidechain_atoms]
                    ).mean(dim=0).unsqueeze(0)
                else:
                    atomCMR_coords = atomCA_coords

            all_atom_coords.append(torch.cat(atom_coords, dim=0))  # [N_res,L_atom,3]
            all_atomCA_coords.append(atomCA_coords)  # [N_res, 3]
            all_atomCMR_coords.append(atomCMR_coords)  # [N_res, 3]
            atom_mask = [resi_num] * len(atom_list)
            all_atom_mask.extend(atom_mask)  # [N_atom]
            all_atom_types.extend(atom_types)  # [N_atom]

            resi_num += 1
        
        assert all_atom_coords, f"Empty chain {chain_id}"
        assert len(all_atom_coords) == len(all_atomCA_coords) == len(all_atomCMR_coords), f"Length unmatched 1"
        assert len(all_atom_mask) == len(all_atom_types), f"Length unmatched 2"

        all_atom_mask = tensor(all_atom_mask)
        all_atom_coords = torch.cat(all_atom_coords, dim=0)
        all_atomCA_coords = torch.cat(all_atomCA_coords, dim=0)
        all_atomCMR_coords = torch.cat(all_atomCMR_coords, dim=0)
        
        chain_to_atomcoords[chain_id] = all_atom_coords
        chain_to_atomCAcoords[chain_id] = all_atomCA_coords
        chain_to_atommask[chain_id] = all_atom_mask
        chain_to_atomtypes[chain_id] = all_atom_types
        chain_to_atomCMRcoords[chain_id] = all_atomCMR_coords

    return chain_to_atomcoords, chain_to_atomCAcoords, chain_to_atomCMRcoords, chain_to_atommask, chain_to_atomtypes


def parse_pdb(id_, fname, chains, mode="resi", is_complex=True):
    r""" Parse a PDB file and create a `Protein` instance based on it.

    Args:
        chains (str): Identifiers of chains of interest.
        mode (str, optional): Create the protein in "resi" or "atom" 
        mode. Defaults to "resi".
        is_complex (bool, optional): Create a single protein or an 
        Ab-Ag complex. Defaults to True.

    Returns:
        Protein: a Complex instance if `is_complex=True` with the 
        antibody CDR, binding interface and patch computed and
        a Protein instance if `is_complex=False`. For a single 
        protein, can't generate patch here because the required 
        interface can only be manually specified by the user later.
    """
    if is_complex and len(chains) < 3:
        raise ValueError("Antibody-antigen complex should contain at least 3 chains")
    try:
        parser = PDB.PDBParser()
        pdb_struct = parser.get_structure(id_, str(fname))
        chain_to_seq = {}
        for chain in pdb_struct.get_chains():
            seq = seq1("".join([residue.resname for residue in chain]))
            seq = "".join(list(filter(lambda x: x in RESIDUE_FLAGS, seq)))
            chain_to_seq[chain.id] = seq
        for chain in list(chain_to_seq.keys()):
            if chain not in chains:
                chain_to_seq.pop(chain)
        
        atom_coords, atom_CA_coords, atom_CMR_coords, atom_mask, atom_types = parse_coords(pdb_struct, chains)
        # surfaces = get_prot_surface(pdb_struct)
        for c in chain_to_seq:
            assert len(chain_to_seq[c]) == len(atom_CA_coords[c]), "sequence length unmatched with coordinates"
        
        if is_complex:
            prot = ImmuneComplex(
                id_=id_, 
                seqs=chain_to_seq,
                antibody_chains=chains[:2],
                antigen_chains=chains[2:],
                atom_coords=atom_coords,
                atom_CA_coords=atom_CA_coords,
                atom_CMR_coords=atom_CMR_coords,
                atom_mask=atom_mask,
                atom_types=atom_types
            )
            prot.initialize()
            qualified, comment = _check_prot(prot)
            if not qualified:
                print(f"{id_}: {comment}")
                return None
            else:
                prot.generate_patch(mode)
                prot.distance_to_pca(mode)
                return prot
        else:
            prot = Protein(
                id_=id_, 
                seqs=chain_to_seq,
                chains=chains,
                atom_coords=atom_coords,
                atom_CA_coords=atom_CA_coords,
                atom_CMR_coords=atom_CMR_coords,
                atom_mask=atom_mask,
                atom_types=atom_types
            )
            return prot
    
    except (AssertionError, abnumber.ChainParseError) as e:
        print(f"{id_}: {e}")
        return None
