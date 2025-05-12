import collections
from collections.abc import Mapping, Sequence
from absl import logging
import numpy as np
import rdkit.Chem as rd_chem
from rdkit.Chem import AllChem as rd_all_chem

def distance(xyz1, xyz2):
    """
    Calculates the distance between two points.
    """
    if type(xyz1) != np.ndarray:
        xyz1 = np.array(xyz1)
    if type(xyz2) != np.ndarray:
        xyz2 = np.array(xyz2)
    dist = np.linalg.norm(xyz2-xyz1)
    return round(dist,3)
    
def vector(xyz1, xyz2):
    """
    Produces a vector array between two points.
    """
    vector = [round(c2 - c1,3) for c1,c2 in zip(xyz1,xyz2)]
    return vector

def calc_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    #return round(np.degrees(angle),3)
    return round(angle,3)

def calc_dihed(a,b,c,d):
    v1 = b - a
    v2 = c - b
    v3 = d - c
    x12 = np.cross(v1,v2)
    x23 = np.cross(v3,v2)
    xx = np.cross(x12,x23)
    y = np.dot(xx, v2)*(1.0/np.linalg.norm(v2))
    x = np.dot(x12,x23)
    dihed = np.arctan2(y,x)
    return round(np.degrees(dihed),3)


def _populate_atoms_in_mol(
    mol: rd_chem.Mol,
    atom_names: Sequence[str],
    atom_types: Sequence[str],
    atom_charges: Sequence[int],
    implicit_hydrogens: bool,
    ligand_name: str,
    atom_leaving_flags: Sequence[str],
):
  """Populate the atoms of a Mol given atom features.

  Args:
    mol: Mol object.
    atom_names: Names of the atoms.
    atom_types: Types of the atoms.
    atom_charges: Charges of the atoms.
    implicit_hydrogens: Whether to mark the atoms to allow implicit Hs.
    ligand_name: Name of the ligand which the atoms are in.
    atom_leaving_flags: Whether the atom is possibly a leaving atom. Values from
      the CCD column `_chem_comp_atom.pdbx_leaving_atom_flag`. The expected
      values are 'Y' (yes), 'N' (no), '?' (unknown/unset, interpreted as no).

  Raises:
    ValueError: If atom type is invalid.
  """
  # Map atom names to the position they will take in the rdkit molecule.
  atom_name_to_idx = {name: i for i, name in enumerate(atom_names)}

  for atom_name, atom_type, atom_charge, atom_leaving_flag in zip(
      atom_names, atom_types, atom_charges, atom_leaving_flags, strict=True
  ):
    try:
      if atom_type == 'X':
        atom_type = '*'
      atom = rd_chem.Atom(atom_type)
    except RuntimeError as e:
      raise ValueError(f'Failed to use atom type: {str(e)}') from e

    if not implicit_hydrogens:
      atom.SetNoImplicit(True)

    atom.SetProp('atom_name', atom_name)
    atom.SetProp('atom_leaving_flag', atom_leaving_flag)
    atom.SetFormalCharge(atom_charge)
    residue_info = rd_chem.AtomPDBResidueInfo()
    residue_info.SetName(_format_atom_name(atom_name, atom_type))
    residue_info.SetIsHeteroAtom(True)
    residue_info.SetResidueName(ligand_name)
    residue_info.SetResidueNumber(1)
    atom.SetPDBResidueInfo(residue_info)
    atom_index = mol.AddAtom(atom)
    assert atom_index == atom_name_to_idx[atom_name]


def _populate_bonds_in_mol(
    mol: rd_chem.Mol,
    atom_names: Sequence[str],
    bond_begins: Sequence[str],
    bond_ends: Sequence[str],
    bond_orders: Sequence[str],
    bond_is_aromatics: Sequence[bool],
):
  """Populate the bonds of a Mol given bond features.

  Args:
    mol: Mol object.
    atom_names: Names of atoms in the molecule.
    bond_begins: Names of atoms at the beginning of the bond.
    bond_ends: Names of atoms at the end of the bond.
    bond_orders: What order the bonds are.
    bond_is_aromatics: Whether the bonds are aromatic.
  """
  atom_name_to_idx = {name: i for i, name in enumerate(atom_names)}
  for begin, end, bond_type, is_aromatic in zip(
      bond_begins, bond_ends, bond_orders, bond_is_aromatics, strict=True
  ):
    begin_name, end_name = atom_name_to_idx[begin], atom_name_to_idx[end]
    bond_idx = mol.AddBond(begin_name, end_name, bond_type)
    mol.GetBondWithIdx(bond_idx - 1).SetIsAromatic(is_aromatic)