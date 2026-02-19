import sys
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

def count_atoms_from_smiles(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol.GetNumAtoms()

    # Fallback: simples Token-Zählen
    tokens = re.findall(r'\[.*?\]|Br|Cl|Si|Na|Ca|[BCNOPSFIK]', smiles)
    return None

def main(csv_path):
    df = pd.read_csv(csv_path)

    # Spaltenname robust finden
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError("Keine SMILES-Spalte gefunden")

    atom_counts = []

    for _, row in df.iterrows():
        atoms = count_atoms_from_smiles(row[smiles_col])
        atom_counts.append(atoms)
    #number of 0
    zero_count = atom_counts.count(None)
    print(f"Anzahl ungültiger SMILES: {zero_count}")
    df["atom_count"] = atom_counts
    df = df.dropna(subset=["atom_count"])

    mean_atoms = df["atom_count"].mean()

    plt.hist(df["atom_count"], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(df["atom_count"].mean(), color='red', linestyle='--', label=f"Mittelwert = {df['atom_count'].mean():.2f}")
    plt.xlabel("Anzahl Atome")
    plt.ylabel("Anzahl Moleküle")
    plt.title("Verteilung der Atomanzahl")
    plt.legend()
    plt.savefig("/hkfs/work/workspace_haic/scratch/rx3495-workspace_C/tmp/test/atom_count_histogram.png", dpi=300)
    plt.close()
    print("Mittelwert Atomanzahl:", df["atom_count"].mean())
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zinc_num_atoms_csv.py <pfad_zur_csv>")
        sys.exit(1)

    main(sys.argv[1])
