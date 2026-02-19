import math
import numpy as np
import torch
import re
import os
import tempfile
import shutil
import wandb
import pandas as pd
from tqdm import tqdm
from torchmetrics import MeanSquaredError, MeanAbsoluteError

try:
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
    from rdkit.Chem.rdForceFieldHelpers import (
        MMFFHasAllMoleculeParams,
        MMFFOptimizeMolecule,
    )

    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn

    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, train_smiles=None, args=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.args = args

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles
        self.cond_val = MeanAbsoluteError()
        
        # Normalisierungsparameter für conditional properties
        self.normalization_params = self._load_normalization_params()

    def _load_normalization_params(self):
        """
        Lade Normalisierungsparameter (mean und std) für alle QM9-Properties
        aus den Trainingsdaten.
        """
        normalization_params = {}
        try:
            import pathlib
            # Versuche, die QM9 Trainingsdaten zu laden
            base_path = pathlib.Path(__file__).parents[2]
            train_file = base_path / "data/qm9/qm9_pyg/raw/train.csv"
            val_file = base_path / "data/qm9/qm9_pyg/raw/val.csv"
            test_file = base_path / "data/qm9/qm9_pyg/raw/test.csv"
            
            if train_file.exists() and val_file.exists() and test_file.exists():
                train_df = pd.read_csv(train_file, index_col=0)
                val_df = pd.read_csv(val_file, index_col=0)
                test_df = pd.read_csv(test_file, index_col=0)
                all_data = pd.concat([train_df, val_df, test_df])
                
                # Berechne mean und std für alle Properties
                properties = ['a', 'b', 'c', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve']
                for prop in properties:
                    if prop in all_data.columns:
                        normalization_params[prop] = {
                            'mean': all_data[prop].mean(),
                            'std': all_data[prop].std()
                        }
                print(f"✓ Normalisierungsparameter geladen für {len(normalization_params)} Properties")
            else:
                print(f"⚠ QM9 Trainingsdaten nicht gefunden - Normalisierung deaktiviert")
        except Exception as e:
            print(f"⚠ Fehler beim Laden der Normalisierungsparameter: {e}")
        
        return normalization_params

    def _normalize_property(self, property_name, value):
        """
        Normalisiere einen Property-Wert using mean und std aus Trainingsdaten.
        Formula: (value - mean) / std
        """
        if property_name in self.normalization_params:
            params = self.normalization_params[property_name]
            if params['std'] > 0:
                return (value - params['mean']) / params['std']
        return value

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        all_smiles_without_test = []

        for graph in tqdm(
            generated, desc="Generated molecules validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            all_smiles_without_test.append(mol2smilesWithNoSanitize(mol))
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        with open(r"final_smiles_all.txt", "w") as fp:
            for smiles in all_smiles_without_test:
                # write each item on a new line
                fp.write("%s\n" % smiles)
            print("All smiles saved")

        print(all_smiles_without_test)
        df = pd.DataFrame(all_smiles_without_test, columns=["SMILES"])
        df.to_csv("final_smiles_all.csv", index=False)
        print("All SMILES saved to CSV")

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in tqdm(unique, desc="Unique molecules novelty check progress"):
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in tqdm(
            generated, desc="Generated molecules relaxed validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def cond_sample_metric(self, samples, input_properties):
        self.num_valid_molecules = 0
        self.num_total = 0
        
        total, used, free = shutil.disk_usage("/")
        print(f"Disk space - Total: {total // (2**30)}GB, Used: {used // (2**30)}GB, Free: {free // (2**30)}GB")
        
        # Zuerst target_keys bestimmen
        target_keys = getattr(self, 'target_keys', [])

        # ODER alternativ extrahieren Sie hier neu:
        if not target_keys and self.args is not None and hasattr(self.args, 'general'):
            dynamic = getattr(self.args.general, 'dynamic', None)
            target = getattr(self.args.general, 'target', None)
            
            if isinstance(dynamic, bool) and dynamic:
                # dynamic=true -> parse target string
                if isinstance(target, str):
                    # Ersetze Kommas durch Leerzeichen, dann split
                    target = target.replace(',', ' ')
                    target_keys = [k.strip() for k in target.split() if k.strip()]
                elif hasattr(target, '__iter__'):  # Für Listen
                    target_keys = [str(k).strip() for k in target if k]
            elif target == "both":
                target_keys = ["mu", "homo"]
            elif target in ["mu", "homo"]:
                target_keys = [target]
        
        # WARNUNG: Wenn alpha (polarizability) in target_keys ist, sagen wir, dass es nicht unterstützt wird
        if "alpha" in target_keys:
            print("WARNING: Polarizability (alpha) calculation is not currently supported. Removing from target_keys.")
            target_keys = [k for k in target_keys if k != "alpha"]
        
        if len(target_keys) == 0:
            print("No valid properties to compute.")
            return float('nan'), 0.0
        
        print(f"Target keys for computation: {target_keys}")
        
        # Mapping von Target-Namen zu Psi4 Berechnungen
        psi4_calculations = {
            "mu": ("dipole", "SCF DIPOLE"),
            "homo": ("homo", "epsilon_a_subset"),
            "lumo": ("lumo", "epsilon_a_subset"),
            "gap": ("gap", None),
            "alpha": ("polarizability", None),
            "r2": ("r2", None),
            "zpve": ("zpve", None),
        }
        
        # Initialisiere computed_properties
        computed_properties = {}
        for key in target_keys:
            if key in psi4_calculations:
                computed_properties[key] = []
        
        true_properties = []
        
        # Psi4 Konfiguration (EINMAL am Anfang)
        psi4.set_num_threads(nthread=4)
        psi4.set_memory("6GB")
        psi4.core.set_output_file("psi4_output.dat", False)
        
        # Set explicit scratch directory to avoid PSIO errors
        import tempfile
        psi4_scratch = os.path.join(tempfile.gettempdir(), "psi4_scratch")
        os.makedirs(psi4_scratch, exist_ok=True)
        os.environ["PSI_SCRATCH"] = psi4_scratch
        
        # Berechnungslevel
        method = "b3lyp"
        basis = "6-31g*"
        
        # Input properties normalisieren
        if isinstance(input_properties, (list, tuple)):
            input_properties = torch.stack(input_properties)
        
        if input_properties.dim() == 1:
            input_properties = input_properties.unsqueeze(-1)
        
        # Prüfe, ob input_properties die richtige Dimension hat
        if input_properties.dim() == 2 and input_properties.shape[1] != len(target_keys):
            print(f"Warning: input_properties shape {input_properties.shape} doesn't match target_keys length {len(target_keys)}")
        
        for i, sample in enumerate(samples):
            self.num_total += 1
            temp_dir = tempfile.mkdtemp(prefix=f"psi4_calc_{i}_")
            
            try:
                # Set per-molecule scratch directory
                psi4.core.IOManager.shared_object().set_default_path(temp_dir)
                psi4.core.clean()  # Clean up before each calculation
            except Exception as e:
                print(f"Warning: Could not set scratch dir for {i}: {e}")
            
            atom_types, edge_types = sample
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Invalid chemistry for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # 3D Struktur generieren
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = i + 1
            params.useRandomCoords = True

            # Optional: best-effort compatibility, no hard failure
            if hasattr(params, "maxAttempts"):
                params.maxAttempts = 1000
            elif hasattr(params, "maxIters"):
                params.maxIters = 1000
            # else: do nothing (new RDKit versions)
            params.useRandomCoords = True
            
            try:
                embed_result = EmbedMolecule(mol, params)
                if embed_result == -1:
                    print(f"Could not embed molecule {i}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
            except Exception as e:
                print(f"Embedding error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # Strukturoptimierung mit MMFF
            try:
                # Prüfe ob MMFF Parameter verfügbar sind
                if not MMFFHasAllMoleculeParams(mol):
                    print(f"No MMFF parameters for molecule {i}")
                    # Alternative: SimpleMolecule Force Field
                    from rdkit.Chem import AllChem
                    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
                else:
                    s = MMFFOptimizeMolecule(mol, maxIters=200)
                    if s != 0:
                        print(f"MMFF optimization issue for molecule {i}: status {s}")
            except Exception as e:
                print(f"Optimization error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            try:
                conf = mol.GetConformer()
                if conf is None:
                    print(f"No conformer for molecule {i}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
            except Exception as e:
                print(f"Conformer error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # Formal charge und Spin-Multiplicity
            try:
                mol_FormalCharge = Chem.GetFormalCharge(mol)
                
                # Spin multiplicity berechnen
                NumRadicalElectrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
                TotalElectronicSpin = NumRadicalElectrons / 2
                mol_spin_multiplicity = int(2 * TotalElectronicSpin + 1)
                
                # Sicherstellen, dass Multiplicity >= 1
                if mol_spin_multiplicity < 1:
                    mol_spin_multiplicity = 1
                    
                mol_input = f"{mol_FormalCharge} {mol_spin_multiplicity}"
                
            except Exception as e:
                print(f"Charge/spin calculation error for sample {i}: {e}")
                mol_input = "0 1"  # Default
                
            # XYZ Koordinaten
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                mol_input += f"\n{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}"
            
            try:
                molecule = psi4.geometry(mol_input)
            except Exception as e:
                print(f"Psi4 geometry error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            print(f"Psi4 calculation for sample {i} starts...")
            
            try:
                # Konfiguriere Psi4 für diese Berechnung
                psi4.set_options({
                    "basis": basis,
                    'scf_type': 'df',
                    'diis_start': 3,
                    'diis_max_vecs': 8,
                    'maxiter': 200,
                    'guess': 'sad',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8,
                    'ints_tolerance': 1e-12,
                })
                
                # Bestimme welche Properties berechnet werden müssen
                properties_to_compute = []
                psi4_property_map = {
                    "mu": "dipole",
                    "homo": "homo",
                    "lumo": "lumo",
                    "gap": "gap",
                    "r2": "r2",
                    "zpve": "zpve",
                }
                
                for key in target_keys:
                    if key in psi4_property_map:
                        properties_to_compute.append(psi4_property_map[key])
                
                # Führe die Berechnung durch
                if len(properties_to_compute) > 0:
                    # Berechne Properties
                    prop_result = psi4.properties(
                        f"{method}/{basis}",
                        properties=properties_to_compute,
                        molecule=molecule,
                        return_wfn=True
                    )
                    
                    if isinstance(prop_result, tuple):
                        energy, wfn = prop_result
                    else:
                        wfn = prop_result
                else:
                    # Nur Energie berechnen
                    energy, wfn = psi4.energy(f"{method}/{basis}", molecule=molecule, return_wfn=True)
                
                # Sammle true_properties
                if i < input_properties.shape[0]:
                    if input_properties.shape[1] == len(target_keys):
                        prop = input_properties[i]
                    else:
                        if input_properties.shape[1] == 1:
                            prop = input_properties[i].repeat(len(target_keys))
                        else:
                            prop = input_properties[i, 0].repeat(len(target_keys))
                    
                    true_properties.append(prop)
                else:
                    print(f"Warning: sample {i} has no corresponding input property")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
            except psi4.driver.SCFConvergenceError:
                print(f"Psi4 did not converge for sample {i}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            except Exception as e:
                print(f"Psi4 calculation error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # Berechne die gewünschten Properties
            sample_properties = {}
            
            try:
                print(f"Target keys for calculation: {target_keys}")
                for key in target_keys:
                    if key == "mu":
                        try:
                            dip_x, dip_y, dip_z = wfn.variable("SCF DIPOLE")[0:3]
                            dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                            sample_properties[key] = dipole_moment
                            print(f"Sample {i} - Dipole moment: {dipole_moment:.6f} Debye")
                        except Exception as e:
                            print(f"Dipole calculation error for sample {i}: {e}")
                            sample_properties[key] = float('nan')
                    
                    elif key in ["homo", "lumo", "gap"]:
                        try:
                            eps_a = wfn.epsilon_a().to_array()
                            LUMO_idx = wfn.nalpha()
                            HOMO_idx = LUMO_idx - 1
                            
                            if key == "homo":
                                homo_value = eps_a[HOMO_idx]
                                sample_properties[key] = homo_value
                                print(f"Sample {i} - HOMO: {homo_value:.6f} Hartree")
                            elif key == "lumo":
                                lumo_value = eps_a[LUMO_idx]
                                sample_properties[key] = lumo_value
                                print(f"Sample {i} - LUMO: {lumo_value:.6f} Hartree")
                            elif key == "gap":
                                gap_value = eps_a[LUMO_idx] - eps_a[HOMO_idx]
                                sample_properties[key] = gap_value
                                print(f"Sample {i} - Gap: {gap_value:.6f} Hartree")
                        except Exception as e:
                            print(f"Orbital energy calculation error for sample {i}: {e}")
                            sample_properties[key] = float('nan')
                    
                    elif key == "alpha":
                        print(f"WARNING: Polarizability (alpha) calculation skipped for sample {i}")
                        sample_properties[key] = float('nan')
                    
                    elif key == "r2":
                        try:
                            r2_value = wfn.variable("SCF <R^2>")
                            sample_properties[key] = r2_value
                            print(f"Sample {i} - R^2: {r2_value:.6f} Bohr^2")
                        except Exception as e:
                            print(f"R^2 calculation error for sample {i}: {e}")
                            sample_properties[key] = float('nan')
                    
                    elif key == "zpve":
                        try:
                            zpve_value = wfn.variable("SCF ZPVE")
                            sample_properties[key] = zpve_value
                            print(f"Sample {i} - ZPVE: {zpve_value:.6f} Hartree")
                        except Exception as e:
                            print(f"ZPVE calculation error for sample {i}: {e}")
                            sample_properties[key] = float('nan')
                
                # Füge berechnete Properties zu den Listen hinzu
                # WICHTIG: Normalisiere die Properties damit sie mit condition_values vergleichbar sind
                for key in target_keys:
                    if key in sample_properties and not math.isnan(sample_properties[key]):
                        normalized_value = self._normalize_property(key, sample_properties[key])
                        computed_properties[key].append(normalized_value)
                    elif key in sample_properties:
                        computed_properties[key].append(sample_properties[key])
                
                self.num_valid_molecules += 1
                
            except Exception as e:
                print(f"Property calculation error for sample {i}: {e}")
            
            finally:
                # Aufräumen
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
        
        if len(true_properties) == 0:
            print("WARNING: No valid molecules passed psi4 calculation. Skipping conditional metrics.")
            # Clean up global scratch directory
            try:
                shutil.rmtree(psi4_scratch, ignore_errors=True)
            except:
                pass
            return float('nan'), 0.0
        
        # Konvertiere true_properties zu Tensor
        try:
            if isinstance(true_properties[0], torch.Tensor):
                true_properties_tensor = torch.stack(true_properties)
            else:
                true_properties_tensor = torch.tensor(true_properties)
        except Exception as e:
            print(f"Error converting true_properties to tensor: {e}")
            return float('nan'), 0.0
        
        # Berechne Metriken
        mae_results = {}
        cond_val_results = {}
        
        for idx, key in enumerate(target_keys):
            if key in computed_properties and len(computed_properties[key]) > 0:
                # Filter NaN Werte
                computed_values = [v for v in computed_properties[key] if not math.isnan(v)]
                if len(computed_values) == 0:
                    print(f"No valid {key} values computed")
                    continue
                    
                computed_tensor = torch.FloatTensor(computed_values)
                
                # Extrahiere entsprechende true values
                if true_properties_tensor.dim() == 1:
                    # Nur ein Target
                    true_values = true_properties_tensor[:len(computed_values)]
                else:
                    # Mehrere Targets
                    true_values = true_properties_tensor[:len(computed_values), idx]
                
                # Berechne MAE
                try:
                    mae = torch.mean(torch.abs(computed_tensor - true_values))
                    mae_results[f"mae_{key}"] = mae.item()
                    print(f"MAE for {key}: {mae.item():.6f}")
                except Exception as e:
                    print(f"MAE calculation error for {key}: {e}")
                    mae_results[f"mae_{key}"] = float('nan')
                
                # Berechne conditional validity
                thresholds = {
                    "mu": 0.1,      # Debye
                    "homo": 0.01,   # Hartree
                    "lumo": 0.01,   # Hartree
                    "gap": 0.02,    # Hartree
                    "alpha": 2.0,   # Bohr^3
                    "r2": 2.0,      # Bohr^2
                    "zpve": 0.001,  # Hartree
                }
                
                threshold = thresholds.get(key, 0.1)
                try:
                    cond_val = (torch.abs(computed_tensor - true_values) < threshold).float().mean()
                    cond_val_results[f"cond_val_{key}"] = cond_val.item()
                    print(f"Conditional validity for {key}: {cond_val.item()*100:.2f}%")
                except Exception as e:
                    print(f"Conditional validity calculation error for {key}: {e}")
                    cond_val_results[f"cond_val_{key}"] = 0.0
        
        # Für backward compatibility
        if len(target_keys) == 1:
            key = target_keys[0]
            # Clean up global scratch directory
            try:
                shutil.rmtree(psi4_scratch, ignore_errors=True)
            except:
                pass
            
            if key in mae_results:
                return mae_results[f"mae_{key}"], self.num_valid_molecules / max(self.num_total, 1)
            else:
                return float('nan'), self.num_valid_molecules / max(self.num_total, 1)
        else:
            # Für multiple Targets
            valid_mae = [v for v in mae_results.values() if not math.isnan(v)]
            valid_cond = [v for v in cond_val_results.values() if not math.isnan(v)]
            
            avg_mae = sum(valid_mae) / len(valid_mae) if valid_mae else float('nan')
            avg_cond_val = sum(valid_cond) / len(valid_cond) if valid_cond else float('nan')
            
            result_dict = {
                "mae_mean": avg_mae,
                "cond_val_mean": avg_cond_val,
                "num_valid": self.num_valid_molecules,
                "num_total": self.num_total,
                "valid_ratio": self.num_valid_molecules / max(self.num_total, 1),
            }
            
            result_dict.update(mae_results)
            result_dict.update(cond_val_results)
            
            # Clean up global scratch directory
            try:
                shutil.rmtree(psi4_scratch, ignore_errors=True)
            except:
                pass
            
            return result_dict, self.num_valid_molecules / max(self.num_total, 1)
            self.num_valid_molecules = 0
            self.num_total = 0
            
            total, used, free = shutil.disk_usage("/")
            print(f"Disk space - Total: {total // (2**30)}GB, Used: {used // (2**30)}GB, Free: {free // (2**30)}GB")
            
            # Zuerst target_keys bestimmen
            target_keys = getattr(self, 'target_keys', [])

            # ODER alternativ extrahieren Sie hier neu:
            if not target_keys and self.args is not None and hasattr(self.args, 'general'):
                dynamic = getattr(self.args.general, 'dynamic', None)
                target = getattr(self.args.general, 'target', None)
                
                if isinstance(dynamic, bool) and dynamic:
                    # dynamic=true -> parse target string
                    if isinstance(target, str):
                        # Ersetze Kommas durch Leerzeichen, dann split
                        target = target.replace(',', ' ')
                        target_keys = [k.strip() for k in target.split() if k.strip()]
                    elif hasattr(target, '__iter__'):  # Für Listen
                        target_keys = [str(k).strip() for k in target if k]
                elif target == "both":
                    target_keys = ["mu", "homo"]
                elif target in ["mu", "homo"]:
                    target_keys = [target]
            print(f"Target keys for computation: {target_keys}")
            
            # Mapping von Target-Namen zu Psi4 Berechnungen (KORREKTE REIHENFOLGE)
            psi4_calculations = {
                "mu": ("dipole", "SCF DIPOLE"),
                "homo": ("homo", "epsilon_a_subset"),
                "lumo": ("lumo", "epsilon_a_subset"),
                "gap": ("gap", None),
                "alpha": ("polarizability", None),
                "r2": ("r2", None),
                "zpve": ("zpve", None),
            }
            
            # Initialisiere computed_properties
            computed_properties = {}
            for key in target_keys:
                if key in psi4_calculations:
                    computed_properties[key] = []
            
            true_properties = []
            
            # Psi4 Konfiguration
            psi4.set_num_threads(nthread=4)
            psi4.set_memory("6GB")  # Increased from 5GB
            psi4.core.set_output_file("psi4_output.dat", False)
            
            # Set explicit scratch directory to avoid PSIO errors
            import tempfile
            psi4_scratch = os.path.join(tempfile.gettempdir(), "psi4_scratch")
            os.makedirs(psi4_scratch, exist_ok=True)
            os.environ["PSI_SCRATCH"] = psi4_scratch
            psi4.core.IOManager.shared_object().set_default_path(psi4_scratch)
            
            # Berechnungslevel
            method = "b3lyp"
            basis = "6-31g*"

            try:
                # Configure Psi4 for property calculation
                psi4.set_options({
                    "basis": basis,
                    'scf_type': 'df',
                    'diis_start': 3,
                    'diis_max_vecs': 8,
                    'maxiter': 200,
                    'guess': 'sad',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8,
                    'ints_tolerance': 1e-12,
                })
                
                # Different calculation strategies based on what properties we need
                properties_to_compute = []
                
                # Map target keys to Psi4 property names
                psi4_property_map = {
                    "mu": "dipole",
                    "homo": "homo",
                    "lumo": "lumo",
                    "gap": "gap",
                    "alpha": "polarizability",
                    "r2": "r2",
                    "zpve": "zpve",
                }
                
                # Build list of properties to compute
                for key in target_keys:
                    if key in psi4_property_map:
                        properties_to_compute.append(psi4_property_map[key])
                
                print(f"Computing properties for sample {i}: {properties_to_compute}")
                
                # SPECIAL HANDLING FOR POLARIZABILITY
                if "polarizability" in properties_to_compute:
                    # Polarizability requires special treatment
                    # Compute energy first
                    energy, wfn = psi4.energy(f"{method}/{basis}", molecule=molecule, return_wfn=True)
                    
                    # Then compute polarizability separately
                    try:
                        # Enable frequency-dependent response for polarizability
                        psi4.set_options({'omega': [0.0, 0.0, 0.0]})  # Static polarizability
                        
                        # Compute polarizability using properties
                        polar_result = psi4.properties(
                            f"{method}/{basis}",
                            properties=['polarizability'],
                            molecule=molecule,
                            ref_wfn=wfn
                        )
                        
                        # Extract polarizability from variables
                        try:
                            alpha_xx = psi4.core.variable("SCF POLARIZABILITY XX")
                            alpha_yy = psi4.core.variable("SCF POLARIZABILITY YY")
                            alpha_zz = psi4.core.variable("SCF POLARIZABILITY ZZ")
                            alpha_iso = (alpha_xx + alpha_yy + alpha_zz) / 3.0
                            print(f"Sample {i} - Polarizability: {alpha_iso:.6f} Bohr^3")
                        except:
                            print(f"Warning: Could not extract polarizability for sample {i}")
                            alpha_iso = float("nan")
                        
                        # Store wavefunction for other properties
                        wfn_for_other = wfn
                        
                    except Exception as e:
                        print(f"Polarizability calculation failed for sample {i}: {e}")
                        alpha_iso = float("nan")
                        # Continue with energy calculation for other properties
                        energy, wfn_for_other = psi4.energy(f"{method}/{basis}", molecule=molecule, return_wfn=True)
                else:
                    # For non-polarizability properties, do regular calculation
                    if len(properties_to_compute) > 0:
                        prop_result = psi4.properties(
                            f"{method}/{basis}",
                            properties=properties_to_compute,
                            molecule=molecule,
                            return_wfn=True
                        )
                        if isinstance(prop_result, tuple):
                            energy, wfn_for_other = prop_result
                        else:
                            wfn_for_other = prop_result
                    else:
                        # Just compute energy if no properties requested
                        energy, wfn_for_other = psi4.energy(f"{method}/{basis}", molecule=molecule, return_wfn=True)
                
                # Sammle true_properties
                if i < input_properties.shape[0]:
                    if input_properties.shape[1] == len(target_keys):
                        prop = input_properties[i]
                    else:
                        if input_properties.shape[1] == 1:
                            prop = input_properties[i].repeat(len(target_keys))
                        else:
                            prop = input_properties[i, 0].repeat(len(target_keys))
                    
                    true_properties.append(prop)
                else:
                    print(f"Warning: sample {i} has no corresponding input property")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
            except psi4.driver.SCFConvergenceError:
                print(f"Psi4 did not converge for sample {i}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            except Exception as e:
                print(f"Psi4 calculation error for sample {i}: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # Input properties normalisieren
            if isinstance(input_properties, (list, tuple)):
                input_properties = torch.stack(input_properties)
            
            if input_properties.dim() == 1:
                input_properties = input_properties.unsqueeze(-1)
            
            # Prüfe, ob input_properties die richtige Dimension hat
            if input_properties.dim() == 2 and input_properties.shape[1] != len(target_keys):
                print(f"Warning: input_properties shape {input_properties.shape} doesn't match target_keys length {len(target_keys)}")
            
            for i, sample in enumerate(samples):
                self.num_total += 1
                temp_dir = tempfile.mkdtemp(prefix=f"psi4_calc_{i}_")
                
                try:
                    # Set per-molecule scratch directory
                    psi4.core.IOManager.shared_object().set_default_path(temp_dir)
                    psi4.core.clean()  # Clean up before each calculation
                except Exception as e:
                    print(f"Warning: Could not set scratch dir for {i}: {e}")
                
                atom_types, edge_types = sample
                mol = build_molecule_with_partial_charges(
                    atom_types, edge_types, self.dataset_info.atom_decoder
                )
                
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    print(f"Invalid chemistry for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                # 3D Struktur generieren
                mol = Chem.AddHs(mol)
                params = ETKDGv3()
                params.randomSeed = i + 1
                params.useRandomCoords = True

                # Optional: best-effort compatibility, no hard failure
                if hasattr(params, "maxAttempts"):
                    params.maxAttempts = 1000
                elif hasattr(params, "maxIters"):
                    params.maxIters = 1000
                # else: do nothing (new RDKit versions)
                params.useRandomCoords = True
                
                try:
                    embed_result = EmbedMolecule(mol, params)
                    if embed_result == -1:
                        print(f"Could not embed molecule {i}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        continue
                except Exception as e:
                    print(f"Embedding error for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                # Strukturoptimierung mit MMFF
                try:
                    # Prüfe ob MMFF Parameter verfügbar sind
                    if not MMFFHasAllMoleculeParams(mol):
                        print(f"No MMFF parameters for molecule {i}")
                        # Alternative: SimpleMolecule Force Field
                        from rdkit.Chem import AllChem
                        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
                    else:
                        s = MMFFOptimizeMolecule(mol, maxIters=200)
                        if s != 0:
                            print(f"MMFF optimization issue for molecule {i}: status {s}")
                except Exception as e:
                    print(f"Optimization error for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                try:
                    conf = mol.GetConformer()
                    if conf is None:
                        print(f"No conformer for molecule {i}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        continue
                except Exception as e:
                    print(f"Conformer error for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                # Formal charge und Spin-Multiplicity
                try:
                    mol_FormalCharge = Chem.GetFormalCharge(mol)
                    
                    # Spin multiplicity berechnen
                    NumRadicalElectrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
                    TotalElectronicSpin = NumRadicalElectrons / 2
                    mol_spin_multiplicity = int(2 * TotalElectronicSpin + 1)
                    
                    # Sicherstellen, dass Multiplicity >= 1
                    if mol_spin_multiplicity < 1:
                        mol_spin_multiplicity = 1
                        
                    mol_input = f"{mol_FormalCharge} {mol_spin_multiplicity}"
                    
                except Exception as e:
                    print(f"Charge/spin calculation error for sample {i}: {e}")
                    mol_input = "0 1"  # Default
                    
                # XYZ Koordinaten
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    mol_input += f"\n{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}"
                
                try:
                    molecule = psi4.geometry(mol_input)
                except Exception as e:
                    print(f"Psi4 geometry error for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                
                
                print(f"Psi4 calculation for sample {i} starts...")
                
                try:
                    energy, wfn = psi4.properties(
                        "b3lyp/6-31g*",
                        properties=["polarizability"],
                        molecule=molecule,
                        return_wfn=True
                    )
                    # Sammle true_properties
                    if i < input_properties.shape[0]:
                        # Bei mehreren Targets: nimm die entsprechenden Spalten
                        if input_properties.shape[1] == len(target_keys):
                            prop = input_properties[i]
                        else:
                            # Fallback: nimm die erste Spalte oder wiederhole
                            if input_properties.shape[1] == 1:
                                prop = input_properties[i].repeat(len(target_keys))
                            else:
                                prop = input_properties[i, 0].repeat(len(target_keys))
                        
                        true_properties.append(prop)
                    else:
                        print(f"Warning: sample {i} has no corresponding input property")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        continue
                    
                except psi4.driver.SCFConvergenceError:
                    print(f"Psi4 did not converge for sample {i}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                except Exception as e:
                    print(f"Psi4 calculation error for sample {i}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    continue
                
                # Berechne die gewünschten Properties
                sample_properties = {}
                
                try:
                    print(f"Target keys for calculation: {target_keys}")
                    for key in target_keys:
                        if key == "mu":
                            try:
                                dip_x, dip_y, dip_z = wave_function.variable("SCF DIPOLE")[0:3]
                                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                                sample_properties[key] = dipole_moment
                            except Exception as e:
                                print(f"Dipole calculation error for sample {i}: {e}")
                                sample_properties[key] = float('nan')
                        
                        elif key in ["homo", "lumo", "gap"]:
                            try:
                                eps_a = eps_a = wfn.epsilon_a().to_array()
                                LUMO_idx = wfn.nalpha()
                                HOMO_idx = LUMO_idx - 1
                                
                                if key == "homo":
                                    sample_properties[key] = eps_a[HOMO_idx]
                                elif key == "lumo":
                                    sample_properties[key] = eps_a[LUMO_idx]
                                elif key == "gap":
                                    sample_properties[key] = eps_a[LUMO_idx] - eps_a[HOMO_idx]
                            except Exception as e:
                                print(f"Orbital energy calculation error for sample {i}: {e}")
                                sample_properties[key] = float('nan')
                        
                        elif key == "alpha":
                            try:
                                # First, let's check what variables are actually available
                                available_vars = psi4.core.variables()
                                
                                # Look for polarizability variables
                                polar_vars = []
                                for var_name in available_vars:
                                    if 'POLARIZABILITY' in var_name:
                                        polar_vars.append(var_name)
                                
                                print(f"Sample {i} - Available polarizability variables: {polar_vars}")
                                
                                # Try different possible variable naming patterns
                                alpha_xx = None
                                alpha_yy = None
                                alpha_zz = None
                                
                                # Common patterns for polarizability variables
                                possible_patterns = [
                                    "SCF POLARIZABILITY XX",  # Most likely
                                    "SCF DIPOLE POLARIZABILITY XX",  # Alternative
                                    "POLARIZABILITY XX"  # Another possibility
                                ]
                                
                                for pattern in possible_patterns:
                                    try:
                                        if pattern in available_vars:
                                            alpha_xx = available_vars[pattern]
                                            alpha_yy = available_vars[pattern.replace("XX", "YY")]
                                            alpha_zz = available_vars[pattern.replace("XX", "ZZ")]
                                            print(f"Found polarizability using pattern: {pattern}")
                                            break
                                    except:
                                        continue
                                
                                # If we couldn't find the variables through patterns, try direct access
                                if alpha_xx is None:
                                    try:
                                        # Method 1: Try to get from wfn
                                        polar_tensor = wfn.polarizability()
                                        if polar_tensor is not None:
                                            # The polarizability tensor is typically a 3x3 matrix
                                            alpha_xx = polar_tensor[0][0]
                                            alpha_yy = polar_tensor[1][1]
                                            alpha_zz = polar_tensor[2][2]
                                            print(f"Got polarizability from wfn.polarizability()")
                                    except:
                                        pass
                                
                                if alpha_xx is None:
                                    # Last resort: try to compute it from the response
                                    try:
                                        # Some Psi4 versions store it in the wavefunction
                                        alpha_xx = wfn.scalar_field("POLARIZABILITY_XX")
                                        alpha_yy = wfn.scalar_field("POLARIZABILITY_YY")
                                        alpha_zz = wfn.scalar_field("POLARIZABILITY_ZZ")
                                        print(f"Got polarizability from wfn.scalar_field()")
                                    except:
                                        pass
                                
                                if alpha_xx is not None and alpha_yy is not None and alpha_zz is not None:
                                    alpha_iso = (alpha_xx + alpha_yy + alpha_zz) / 3.0  # Bohr^3
                                    print(f"Sample {i} - Isotropic Polarizability (Bohr^3): {alpha_iso:.6f}")
                                    sample_properties[key] = float(alpha_iso)
                                else:
                                    print(f"Warning: Could not extract polarizability for sample {i}")
                                    # Try to get it as a scalar property directly
                                    try:
                                        # Psi4 might store isotropic polarizability in a different variable
                                        if "SCF POLARIZABILITY" in available_vars:
                                            alpha_iso = available_vars["SCF POLARIZABILITY"]
                                            sample_properties[key] = float(alpha_iso)
                                            print(f"Got isotropic polarizability directly: {alpha_iso:.6f}")
                                        elif "SCF TOTAL POLARIZABILITY" in available_vars:
                                            alpha_iso = available_vars["SCF TOTAL POLARIZABILITY"]
                                            sample_properties[key] = float(alpha_iso)
                                            print(f"Got total polarizability directly: {alpha_iso:.6f}")
                                        else:
                                            sample_properties[key] = float("nan")
                                    except:
                                        sample_properties[key] = float("nan")
                                                
                            except Exception as e:
                                print(f"Polarizability calculation error for sample {i}: {e}")
                                sample_properties[key] = float("nan")
                                        
                    # Füge berechnete Properties zu den Listen hinzu
                    # WICHTIG: Normalisiere die Properties damit sie mit condition_values vergleichbar sind
                    for key in target_keys:
                        if key in sample_properties and not math.isnan(sample_properties[key]):
                            normalized_value = self._normalize_property(key, sample_properties[key])
                            computed_properties[key].append(normalized_value)
                        elif key in sample_properties:
                            computed_properties[key].append(sample_properties[key])
                    
                    self.num_valid_molecules += 1
                    
                except Exception as e:
                    print(f"Property calculation error for sample {i}: {e}")
                
                finally:
                    # Aufräumen
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
            
            if len(true_properties) == 0:
                print("WARNING: No valid molecules passed psi4 calculation. Skipping conditional metrics.")
                # Clean up global scratch directory
                try:
                    shutil.rmtree(psi4_scratch, ignore_errors=True)
                except:
                    pass
                return float('nan'), 0.0
            
            # Konvertiere true_properties zu Tensor
            try:
                if isinstance(true_properties[0], torch.Tensor):
                    true_properties_tensor = torch.stack(true_properties)
                else:
                    true_properties_tensor = torch.tensor(true_properties)
            except Exception as e:
                print(f"Error converting true_properties to tensor: {e}")
                return float('nan'), 0.0
            
            # Berechne Metriken
            mae_results = {}
            cond_val_results = {}
            
            for idx, key in enumerate(target_keys):
                if key in computed_properties and len(computed_properties[key]) > 0:
                    # Filter NaN Werte
                    computed_values = [v for v in computed_properties[key] if not math.isnan(v)]
                    if len(computed_values) == 0:
                        print(f"No valid {key} values computed")
                        continue
                        
                    computed_tensor = torch.FloatTensor(computed_values)
                    
                    # Extrahiere entsprechende true values
                    if true_properties_tensor.dim() == 1:
                        # Nur ein Target
                        true_values = true_properties_tensor[:len(computed_values)]
                    else:
                        # Mehrere Targets
                        true_values = true_properties_tensor[:len(computed_values), idx]
                    
                    # Berechne MAE
                    try:
                        mae = torch.mean(torch.abs(computed_tensor - true_values))
                        mae_results[f"mae_{key}"] = mae.item()
                    except Exception as e:
                        print(f"MAE calculation error for {key}: {e}")
                        mae_results[f"mae_{key}"] = float('nan')
                    
                    # Berechne conditional validity
                    thresholds = {
                        "mu": 0.1,      # Debye
                        "homo": 0.01,   # Hartree
                        "lumo": 0.01,   # Hartree
                        "gap": 0.02,    # Hartree
                        "alpha": 2.0,   # Bohr^3
                    }
                    
                    threshold = thresholds.get(key, 0.1)
                    try:
                        cond_val = (torch.abs(computed_tensor - true_values) < threshold).float().mean()
                        cond_val_results[f"cond_val_{key}"] = cond_val.item()
                    except Exception as e:
                        print(f"Conditional validity calculation error for {key}: {e}")
                        cond_val_results[f"cond_val_{key}"] = 0.0
            
            # Für backward compatibility
            if len(target_keys) == 1:
                key = target_keys[0]
                # Clean up global scratch directory
                try:
                    shutil.rmtree(psi4_scratch, ignore_errors=True)
                except:
                    pass
                
                if key in mae_results:
                    return mae_results[f"mae_{key}"], self.num_valid_molecules / max(self.num_total, 1)
                else:
                    return float('nan'), self.num_valid_molecules / max(self.num_total, 1)
            else:
                # Für multiple Targets
                valid_mae = [v for v in mae_results.values() if not math.isnan(v)]
                valid_cond = [v for v in cond_val_results.values() if not math.isnan(v)]
                
                avg_mae = sum(valid_mae) / len(valid_mae) if valid_mae else float('nan')
                avg_cond_val = sum(valid_cond) / len(valid_cond) if valid_cond else float('nan')
                
                result_dict = {
                    "mae_mean": avg_mae,
                    "cond_val_mean": avg_cond_val,
                    "num_valid": self.num_valid_molecules,
                    "num_total": self.num_total,
                    "valid_ratio": self.num_valid_molecules / max(self.num_total, 1),
                }
                
                result_dict.update(mae_results)
                result_dict.update(cond_val_results)
                
                # Clean up global scratch directory
                try:
                    shutil.rmtree(psi4_scratch, ignore_errors=True)
                except:
                    pass
                
                return result_dict, self.num_valid_molecules / max(self.num_total, 1)

    def evaluate(self, generated, input_properties=None, test=False):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        valid, validity, num_components, all_smiles = self.compute_validity(generated)

        if test:
            with open(r"final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print("All smiles saved")

            print(all_smiles)
            df = pd.DataFrame(all_smiles, columns=["SMILES"])
            df.to_csv("final_smiles.csv", index=False)
            print("All SMILES saved to CSV")

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(
            f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%"
        )
        target_keys = []
        if hasattr(self, 'target_keys'):
            target_keys = self.target_keys
        elif self.args is not None and hasattr(self.args, 'general'):
            dynamic = getattr(self.args.general, 'dynamic', None)
            target = getattr(self.args.general, 'target', None)
            
            # Handle dynamic parameter
            if dynamic is not None:
                if isinstance(dynamic, bool) and dynamic:
                    # dynamic=true -> use target
                    if isinstance(target, str):
                        separators = [' ', ',', ';', ':', '|']
                        separator = ' '
                        for sep in separators:
                            if sep in target:
                                separator = sep
                                break
                        target_keys = [k.strip() for k in target.split(separator) if k.strip()]
                    elif isinstance(target, (list, omegaconf.ListConfig)):
                        target_keys = [str(k).strip() for k in target if k]
            
            # If no dynamic keys, check legacy target
            if not target_keys and target:
                if target == "both":
                    target_keys = ["mu", "homo"]
                elif target in ["mu", "homo"]:
                    target_keys = [target]
        
        print(f"DEBUG evaluate: target_keys = {target_keys}")
        
        cond_mae = cond_val = -1.0
        cond_mae_dict = None
        cond_valid_ratio = 0.0

        if input_properties is not None and len(target_keys) > 0:
            cond_metrics, cond_valid_ratio = self.cond_sample_metric(generated, input_properties)
            
            if isinstance(cond_metrics, dict):
                cond_mae_dict = cond_metrics
                # Extrahiere den mean für backward compatibility
                cond_mae = cond_metrics.get("mae_mean", -1.0)
                cond_val = cond_metrics.get("cond_val_mean", -1.0)
            else:
                cond_mae = cond_metrics
                cond_val = cond_valid_ratio  
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(
                f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                )
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return (
            [validity, relaxed_validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
            cond_mae_dict if cond_mae_dict is not None else [cond_mae, cond_val],
        )


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def mol2smilesWithNoSanitize(mol):
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def build_molecule_with_partial_charges(
    atom_types, edge_types, atom_decoder, verbose=False
):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [
            (s, len(s)) for s in sm.split(".")
        ]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


if __name__ == "__main__":
    smiles_mol = "C1CCC1"
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

use_rdkit = True


def check_stability(
    atom_types, edge_types, dataset_info, debug=False, atom_decoder=None
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)

def format_final_metrics_report(dic, target_keys=None):
    def pct(x):
        return "nan" if x != x else f"{x*100:6.2f}%"

    def val(x):
        return "nan" if x != x else f"{x:8.4f}"

    lines = []
    lines.append("=" * 80)
    lines.append("FINAL MOLECULAR METRICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("📊 BASIC QUALITY METRICS:")
    lines.append(f"   • Validity:             {pct(dic['Validity'])}")
    lines.append(f"   • Relaxed Validity:     {pct(dic['Relaxed_Validity'])}")
    lines.append(f"   • Uniqueness:           {pct(dic['Uniqueness'])}")
    lines.append(f"   • Novelty:              {pct(dic['Novelty'])}")
    lines.append("")
    lines.append("📏 MOLECULE SIZE DISTRIBUTION:")
    lines.append(f"   • Average components:   {dic['nc_mu']:.2f}")
    lines.append(f"   • Max components:       {dic['nc_max']:.2f}")
    lines.append("")
    lines.append(f"🎯 CONDITIONAL GENERATION METRICS ({dic['target_type']}):")
    lines.append("")
    lines.append("   Target-specific metrics:")

    # If explicit target_keys provided, render them in order.
    if target_keys:
        units = {
            "mu": "Debye",
            "homo": "Hartree",
            "lumo": "Hartree",
            "gap": "Hartree",
            "alpha": "Bohr³",
            "r2": "Bohr²",
            "zpve": "Hartree",
        }
        for key in target_keys:
            mae_key = f"cond_mae_{key}"
            val_key = f"cond_val_{key}"
            mae_val = dic.get(mae_key, dic.get("cond_mae", float("nan")))
            cond_val = dic.get(val_key, dic.get("cond_val", float("nan")))
            unit = units.get(key, "")
            if mae_val != -1.0 and not (isinstance(mae_val, float) and mae_val != mae_val):
                lines.append(f"   • {key.upper():10} MAE: {val(mae_val)} {unit}")
            if cond_val != -1.0 and not (isinstance(cond_val, float) and cond_val != cond_val):
                lines.append(f"   • {key.upper():10} Validity: {pct(cond_val)}")
    else:
        # Backwards compatible rendering for MU/HOMO
        if "cond_mae_mu" in dic or "cond_val_mu" in dic:
            lines.append(
                f"   • MU         MAE:      {val(dic.get('cond_mae_mu', float('nan')))} Debye"
            )
            lines.append(
                f"   • MU         Validity: {pct(dic.get('cond_val_mu', float('nan')))}"
            )

        if "cond_mae_homo" in dic:
            lines.append(f"   • HOMO       MAE:      {val(dic['cond_mae_homo'])} Hartree")
            lines.append(f"   • HOMO       Validity: {pct(dic['cond_val_homo'])}")

        lines.append(
            f"   • MAE:                 {val(dic.get('cond_mae', float('nan')))}"
        )
        lines.append(
            f"   • Conditional Val:     {pct(dic.get('cond_val', float('nan')))}"
        )
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)

def compute_molecular_metrics(
    molecule_list,
    train_smiles,
    dataset_info,
    labels,
    args=None,
    test=False,
    epoch=None,
    name=None,
    val_counter=None,
):
    """molecule_list: list of (atom_types, edge_types) pairs"""
    
    # 1. Stabilitätsanalyse (wenn Wasserstoffatome enthalten sind)
    if not dataset_info.remove_h:
        print(f"Analyzing molecule stability...")
        
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)
        
        for i, mol in tqdm(
            enumerate(molecule_list), desc="Stability computation progress"
        ):
            atom_types, edge_types = mol
            
            validity_results = check_stability(atom_types, edge_types, dataset_info)
            
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
        
        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {"mol_stable": -1, "atm_stable": -1}
    
    # 2. Target-Erkennung für dynamische Properties
    target_keys = []
    target_type = "unknown"
    
    if args is not None and hasattr(args, 'general'):
        # Get dynamic and target parameters
        dynamic_param = getattr(args.general, 'dynamic', None)
        target_param = getattr(args.general, 'target', None)
        
        print(f"DEBUG compute_molecular_metrics: dynamic={dynamic_param} (type: {type(dynamic_param)}), target={target_param}")
        
        # Handle dynamic parameter (can be bool or string)
        if dynamic_param is not None:
            if isinstance(dynamic_param, bool) and dynamic_param:
                # dynamic=true -> use target string
                if isinstance(target_param, str):
                    # Support different separators
                    separators = [' ', ',', ';', ':', '|']
                    separator = ' '
                    for sep in separators:
                        if sep in target_param:
                            separator = sep
                            break
                    target_keys = [k.strip() for k in target_param.split(separator) if k.strip()]
                    target_type = f"dynamic_{'_'.join(target_keys)}"
                elif isinstance(target_param, (list, omegaconf.ListConfig)):
                    # target is a list from Hydra
                    target_keys = [str(k).strip() for k in target_param if k]
                    target_type = f"dynamic_{'_'.join(target_keys)}"
                else:
                    print(f"WARNING: dynamic=true but target is not string/list: {target_param}")
        
        # If no dynamic keys found, check legacy target
        if not target_keys and target_param:
            target_type = target_param
            if target_param == "both":
                target_keys = ["mu", "homo"]
            elif target_param in ["mu", "homo"]:
                target_keys = [target_param]
    
    print(f"DEBUG: Final target_keys: {target_keys}, target_type: {target_type}")
    
    # 3. RDKit Metriken berechnen
    metrics = BasicMolecularMetrics(dataset_info, train_smiles, args)
    rdkit_metrics = metrics.evaluate(molecule_list, labels, test)
    all_smiles = rdkit_metrics[-2]
    
    # 4. Basis-Metriken Dictionary
    nc = rdkit_metrics[-3]
    dic = {
        "Validity": rdkit_metrics[0][0],
        "Relaxed_Validity": rdkit_metrics[0][1],
        "Uniqueness": rdkit_metrics[0][2],
        "Novelty": rdkit_metrics[0][3],
        "nc_max": nc["nc_max"],
        "nc_mu": nc["nc_mu"],
        "target_type": target_type,
    }
    # Defer formatting & writing the final report until AFTER
    # conditional metrics are merged into `dic` so the file
    # contains any dynamic targets (e.g., `gap`, `lumo`).
    # 5. Konditionale Metriken verarbeiten
    cond_metrics = rdkit_metrics[-1]
    
    if isinstance(cond_metrics, dict):
        # Neue Format: dynamische oder multiple Targets
        print("\n" + "="*60)
        print("DYNAMIC TARGET METRICS")
        print("="*60)
        
        # Füge alle berechneten Metriken hinzu
        dic.update(cond_metrics)
        
        # Für backward compatibility
        dic["cond_mae"] = cond_metrics.get("mae_mean", -1.0)
        dic["cond_val"] = cond_metrics.get("cond_val_mean", -1.0)
        
        # Zusätzliche Informationen
        dic["num_valid_molecules"] = cond_metrics.get("num_valid", 0)
        dic["valid_ratio"] = cond_metrics.get("valid_ratio", 0.0)
        
        # Dynamische Targets einzeln ausgeben
        for key in target_keys:
            mae_key = f"mae_{key}"
            val_key = f"cond_val_{key}"
            if mae_key in cond_metrics:
                dic[f"cond_mae_{key}"] = cond_metrics[mae_key]
            if val_key in cond_metrics:
                dic[f"cond_val_{key}"] = cond_metrics[val_key]
                
    elif isinstance(cond_metrics, (list, tuple)) and len(cond_metrics) >= 2:
        # Altes Format: single target
        dic.update({
            "cond_mae": cond_metrics[0],
            "cond_val": cond_metrics[1],
        })
        
        # Für spezifische Targets
        if target_type == "mu":
            dic["cond_mae_mu"] = cond_metrics[0]
            dic["cond_val_mu"] = cond_metrics[1]
        elif target_type == "homo":
            dic["cond_mae_homo"] = cond_metrics[0]
            dic["cond_val_homo"] = cond_metrics[1]
        elif target_type == "both":
            # Bei "both" erwarten wir das neue Dictionary Format
            pass
    
    # After conditional metrics are processed, format and write the report
    report = format_final_metrics_report(dic, target_keys=target_keys)

    # Write report to a deterministic location. If caller provides
    # `name`, `epoch` and `val_counter` (passed from training loop),
    # put the report next to other per-epoch outputs in `graphs/{name}`.
    # Otherwise fall back to `outputs/final_molecular_metrics.txt`.
    try:
        provided_name = name
        provided_epoch = epoch
        provided_val_counter = val_counter
    except Exception:
        provided_name = provided_epoch = provided_val_counter = None

    if provided_name is not None and provided_epoch is not None and provided_val_counter is not None:
        out_dir = os.path.join("graphs", provided_name)
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, f"val_epoch{provided_epoch}_res_{provided_val_counter}_general.txt")
    else:
        out_dir = os.path.join("outputs")
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "final_molecular_metrics.txt")

    with open(report_path, "w") as f:
        f.write(report)

    # 6. FINALE AUSGABE (Console Print)
    print("\n" + "="*80)
    print("FINAL MOLECULAR METRICS REPORT")
    print("="*80)
    
    # Grundlegende Qualitätsmetriken
    print(f"\n📊 BASIC QUALITY METRICS:")
    print(f"   • Validity:           {dic['Validity']*100:6.2f}%")
    print(f"   • Relaxed Validity:   {dic['Relaxed_Validity']*100:6.2f}%")
    print(f"   • Uniqueness:         {dic['Uniqueness']*100:6.2f}%")
    print(f"   • Novelty:            {dic['Novelty']*100:6.2f}%")
    
    # Molekülgrößen-Verteilung
    print(f"\n📏 MOLECULE SIZE DISTRIBUTION:")
    print(f"   • Average components: {dic['nc_mu']:6.2f}")
    print(f"   • Max components:     {dic['nc_max']:6.2f}")
    
    # Stabilitätsmetriken (wenn verfügbar)
    if validity_dict['mol_stable'] != -1:
        print(f"\n⚛️  STABILITY METRICS:")
        print(f"   • Molecule stability:  {validity_dict['mol_stable']*100:6.2f}%")
        print(f"   • Atom stability:      {validity_dict['atm_stable']*100:6.2f}%")
    
    # Konditionale Generierungsmetriken
    print(f"\n🎯 CONDITIONAL GENERATION METRICS ({target_type}):")
    
    if 'num_valid_molecules' in dic:
        print(f"   • Valid molecules:     {dic['num_valid_molecules']}/{len(molecule_list)}")
        print(f"   • Valid ratio:         {dic['valid_ratio']*100:6.2f}%")
    
    # Dynamische Targets ausgeben
    if target_keys:
        print(f"\n   Target-specific metrics:")
        for key in target_keys:
            mae = dic.get(f"cond_mae_{key}", dic.get("cond_mae", -1.0))
            cond_val = dic.get(f"cond_val_{key}", dic.get("cond_val", -1.0))
            
            # Einheiten bestimmen
            units = {
                "mu": "Debye",
                "homo": "Hartree", 
                "lumo": "Hartree",
                "gap": "Hartree",
                "alpha": "Bohr³",
                "r2": "Bohr²",
                "zpve": "Hartree",
            }
            unit = units.get(key, "")
            
            if mae != -1.0:
                print(f"   • {key.upper():10} MAE: {mae:8.4f} {unit}")
            if cond_val != -1.0:
                print(f"   • {key.upper():10} Validity: {cond_val*100:6.2f}%")
    
    # Durchschnittswerte für multiple Targets
    if isinstance(cond_metrics, dict) and 'mae_mean' in cond_metrics:
        print(f"\n   Summary metrics:")
        print(f"   • Average MAE:        {cond_metrics.get('mae_mean', -1.0):8.4f}")
        print(f"   • Average Validity:   {cond_metrics.get('cond_val_mean', -1.0)*100:6.2f}%")
    
    # Single target fallback
    elif 'cond_mae' in dic and dic['cond_mae'] != -1.0:
        print(f"   • MAE:                {dic['cond_mae']:8.4f}")
        print(f"   • Conditional Val:    {dic['cond_val']*100:6.2f}%")
    
    print("\n" + "="*80)
    
    # 7. Optional: SMILES Ausgabe bei Test
    if test and all_smiles:
        print(f"\n💾 Generated SMILES saved (total: {len(all_smiles)})")
        
        # Valid SMILES zählen
        valid_smiles = [s for s in all_smiles if s is not None]
        print(f"   • Valid SMILES: {len(valid_smiles)}")
        
        # Unique SMILES
        if valid_smiles:
            unique_smiles = list(set(valid_smiles))
            print(f"   • Unique SMILES: {len(unique_smiles)}")
    
    # 8. WandB Logging
    if wandb.run:
        # Prepare wandb dictionary
        wandb_dict = {}
        
        # Basic metrics
        wandb_dict.update({
            "metrics/validity": dic["Validity"],
            "metrics/relaxed_validity": dic["Relaxed_Validity"],
            "metrics/uniqueness": dic["Uniqueness"],
            "metrics/novelty": dic["Novelty"],
            "metrics/nc_mu": dic["nc_mu"],
            "metrics/nc_max": dic["nc_max"],
        })
        
        # Stability metrics
        if validity_dict['mol_stable'] != -1:
            wandb_dict.update({
                "stability/mol_stable": validity_dict['mol_stable'],
                "stability/atm_stable": validity_dict['atm_stable'],
            })
        
        # Conditional metrics
        if 'cond_mae' in dic and dic['cond_mae'] != -1.0:
            wandb_dict["conditional/mae"] = dic["cond_mae"]
        
        if 'cond_val' in dic and dic['cond_val'] != -1.0:
            wandb_dict["conditional/validity"] = dic["cond_val"]
        
        # Dynamic target metrics
        for key in target_keys:
            if f"cond_mae_{key}" in dic:
                wandb_dict[f"conditional/mae_{key}"] = dic[f"cond_mae_{key}"]
            if f"cond_val_{key}" in dic:
                wandb_dict[f"conditional/val_{key}"] = dic[f"cond_val_{key}"]
        
        # Additional info
        if 'num_valid_molecules' in dic:
            wandb_dict["conditional/num_valid"] = dic["num_valid_molecules"]
            wandb_dict["conditional/valid_ratio"] = dic["valid_ratio"]
        
        wandb.log(wandb_dict)
        print(f"📈 Metrics logged to WandB")
    
    return validity_dict, rdkit_metrics, all_smiles, dic