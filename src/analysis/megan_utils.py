from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen


# =============================================================================
# Model & Processing
# =============================================================================

def load_model(checkpoint_path: str | Path):
    from graph_attention_student.torch.megan import Megan
    return Megan.load(str(checkpoint_path))


def load_processing(processing_path: str | Path):
    from visual_graph_datasets.util import dynamic_import
    module = dynamic_import(str(processing_path))
    return module.processing


# =============================================================================
# SMILES → Graph → Embedding
# =============================================================================

def smiles_to_graph(smiles: str, processing) -> dict:
    return processing.process(smiles)


def graph_to_embedding(graph: dict, model) -> tuple[np.ndarray, float]:
    """
    Returns:
        embedding: np.ndarray (D, K)
        prediction: float
    """
    info = model.forward_graph(graph)
    
    prediction = float(info["graph_output"][0])
    embedding = info["graph_embedding"]
    
    return embedding, prediction


def smiles_to_embedding(smiles: str, model, processing):
    graph = smiles_to_graph(smiles, processing)
    return graph_to_embedding(graph, model)

def get_qed(smiles):

    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return QED.qed(mol)


# -----------------------------
# SMARTS detection
# -----------------------------
def detect_oh(mol):

    pattern = Chem.MolFromSmarts("[OX2H]")
    return list(mol.GetSubstructMatches(pattern))


def detect_nh2(mol):

    pattern = Chem.MolFromSmarts("[NX3H2]")
    return list(mol.GetSubstructMatches(pattern))


def detect_p(mol):
    pattern = Chem.MolFromSmarts("[nX2]")
    return list(mol.GetSubstructMatches(pattern))
    
def detect_O(mol):
    pattern = Chem.MolFromSmarts("[#6]=O")
    return list(mol.GetSubstructMatches(pattern))

def match_pattern(smiles):
        pattern = [-100,-100,-100,-100]

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return pattern

        has_nh2 = int(len(detect_nh2(mol)) > 0)
        has_p   = int(len(detect_p(mol)) > 0)
        has_oh  = int(len(detect_oh(mol)) > 0)
        has_O   = int(len(detect_O(mol)) > 0)

        return [has_nh2, has_p, has_oh, has_O]

        
        
# =============================================================================
# Centroids
# =============================================================================

def load_centroids(centroids_path: Path) -> list[dict]:
    with open(centroids_path, "r") as f:
        return json.load(f)


# =============================================================================
# Embedding Handling
# =============================================================================

def extract_embeddings_from_row(row: pd.Series) -> tuple[np.ndarray, int]:
    embedding_cols = [col for col in row.index if col.startswith("embedding_ch")]
    
    if not embedding_cols:
        raise ValueError("No embedding columns found")

    channels = set()
    dims = set()

    for col in embedding_cols:
        parts = col.replace("embedding_ch", "").split("_dim")
        channels.add(int(parts[0]))
        dims.add(int(parts[1]))

    n_channels = max(channels) + 1
    embedding_dim = max(dims) + 1

    embedding = np.zeros((embedding_dim, n_channels))

    for col in embedding_cols:
        parts = col.replace("embedding_ch", "").split("_dim")
        ch = int(parts[0])
        dim = int(parts[1])
        embedding[dim, ch] = row[col]

    return embedding, n_channels


def logp_smiles(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logp = Crippen.MolLogP(mol)
    return  logp

def compute_distances(
    embedding: np.ndarray,
    centroids: list[dict]
) -> list[float]:
    """
    Compute cosine distances between embedding and centroids.

    Args:
        embedding: shape (D, K)
        centroids: list of dicts with keys [index, channel, centroid]

    Returns:
        List of distances ordered by cluster index
    """
    n_channels = embedding.shape[1]
    distances = []

    for cluster_info in centroids:
        channel = cluster_info["channel"]
        centroid = np.array(cluster_info["centroid"])

        if channel >= n_channels:
            distances.append(np.nan)
            continue

        channel_embedding = embedding[:, channel]

        if len(channel_embedding) != len(centroid):
            distances.append(np.nan)
            continue

        dist = cosine(channel_embedding, centroid)
        distances.append(dist)

    return distances


def batch_smiles_to_distances(smiles_list, model, processing, centroids):
    return [
        smiles_to_centroid_distances(s, model, processing, centroids)
        for s in smiles_list
    ]

def smiles_to_centroid_distances(
    smiles: str
) -> dict:
    """

    Returns:
        {
            "smiles": str,
            "prediction": float,
            "distances": list[float]
        }
    """
    centroids = load_centroids("/hkfs/work/workspace_haic/scratch/rx3495-workspace_C/Dataset_Synth2_version2/centroids.json")
    model = load_model("/hkfs/work/workspace_haic/scratch/rx3495-workspace_C/Dataset_Synth2_version2/model.ckpt")
    processing = load_processing("/hkfs/work/workspace_haic/scratch/rx3495-workspace_C/Dataset_Synth2_version2/process.py")
    try:
        embedding, prediction = smiles_to_embedding(smiles, model, processing)
        distances = compute_distances(embedding, centroids)

        return {
            "smiles": smiles,
            "prediction": prediction,
            "distances": distances
        }

    except Exception as e:
        return {
            "smiles": smiles,
            "prediction": None,
            "distances": None
        }