# src/chem.py

import hashlib
import os
import joblib
import torch
import logging
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("PROTAC_Pipeline.chem")

# Suppress RDKit console spam for invalid SMILES generated during RL
RDLogger.DisableLog('rdApp.*')

# ==========================================
# LAZY-LOADED GLOBALS (Performance Optimization)
# ==========================================
_descriptor_calc = None
_chemberta_tokenizer = None
_chemberta_model = None
_device = None

def _get_descriptor_calculator():
    global _descriptor_calc
    if _descriptor_calc is None:
        nms = [x[0] for x in Descriptors._descList]
        _descriptor_calc = (MoleculeDescriptors.MolecularDescriptorCalculator(nms), len(nms))
    return _descriptor_calc

def _get_chemberta(model_name):
    global _chemberta_tokenizer, _chemberta_model, _device
    if _chemberta_model is None:
        logger.info(f"Loading ChemBERTa model ({model_name}) into memory...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _chemberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _chemberta_model = AutoModel.from_pretrained(model_name).to(_device)
        _chemberta_model.eval()
    return _chemberta_tokenizer, _chemberta_model, _device
# ==========================================

def batch_process_morgan(smiles_list, radius=3, n_bits=2048):
    """Generates Morgan Fingerprints (ECFP)."""
    # Generator is lightweight, fine to instantiate here
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        fps.append(gen.GetFingerprintAsNumPy(mol) if mol else np.zeros(n_bits))
    return np.array(fps).astype(float)

def batch_process_2D_descriptors(smiles_list):
    """Generates a dense profile of ~200 RDKit 2D physicochemical properties."""
    calc, num_descriptors = _get_descriptor_calculator()
    results = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            ds = list(calc.CalcDescriptors(mol))
            results.append(np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            results.append([0.0] * num_descriptors)
    return np.array(results)

def batch_process_pharmacophore(smiles_list, cache_dir="data/processed", use_cache=True):
    """Generates 2D Pharmacophore fingerprints with optional disk caching."""
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        smiles_hash = hashlib.md5("".join(smiles_list).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"pharmacophore_cache_{smiles_hash}.joblib")

        if os.path.exists(cache_path):
            return joblib.load(cache_path)

    factory = Gobbi_Pharm2D.factory
    sig_size = factory.GetSigSize()
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        fps.append(np.array(Generate.Gen2DFingerprint(mol, factory)) if mol else np.zeros(sig_size))
    
    X_phar = np.array(fps).astype(float)
    
    if use_cache:
        joblib.dump(X_phar, cache_path)
        
    return X_phar

def batch_process_chemberta_mtr(smiles_list, model_name="deepchem/ChemBERTa-77M-MTR", 
                               batch_size=32, cache_dir="data/processed", use_cache=True):
    """Generates 768-dim chemical embeddings using the ChemBERTa-2 MTR model."""
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        smiles_hash = hashlib.md5("".join(smiles_list).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"chemberta_mtr_cache_{smiles_hash}.joblib")

        if os.path.exists(cache_path):
            return joblib.load(cache_path)

    tokenizer, model, device = _get_chemberta(model_name)
    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, 
                           truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        all_embeddings.append(batch_mean_embeddings)

    X_mtr = np.vstack(all_embeddings)
    
    if use_cache:
        joblib.dump(X_mtr, cache_path)
        
    return X_mtr