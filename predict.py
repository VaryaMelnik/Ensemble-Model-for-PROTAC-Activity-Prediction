import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
import logging

# Suppress warnings to keep stdout clean for REINVENT parsing
warnings.filterwarnings('ignore')

from src.chem import (
    batch_process_2D_descriptors,
    batch_process_morgan, 
    batch_process_pharmacophore, 
    batch_process_chemberta_mtr,
)
from src.trainer import PROTACTrainer

# ==========================================
# FOLDER CONFIGURATION
# ==========================================
CSV_IN_DIR = "data/link_invent_outputs"
CSV_OUT_DIR = "data/link_invent_outputs"
MODEL_BASE_DIR = "models"
EMB_BASE_DIR = "data/processed"
# ==========================================

# Set up logging strictly to stderr so it doesn't break Link-INVENT stdout parsing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr  
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble PROTAC Predictor (Fast)")
    
    # Input options
    parser.add_argument("-smiles", type=str, default=None, help="Semicolon separated SMILES (for Link-INVENT)")
    parser.add_argument("-csv_file", type=str, default=None, help=f"Input CSV filename (looks in {CSV_IN_DIR}/)")
    parser.add_argument("-smiles_col", type=str, default="SMILES", help="Column name for SMILES in the CSV file")
    
    # Optional output for CSV batch mode
    parser.add_argument("-out_file", type=str, default=None, help=f"Output CSV filename (saves to {CSV_OUT_DIR}/)")
    
    # Shortened model and embedding arguments
    parser.add_argument("-model_name", type=str, default="final_model_FULL", help=f"Model folder name inside {MODEL_BASE_DIR}/")
    parser.add_argument("-target", type=str, required=True, help=f"Target name (e.g. 'BRD4' -> looks for {EMB_BASE_DIR}/BRD4_embedding.npy)")
    parser.add_argument("-e3", type=str, required=True, help=f"E3 ligase name (e.g. 'CRBN' -> looks for {EMB_BASE_DIR}/CRBN_embedding.npy)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve full paths
    csv_path = os.path.join(CSV_IN_DIR, args.csv_file) if args.csv_file else None
    out_path = os.path.join(CSV_OUT_DIR, args.out_file) if args.out_file else None
    model_dir = os.path.join(MODEL_BASE_DIR, args.model_name)
    target_emb_path = os.path.join(EMB_BASE_DIR, f"{args.target}_embedding.npy")
    e3_emb_path = os.path.join(EMB_BASE_DIR, f"{args.e3}_embedding.npy")

    # 1. Determine the source of SMILES and caching strategy
    df = None
    should_cache = True
    
    if csv_path:
        logger.info(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if args.smiles_col not in df.columns:
            logger.error(f"Column '{args.smiles_col}' not found in {csv_path}.")
            sys.exit(1)
        smiles_list = df[args.smiles_col].dropna().astype(str).tolist()
    elif args.smiles:
        smiles_list = args.smiles.split(";")
        should_cache = False # Disable cache for Link-INVENT prediction mode to save I/O
    else:
        logger.error("You must provide either -smiles or -csv_file")
        sys.exit(1)

    if not smiles_list:
        logger.error("No valid SMILES provided.")
        sys.exit(1)

    num_mols = len(smiles_list)
    logger.info(f"Processing {num_mols} molecule(s)...")

    # 2. Load Processors & Model
    logger.info(f"Loading processors and meta-ensemble from {model_dir}...")
    
    if os.path.exists(os.path.join(model_dir, "multi_view_processors_FULL.joblib")):
        proc_path = os.path.join(model_dir, "multi_view_processors_FULL.joblib")
    else:
        proc_path = os.path.join(model_dir, "multi_view_processors.joblib")
        
    processors = joblib.load(proc_path)
    trainer = PROTACTrainer.load_models(model_dir) 

    # 3. Extract Raw Chemical Features
    logger.info(f"Extracting raw chemical features (Cache enabled: {should_cache})...")
    X_phar = batch_process_pharmacophore(smiles_list, use_cache=should_cache)
    X_morg = batch_process_morgan(smiles_list, radius=3, n_bits=2048)
    X_svm_d = batch_process_2D_descriptors(smiles_list)
    X_mtr = batch_process_chemberta_mtr(smiles_list, use_cache=should_cache)

    # Convert to numpy arrays explicitly just in case
    X_phar = np.array(X_phar)
    X_morg = np.array(X_morg)
    X_svm_d = np.array(X_svm_d)
    X_mtr = np.array(X_mtr)

    # 4. Load Pre-computed Protein Embeddings Directly from Disk
    logger.info(f"Loading embeddings for Target ({args.target}) and E3 ({args.e3})...")
    t_emb_single = np.load(target_emb_path)
    e_emb_single = np.load(e3_emb_path)
    
    # Tile them to match the batch size of incoming SMILES
    t_tr = np.tile(t_emb_single, (num_mols, 1))
    e_tr = np.tile(e_emb_single, (num_mols, 1))

    # 5. Apply Transformers (PCA, Scalers, etc.)
    logger.info("Applying view-specific transformers (PCA, Scalers)...")
    t_proc = processors['pca_t'].transform(processors['sc_t'].transform(t_tr))
    e_proc = processors['pca_e'].transform(processors['sc_e'].transform(e_tr))
    prot_stack = np.hstack([t_proc, e_proc])
    
    views_infer = {
        "XGBoost": np.hstack([prot_stack, processors['vt_phar'].transform(X_phar)]),
        "RandomForest": np.hstack([prot_stack, processors['sc_m3'].transform(X_morg)]),
        "SVM": np.hstack([prot_stack, processors['sc_svm'].transform(X_svm_d)]),
        "KNN": np.hstack([t_proc, e_proc, processors['pca_mtr'].transform(processors['sc_mtr'].transform(X_mtr))])
    }

    # 6. Predict Active Probability
    logger.info("Executing ensemble predictions...")
    # Capture all three outputs from the meta-learner
    binary_preds, meta_probs, uncertainties = trainer.ensemble_predict(views_infer)

    # 7. Output Handling
    if csv_path and out_path:
        # Save all the rich meta-learner results to the CSV
        df['Active_Class'] = binary_preds
        df['Active_Probability'] = meta_probs
        df['Epistemic_Uncertainty'] = uncertainties
        
        df.to_csv(out_path, index=False)
        logger.info(f"Successfully saved predictions to {out_path}")
    else:
        # Link-INVENT / default stdout requirement (space-separated floats)
        # Link-INVENT only needs the continuous probabilities to guide the reinforcement learning
        logger.info("Passing continuous scores to standard output...")
        print(" ".join(map(str, meta_probs)))

if __name__ == "__main__":
    main()