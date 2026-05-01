# src/data_loader.py

import pandas as pd
import numpy as np
import logging
import os
import joblib

logger = logging.getLogger("PROTAC_Pipeline.data_loader")

def load_protac_data(filepath):
    """Loads the CSV file."""
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def precalculate_protein_embeddings(df, embedder, cache_path="data/processed/target+e3_embeddings_qbind_1280.joblib"):
    """
    Generates ESM embeddings for both unique Targets and E3 ligases, 
    then maps them back to the dataframe.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Check for existing cache
    if os.path.exists(cache_path):
        logger.info(f"Loading protein embeddings from cache: {cache_path}")
        embedding_map = joblib.load(cache_path)
    else:
        logger.info("No cache found. Starting fresh precalculation.")
        embedding_map = {}

    # 2. Identify unique proteins across BOTH columns that need embedding
    unique_proteins = pd.concat([df['Target'], df['E3 ligase']]).unique()
    proteins_to_process = [p for p in unique_proteins if p not in embedding_map]

    if proteins_to_process:
        logger.info(f"Processing {len(proteins_to_process)} new unique proteins (Targets/E3s)...")
        for protein in proteins_to_process:
            fasta = embedder.fetch_fasta(protein)
            emb = embedder.get_embedding(fasta)
            
            if emb is not None:
                embedding_map[protein] = emb
            else:
                logger.warning(f"Failed to embed {protein}. Using zero vector.")
                embedding_map[protein] = np.zeros(480) 

        joblib.dump(embedding_map, cache_path)
        logger.info(f"Cache updated and saved to {cache_path}")

    # 3. Map the embeddings back to separate columns in the dataframe
    logger.info("Applying Target and E3 embeddings to dataframe...")
    df['target_embedding'] = df['Target'].map(embedding_map)
    df['e3_embedding'] = df['E3 ligase'].map(embedding_map)
    
    return df