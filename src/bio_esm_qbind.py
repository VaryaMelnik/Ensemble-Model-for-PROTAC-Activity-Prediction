# src/bio_esm_qbind.py

import torch
import numpy as np
import logging
import pandas as pd
from Bio import Entrez
from transformers import AutoTokenizer, EsmModel
from peft import PeftModel

logger = logging.getLogger("PROTAC_Pipeline.bio_esm")

class ProteinEmbedder:
    def __init__(self, model_name="AmelieSchreiber/esm2_t33_650M_qlora_binding_16M", email="melnik.varvara@gmail.com"):
        Entrez.email = email
        logger.info(f"Initializing High-Res 1280-dim QBind: {model_name}")
        
        base_model_path = "facebook/esm2_t33_650M_UR50D"
        
        # 1. Load Tokenizer and Base Model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = EsmModel.from_pretrained(base_model_path, output_hidden_states=True)
        
        # 2. Attach QBind LoRA weights
        try:
            self.model = PeftModel.from_pretrained(base_model, model_name)
            logger.info("1280-dim QBind adapter integrated successfully.")
        except Exception as e:
            logger.warning(f"Could not load adapter: {e}. Falling back to Vanilla ESM-2.")
            self.model = base_model

        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fetch_fasta(self, protein_name):
        logger.info(f"Fetching FASTA for protein: {protein_name}")
        try:
            handle = Entrez.esearch(db="protein", term=protein_name, retmax=1)
            record = Entrez.read(handle)
            
            if not record['IdList'] and " " in protein_name:
                fallback_name = protein_name.split()[0]
                logger.warning(f"No match for '{protein_name}', trying fallback: '{fallback_name}'")
                handle = Entrez.esearch(db="protein", term=fallback_name, retmax=1)
                record = Entrez.read(handle)

            if not record['IdList']:
                logger.error(f"No NCBI ID found for {protein_name} or its fallback.")
                return None
            
            id = record['IdList'][0]
            fetch_handle = Entrez.efetch(db="protein", id=id, rettype="fasta", retmode="text")
            return "".join(fetch_handle.read().split("\n")[1:])
        except Exception as e:
            logger.error(f"NCBI Fetch Error for {protein_name}: {e}")
            return None

    def get_embedding(self, sequence):
        if not sequence: return None
        
        # Equivalent to batch_converter
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state is the final layer output (1280-dim)
            last_hidden_state = outputs.last_hidden_state 
            
            # Mean pooling (excluding CLS/EOS)
            mask = inputs['attention_mask'][0, 1:-1].bool()
            embeddings = last_hidden_state[0, 1:-1]
            return embeddings[mask].mean(0).cpu().numpy()