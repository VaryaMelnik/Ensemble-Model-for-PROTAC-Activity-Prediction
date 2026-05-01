import sys
import os
import numpy as np
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import matthews_corrcoef

from src.logger import setup_logger
from src.config import load_config
from src.data_loader import load_protac_data, precalculate_protein_embeddings
from src.bio_esm_qbind import ProteinEmbedder
from src.chem import (
    batch_process_2D_descriptors,
    batch_process_morgan, 
    batch_process_pharmacophore, 
    batch_process_chemberta_mtr,
)
from src.trainer_ukr import PROTACTrainer

RUN_Y_RANDOMIZATION = False
TRAIN_FINAL_MODEL = True

def load_expert_params(config, base_path="configs/tuning_final/"):
    experts = {
        'xgb': 'best_params_xgboost.json', 
        'rf': 'best_params_randomforest.json', 
        'svm': 'best_params_svm.json',
        'knn': 'best_params_knn.json'
    }
    for model_key, filename in experts.items():
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config['models'][model_key].update(json.load(f))
            except Exception: pass
    return config

def build_multi_view_stack(df_train, df_val, df_test, save_dir, logger):
    def extract_raw_features(df):
        smiles = df['Smiles'].tolist()
        # Molecular Descriptors
        X_phar = np.array(batch_process_pharmacophore(smiles))
        X_morg = np.array(batch_process_morgan(smiles, radius=3, n_bits=2048))
        X_svm_d = np.array(batch_process_2D_descriptors(smiles))
        X_mtr = np.array(batch_process_chemberta_mtr(smiles))
        # Protein Embeddings
        X_t = np.stack(df['target_embedding'].values)
        X_e = np.stack(df['e3_embedding'].values)
        return X_phar, X_morg, X_svm_d, X_mtr, X_t, X_e

    # 1. Raw Extraction
    logger.info("Extracting raw molecular and protein features...")
    phar_tr, m3_tr, svm_tr, mtr_tr, t_tr, e_tr = extract_raw_features(df_train)
    phar_vl, m3_vl, svm_vl, mtr_vl, t_vl, e_vl = extract_raw_features(df_val)
    phar_ts, m3_ts, svm_ts, mtr_ts, t_ts, e_ts = extract_raw_features(df_test)

    # 2. Fit Transformers (Train Only) - Following the Tuning Logic
    logger.info("Fitting transformers based on tuning pipeline logic...")
    
    # Protein Pipeline (Scale + PCA)
    sc_t = StandardScaler().fit(t_tr)
    pca_t = PCA(n_components=0.99, random_state=42).fit(sc_t.transform(t_tr))
    
    sc_e = StandardScaler().fit(e_tr)
    pca_e = PCA(n_components=0.99, random_state=42).fit(sc_e.transform(e_tr))

    # View Specific Transformers
    vt_phar = VarianceThreshold(threshold=(.995 * (1 - .995))).fit(phar_tr)
    sc_m3 = StandardScaler().fit(m3_tr)
    sc_svm = StandardScaler().fit(svm_tr)
    
    sc_mtr = StandardScaler().fit(mtr_tr)
    pca_mtr = PCA(n_components=0.99, random_state=42).fit(sc_mtr.transform(mtr_tr))

    def transform_to_views(phar, m3, svm_d, mtr, t_raw, e_raw):
        # Apply Protein Logic
        t_proc = pca_t.transform(sc_t.transform(t_raw))
        e_proc = pca_e.transform(sc_e.transform(e_raw))
        prot_stack = np.hstack([t_proc, e_proc])
        
        # XGBoost View: Protein + Pharmacophore VT
        xgb_view = np.hstack([prot_stack, vt_phar.transform(phar)])
        
        # RF View: Protein + Scaled Morgan
        rf_view  = np.hstack([prot_stack, sc_m3.transform(m3)])
        
        # SVM View: Protein + Scaled 2D Descriptors
        svm_view = np.hstack([prot_stack, sc_svm.transform(svm_d)])
        
        # KNN View: Protein + MTR PCA
        mtr_proc = pca_mtr.transform(sc_mtr.transform(mtr))
        knn_view = np.hstack([t_proc, e_proc, mtr_proc])

        return {
            "XGBoost": xgb_view, 
            "RandomForest": rf_view, 
            "SVM": svm_view, 
            "KNN": knn_view
        }

    # 3. Transform All Splits
    views_train = transform_to_views(phar_tr, m3_tr, svm_tr, mtr_tr, t_tr, e_tr)
    views_val   = transform_to_views(phar_vl, m3_vl, svm_vl, mtr_vl, t_vl, e_vl)
    views_test  = transform_to_views(phar_ts, m3_ts, svm_ts, mtr_ts, t_ts, e_ts)

    # 4. Save Transformers for Inference
    joblib.dump({
        'sc_t': sc_t, 'pca_t': pca_t, 
        'sc_e': sc_e, 'pca_e': pca_e,
        'vt_phar': vt_phar, 
        'sc_m3': sc_m3, 
        'sc_svm': sc_svm, 
        'sc_mtr': sc_mtr, 'pca_mtr': pca_mtr
    }, os.path.join(save_dir, "multi_view_processors.joblib"))

    return views_train, views_val, views_test

def build_full_multi_view_stack(df_full, save_dir, logger):
    """Fits transformers and generates views on the ENTIRE dataset for final production."""
    def extract_raw_features(df):
        smiles = df['Smiles'].tolist()
        X_phar = np.array(batch_process_pharmacophore(smiles))
        X_morg = np.array(batch_process_morgan(smiles, radius=3, n_bits=2048))
        X_svm_d = np.array(batch_process_2D_descriptors(smiles))
        X_mtr = np.array(batch_process_chemberta_mtr(smiles))
        X_t = np.stack(df['target_embedding'].values)
        X_e = np.stack(df['e3_embedding'].values)
        return X_phar, X_morg, X_svm_d, X_mtr, X_t, X_e

    logger.info("Extracting raw features for FINAL FULL model...")
    phar_f, m3_f, svm_f, mtr_f, t_f, e_f = extract_raw_features(df_full)

    logger.info("Fitting FINAL transformers on the entire dataset...")
    
    # Protein Pipeline
    sc_t = StandardScaler().fit(t_f)
    pca_t = PCA(n_components=0.99, random_state=42).fit(sc_t.transform(t_f))
    
    sc_e = StandardScaler().fit(e_f)
    pca_e = PCA(n_components=0.99, random_state=42).fit(sc_e.transform(e_f))

    # View Specific Transformers
    vt_phar = VarianceThreshold(threshold=(.995 * (1 - .995))).fit(phar_f)
    sc_m3 = StandardScaler().fit(m3_f)
    sc_svm = StandardScaler().fit(svm_f)
    
    sc_mtr = StandardScaler().fit(mtr_f)
    pca_mtr = PCA(n_components=0.99, random_state=42).fit(sc_mtr.transform(mtr_f))

    # Apply Transformations
    t_proc = pca_t.transform(sc_t.transform(t_f))
    e_proc = pca_e.transform(sc_e.transform(e_f))
    prot_stack = np.hstack([t_proc, e_proc])
    
    xgb_view = np.hstack([prot_stack, vt_phar.transform(phar_f)])
    rf_view  = np.hstack([prot_stack, sc_m3.transform(m3_f)])
    svm_view = np.hstack([prot_stack, sc_svm.transform(svm_f)])
    mtr_proc = pca_mtr.transform(sc_mtr.transform(mtr_f))
    knn_view = np.hstack([t_proc, e_proc, mtr_proc])

    views_full = {
        "XGBoost": xgb_view, 
        "RandomForest": rf_view, 
        "SVM": svm_view, 
        "KNN": knn_view
    }

    # Save Transformers for Production Inference
    joblib.dump({
        'sc_t': sc_t, 'pca_t': pca_t, 
        'sc_e': sc_e, 'pca_e': pca_e,
        'vt_phar': vt_phar, 
        'sc_m3': sc_m3, 
        'sc_svm': sc_svm, 
        'sc_mtr': sc_mtr, 'pca_mtr': pca_mtr
    }, os.path.join(save_dir, "multi_view_processors_FULL.joblib"))

    return views_full

def log_expert_correlations(trainer, X_test_views, y_test, save_path, logger):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']
    plt.rcParams['axes.unicode_minus'] = False 

    preds = {}
    for name, model in trainer.models_list.items():
        try:
            preds[name] = model.predict_proba(X_test_views[name])[:, 1]
        except Exception: pass
        
    if len(preds) < 2: return
    
    df_preds = pd.DataFrame(preds)
    df_preds['True_Label'] = y_test
    
    df_actives = df_preds[df_preds['True_Label'] == 1].drop(columns=['True_Label'])
    df_inactives = df_preds[df_preds['True_Label'] == 0].drop(columns=['True_Label'])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    heatmap_kwargs = {
        'annot': True,
        'cmap': "RdBu_r",
        'vmin': -1,
        'vmax': 1,
        'center': 0,
        'annot_kws': {"size": 16, "fontweight": "bold"}
    }

    # Plot heatmaps
    sns.heatmap(df_inactives.corr(), ax=axes[0], **heatmap_kwargs)
    sns.heatmap(df_actives.corr(), ax=axes[1], **heatmap_kwargs)

    cbar_0 = axes[0].collections[0].colorbar
    cbar_0.ax.tick_params(labelsize=16) 

    axes[0].set_title(f"Correlation on INACTIVES (n={len(df_inactives)})", 
                      fontsize=20, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    
    for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        label.set_fontsize(16)
        label.set_fontweight('bold')

    cbar_1 = axes[1].collections[0].colorbar
    cbar_1.ax.tick_params(labelsize=16) 

    axes[1].set_title(f"Correlation on ACTIVES (n={len(df_actives)})", 
                      fontsize=20, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    
    for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        label.set_fontsize(16)
        label.set_fontweight('bold')

    plt.suptitle("Expert Probability Correlation: Imbalance Breakdown", 
                 fontsize=24, y=0.98, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    plt.savefig(save_path, dpi=400)
    plt.close()
    
    logger.info("Saved split expert correlation heatmaps.")   

def main():
    config = load_config("configs/config.yaml")
    config = load_expert_params(config)
    input_data_path = "data/raw/protac_cleaned.csv" 
    suffix = "_y_rand" if RUN_Y_RANDOMIZATION else "_y_true"
    
    # Define our two potential paths
    split_model_dir = f"models/meta_model{suffix}/"
    full_model_dir = "models/final_model_FULL/"
    
    # Dynamically select the active directory so the logger goes to the right place
    active_dir = full_model_dir if TRAIN_FINAL_MODEL else split_model_dir
    os.makedirs(active_dir, exist_ok=True)
    
    # Initialize logger ONCE into the correct active directory
    logger = setup_logger(os.path.join(active_dir, "pipeline.log"))
    
    logger.info("=== STARTING HETEROGENEOUS MULTI-VIEW STACKING ===")

    try:
        df_final = load_protac_data(input_data_path)
        embedder = ProteinEmbedder(model_name=config['bio_features']['esm_model'])
        
        # ---------------------------------------------------------
        # PHASE 1: STANDARD EVALUATION (Train/Val/Test)
        # ---------------------------------------------------------
        if not TRAIN_FINAL_MODEL:
            df_train, df_temp = train_test_split(df_final, test_size=0.20, random_state=42, stratify=df_final['Activity_Label'])
            df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp['Activity_Label'])

            df_train = precalculate_protein_embeddings(df_train, embedder)
            df_val = precalculate_protein_embeddings(df_val, embedder)
            df_test = precalculate_protein_embeddings(df_test, embedder)

            views_train, views_val, views_test = build_multi_view_stack(df_train, df_val, df_test, active_dir, logger)
            y_train, y_val, y_test = df_train['Activity_Label'].values, df_val['Activity_Label'].values, df_test['Activity_Label'].values

            if RUN_Y_RANDOMIZATION:
                logger.warning("Applying Y-Randomization...")
                y_train = shuffle(y_train, random_state=123)

            input_dims_dict = {name: view.shape[1] for name, view in views_train.items()}
            trainer = PROTACTrainer(input_dims_dict=input_dims_dict, config=config)
            
            trainer.train_all(views_train, y_train, views_val, y_val, retrain_meta=True)
            
            metrics = trainer.evaluate(views_test, y_test)
            logger.info(f"FINAL TEST MCC: {metrics['mcc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            
            results_df = pd.DataFrame({
                'y_true': metrics['ground_truth'],
                'y_prob': metrics['probabilities']
            })
            
            pred_filename = "test_preds_meta.csv" if not RUN_Y_RANDOMIZATION else "test_preds_averaging_rand.csv"
            results_df.to_csv(os.path.join(active_dir, pred_filename), index=False)
            logger.info(f"Saved test predictions to {pred_filename}")

            for expert_name, model in trainer.models_list.items():
                expert_tune_probs = model.predict_proba(
                    views_val[expert_name]
                )[trainer.idx_tune, 1]

                # Той самий пошук порогу, що і в trainer
                best_mcc, best_t = -1, 0.5
                for t in np.linspace(0.05, 0.95, 91):
                    mcc = matthews_corrcoef(
                        trainer.y_tune, (expert_tune_probs >= t).astype(int)
                    )
                    if mcc > best_mcc:
                        best_mcc, best_t = mcc, t

                expert_test_probs = model.predict_proba(views_test[expert_name])[:, 1]
                expert_test_preds = (expert_test_probs >= best_t).astype(int)

                fname = f"test_preds_{expert_name.lower()}.csv"
                pd.DataFrame({
                    'y_true': y_test,
                    'y_prob': expert_test_probs,
                    'y_pred': expert_test_preds,
                    'threshold': best_t,
                    'mcc': matthews_corrcoef(y_test, expert_test_preds)
                }).to_csv(os.path.join(active_dir, fname), index=False)

            with open(os.path.join(active_dir, "test_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
                
            trainer.plot_expert_correlations(views_test=views_test, y_test=y_test, output_path=os.path.join(active_dir, "expert_correlation_split.png"))
            trainer.plot_confusion_matrix(metrics, os.path.join(active_dir, "test_confusion_matrix.png"))
            trainer.plot_meta_importance(os.path.join(active_dir, "meta_importance.png"))
            trainer.plot_epistemic_uncertainty(y_test, metrics, os.path.join(active_dir, "epistemic_uncertainty.png"))
            trainer.run_exhaustive_upset_study(views_val, y_val, views_test, y_test, os.path.join(active_dir, "ablation_upset_matrix.png"), logger)
            trainer.save_models(active_dir)

        # ---------------------------------------------------------
        # PHASE 2: PRODUCTION DEPLOYMENT (Train on Everything)
        # ---------------------------------------------------------
        else:
            logger.info("TRAIN_FINAL_MODEL is True. Training ultimate production model on 100% of data.")
            
            df_full = precalculate_protein_embeddings(df_final, embedder)
            y_full = df_full['Activity_Label'].values
            
            views_full = build_full_multi_view_stack(df_full, active_dir, logger)
            
            # Load the trainer from Phase 1 so it already has the optimized Meta-Learner & Threshold!
            logger.info(f"Loading optimized Meta-Learner and Threshold from {split_model_dir}")
            trainer_full = PROTACTrainer.load_models(split_model_dir)
            
            # Train ONLY the base experts on the full data
            logger.info("Retraining base experts on full dataset. Meta-learner remains locked.")
            trainer_full.train_all(views_train=views_full, y_train=y_full, retrain_meta=False)
            
            # Save strictly to the active directory (the full_model folder)
            trainer_full.save_models(active_dir)
            logger.info(f"Successfully trained and saved the FULL production model to {active_dir}")

    except Exception as e:
        logger.critical("Training failed:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()