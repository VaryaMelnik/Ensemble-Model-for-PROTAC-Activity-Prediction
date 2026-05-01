import logging
import joblib
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec   
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    precision_score, recall_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split

from src.models import get_xgb_model, get_rf_model, get_svm_model, get_knn_model

logger = logging.getLogger("PROTAC_Pipeline.trainer")

class PROTACTrainer:
    def __init__(self, input_dims_dict=None, config=None):
        if input_dims_dict is None or config is None:
            return
            
        self.config = config
        self.ablation_results = None 
        
        logger.info(f"Initializing Hard Voting Ensemble (Baseline). Dims: {input_dims_dict}")
        
        self.models_list = {
            "XGBoost": get_xgb_model(config['models'].get('xgb')),
            "RandomForest": get_rf_model(config['models'].get('rf')),
            "SVM": get_svm_model(config['models'].get('svm')),
            "KNN": get_knn_model(config['models'].get('knn'))
        }
        
        # Hard voting uses an integer threshold (e.g., requires 2 out of 4 votes)
        self.optimized_vote_threshold = 2
        
        self.idx_tune = None
        self.y_tune = None

    def train_all(self, views_train, y_train, views_val=None, y_val=None, retrain_meta=True):
        """Trains experts and optimizes the integer vote threshold."""
        logger.info("Stage 1: Training Heterogeneous Experts...")
        
        if retrain_meta:
            # We split validation to find an unbiased vote threshold
            indices = np.arange(len(y_val))
            _, self.idx_tune = train_test_split(
                indices, test_size=0.5, random_state=42, stratify=y_val
            )
            self.y_tune = y_val[self.idx_tune]
            expert_tune_votes = []

        for name, model in self.models_list.items():
            logger.info(f"Training {name} Expert...")
            model.fit(views_train[name], y_train)
                
            if retrain_meta:
                # Get hard predictions (0 or 1) for the tuning set
                val_preds = model.predict(views_val[name])
                expert_tune_votes.append(val_preds[self.idx_tune])

        if not retrain_meta:
            logger.info("Threshold Optimization Skipped: Using pre-optimized vote threshold.")
            return

        # Stage 2: Hard Voting Threshold Optimization
        logger.info("Stage 2: Optimizing Required Votes for Hard Ensemble...")
        
        # Sum the votes across all experts
        X_tune_votes = np.column_stack(expert_tune_votes)
        total_votes_tune = np.sum(X_tune_votes, axis=1)
        
        best_v, best_mcc = 2, -1
        # Test requiring 1, 2, 3, or 4 votes to predict "Active"
        for v in range(1, len(self.models_list) + 1):
            mcc = matthews_corrcoef(self.y_tune, (total_votes_tune >= v).astype(int))
            if mcc > best_mcc: 
                best_mcc, best_v = mcc, v
                
        self.optimized_vote_threshold = int(best_v)
        logger.info(f"Final Optimized Vote Threshold: {self.optimized_vote_threshold} out of {len(self.models_list)} (Val MCC: {best_mcc:.4f})")

    def ensemble_predict(self, views_test):
        expert_votes = []
        expert_probs = []
        
        for name in self.models_list.keys():
            expert_votes.append(self.models_list[name].predict(views_test[name]))
            expert_probs.append(self.models_list[name].predict_proba(views_test[name])[:, 1])
        
        X_votes_test = np.column_stack(expert_votes)
        X_probs_test = np.column_stack(expert_probs)
        
        # Total positive votes for each molecule
        total_votes = np.sum(X_votes_test, axis=1)
        
        # Pseudo-probability (fraction of votes) for ROC-AUC calculations
        pseudo_p = total_votes / len(self.models_list)
        
        # Final discrete prediction based on optimized vote threshold
        preds = (total_votes >= self.optimized_vote_threshold).astype(int)
        
        # Epistemic uncertainty (std dev of raw probabilities) remains relevant for consistency
        uncertainties = np.std(X_probs_test, axis=1)
        
        return preds, pseudo_p, uncertainties

    def evaluate(self, views_test, y_test):
        preds, pseudo_probs, uncertainties = self.ensemble_predict(views_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        precision_pts, recall_pts, _ = precision_recall_curve(y_test, pseudo_probs)
        
        conf_threshold = 0.20
        high_conf_mask = uncertainties < conf_threshold
        coverage = np.mean(high_conf_mask) * 100
        
        high_conf_mcc = matthews_corrcoef(y_test[high_conf_mask], preds[high_conf_mask]) if coverage > 0 else 0.0

        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "mcc": matthews_corrcoef(y_test, preds),
            "roc_auc": roc_auc_score(y_test, pseudo_probs), # Uses vote fractions
            "probabilities": pseudo_probs.tolist(),
            "ground_truth": y_test.tolist(),
            "auprc": auc(recall_pts, precision_pts),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, 
            "selected_threshold": self.optimized_vote_threshold, # Integer votes needed
            "meta_weights": [1.0/len(self.models_list)] * len(self.models_list), # Implicit equal weights
            "epistemic_uncertainties": uncertainties.tolist(),
            "predictions": preds.tolist(),
            "high_confidence_coverage": coverage,
            "high_confidence_mcc": high_conf_mcc
        }

    # --- ABLATION STUDY ---
    def run_ablation_study(self, views_val, y_val, views_test, y_test):
        logger.info("--- RUNNING ENSEMBLE ABLATION STUDY ---")
        expert_names = list(self.models_list.keys())
        
        meta_probs, tune_probs, test_probs = {}, {}, {}
        
        for name, model in self.models_list.items():
            full_val_probs = model.predict_proba(views_val[name])[:, 1]
            meta_probs[name] = full_val_probs[self.idx_meta]
            tune_probs[name] = full_val_probs[self.idx_tune]
            test_probs[name] = model.predict_proba(views_test[name])[:, 1]
                
        # Base Model Metrics
        X_test_base = np.column_stack([test_probs[n] for n in expert_names])
        base_preds = (self.meta_learner.predict_proba(X_test_base)[:, 1] >= self.optimized_threshold).astype(int)
        base_mcc = matthews_corrcoef(y_test, base_preds)
        
        ablation_results = {}
        
        for dropped_expert in expert_names:
            active_experts = [n for n in expert_names if n != dropped_expert]
            
            # Stack using only active experts
            X_meta_ablated = np.column_stack([meta_probs[n] for n in active_experts])
            X_tune_ablated = np.column_stack([tune_probs[n] for n in active_experts])
            X_test_ablated = np.column_stack([test_probs[n] for n in active_experts])
            
            # Retrain simplified Meta-Learner
            temp_meta = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)
            temp_meta.fit(X_meta_ablated, self.y_meta)
            
            # Retune threshold
            temp_tune_p = temp_meta.predict_proba(X_tune_ablated)[:, 1]
            best_t, best_mcc = 0.5, -1
            for t in np.linspace(0.05, 0.95, 91):
                mcc = matthews_corrcoef(self.y_tune, (temp_tune_p >= t).astype(int))
                if mcc > best_mcc: 
                    best_mcc, best_t = mcc, t
                    
            # Evaluate Ablated Model on Test Set
            temp_test_p = temp_meta.predict_proba(X_test_ablated)[:, 1]
            ablated_preds = (temp_test_p >= best_t).astype(int)
            ablated_mcc = matthews_corrcoef(y_test, ablated_preds)
            
            mcc_drop = base_mcc - ablated_mcc
            ablation_results[dropped_expert] = {"mcc": ablated_mcc, "mcc_drop": mcc_drop}
            logger.info(f"Dropped {dropped_expert} | New MCC: {ablated_mcc:.4f} (Drop: {mcc_drop:.4f})")
            
        self.ablation_results = ablation_results
        return ablation_results

    # --- VISUALIZATIONS ---
    def plot_epistemic_uncertainty(self, y_test, metrics, output_path):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']
        plt.rcParams['axes.unicode_minus'] = False 

        uncertainties = np.array(metrics['epistemic_uncertainties'])
        preds = np.array(metrics['predictions'])
        correct_mask = (preds == y_test)
        
        plt.figure(figsize=(12, 8))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        color_correct = '#0571b0' 
        color_incorrect = '#ca0020'
        
        if np.any(correct_mask):
            sns.histplot(uncertainties[correct_mask], color=color_correct, 
                         label='Correct Predictions', kde=True, stat='density', 
                         alpha=0.7, bins=20, edgecolor='black', linewidth=0.6,
                         line_kws={'linewidth': 2, 'color': 'black', 'zorder': 5})
            
        if np.any(~correct_mask):
            sns.histplot(uncertainties[~correct_mask], color=color_incorrect, 
                         label='Incorrect Predictions', kde=True, stat='density', 
                         alpha=0.7, bins=20, edgecolor='black', linewidth=0.6,
                         line_kws={'linewidth': 2, 'color': 'black', 'zorder': 5})
        
        plt.title('Epistemic Uncertainty: Correct vs. Incorrect Predictions', fontsize=20, fontweight='bold')
        plt.xlabel('Expert Disagreement (Standard Deviation of Probabilities)', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 18})
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=400)
        plt.close()

    def plot_ablation_study(self, output_path):
        if not self.ablation_results: return
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']
        plt.rcParams['axes.unicode_minus'] = False 

        experts = list(self.ablation_results.keys())
        drops = [self.ablation_results[e]['mcc_drop'] for e in experts]
        
        plt.figure(figsize=(12, 8))
        
        if len(drops) > 0:
            v_extrema = max(abs(min(drops)), abs(max(drops)))
            if v_extrema == 0: v_extrema = 1.0
            norm = plt.Normalize(vmin=-v_extrema, vmax=v_extrema)
            colors = plt.cm.RdBu_r(norm(drops))
        else:
            colors = []
        
        bars = plt.barh(experts, drops, color=colors, edgecolor='black', linewidth=0.6, alpha=1.0)
        plt.axvline(0, color='black', linewidth=1.2)
        
        plt.title('Ablation Study: Impact of Removing Individual Experts', fontsize=20, fontweight='bold')
        plt.xlabel('Drop in Test MCC (Larger = Expert is more crucial)', fontsize=18)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        for bar, drop in zip(bars, drops):
            width = bar.get_width()
            label_x = width + (0.001 if width >= 0 else -0.001)
            ha = 'left' if width >= 0 else 'right'
            plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                     f'{drop:.4f}', va='center', fontsize=16, fontweight='bold', ha=ha)

        plt.tight_layout()
        plt.savefig(output_path, dpi=400)
        plt.close()

    def plot_confusion_matrix(self, metrics, output_path):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']
        
        cm = [[metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
              [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]]
        
        plt.figure(figsize=(9, 8))
        
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Inactive', 'Active'], 
                    yticklabels=['Inactive', 'Active'],
                    annot_kws={"size": 18, "fontweight": "bold"})
        
        plt.title(f"Meta-Learner Confusion Matrix\nMCC: {metrics['mcc']:.3f}", 
                  fontsize=20, fontweight='bold', pad=20)
        
        plt.ylabel('True Class', fontsize=20, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=20, fontweight='bold')
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=400)
        plt.close()

    """def plot_meta_importance(self, output_path):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']

        weights = self.meta_learner.coef_[0]
        expert_names = list(self.models_list.keys())
        
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        sorted_names = [expert_names[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))

        predefined_blues = ['#08519c', '#3182bd', '#6baed6', "#91c6e2"]
        
        num_experts = len(sorted_weights)
        expert_colors = [predefined_blues[i % len(predefined_blues)] for i in range(num_experts)]

        bars = plt.barh(sorted_names, sorted_weights, color=expert_colors, 
                        edgecolor='black', linewidth=0.6, alpha=1.0)
        
        plt.axvline(0, color='black', linewidth=1.2)
        
        plt.xlabel('Meta-Learner Coefficient (Importance)', fontsize=18, fontweight='bold')
        plt.title('Expert Contribution to Final Stacking Ensemble', fontsize=20, fontweight='bold', pad=15)
        
        plt.gca().invert_yaxis() 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            label_x = width + (0.01 if width >= 0 else -0.01)
            ha = 'left' if width >= 0 else 'right'
            plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', va='center', fontsize=16, fontweight='bold', ha=ha)

        plt.tight_layout()
        plt.savefig(output_path, dpi=400)
        plt.close()"""

    def run_exhaustive_upset_study(self, views_val, y_val, views_test, y_test, output_path, logger):
        logger.info("--- RUNNING UPSET STUDY (HARD VOTING BASELINE) ---")
        expert_names = list(self.models_list.keys())
        
        tune_votes, test_votes = {}, {}
        for name, model in self.models_list.items():
            tune_votes[name] = model.predict(views_val[name])[self.idx_tune]
            test_votes[name] = model.predict(views_test[name])

        results = []
        for r in range(1, len(expert_names) + 1):
            for combo in itertools.combinations(expert_names, r):
                # Calculate total votes for the subset
                subset_tune_votes = np.sum(np.column_stack([tune_votes[n] for n in combo]), axis=1)
                subset_test_votes = np.sum(np.column_stack([test_votes[n] for n in combo]), axis=1)
                
                # Re-optimize integer vote threshold for this specific subset
                best_v, best_mcc = 1, -1
                for v in range(1, len(combo) + 1):
                    mcc = matthews_corrcoef(self.y_tune, (subset_tune_votes >= v).astype(int))
                    if mcc > best_mcc: best_mcc, best_v = mcc, v
                
                preds = (subset_test_votes >= best_v).astype(int)
                results.append({'Size': r, 'Combination': combo, 'MCC': matthews_corrcoef(y_test, preds)})

        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values('MCC', ascending=True).reset_index(drop=True)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']
        
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05) 
        
        ax_bar = fig.add_subplot(gs[0])
        ax_matrix = fig.add_subplot(gs[1], sharex=ax_bar)

        cmap = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
        color_map = {1: cmap[0], 2: cmap[1], 3: cmap[2], 4: cmap[3]}
        bar_colors = [color_map[s] for s in df_res['Size']]

        bars = ax_bar.bar(df_res.index, df_res['MCC'], color=bar_colors, edgecolor='black', linewidth=0.8)
        
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', 
                        ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=90)

        best_single = df_res[df_res['Size'] == 1]['MCC'].max()
        ax_bar.axhline(best_single, color='red', linestyle='--', linewidth=1.5, label='Best Single Expert')
        
        ax_bar.set_ylabel('Test MCC', fontsize=18, fontweight='bold')
        ax_bar.set_title('Exhaustive Combinatorial Ablation: Meta-Learner Retrained per Subset', 
                         fontsize=20, fontweight='bold', pad=15)
        ax_bar.legend(loc='upper left', fontsize=16)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
        ax_bar.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_bar.set_ylim(0, max(df_res['MCC']) + 0.08)
        ax_bar.tick_params(axis='y', labelsize=16)

        ax_matrix.invert_yaxis()
        for i, row in df_res.iterrows():
            combo = row['Combination']
            y_coords = [expert_names.index(exp) for exp in combo]
            if len(y_coords) > 1:
                ax_matrix.plot([i, i], [min(y_coords), max(y_coords)], color='black', linewidth=2, zorder=1)
            ax_matrix.scatter([i]*len(y_coords), y_coords, color='black', s=130, zorder=2)
            inactive_y = [idx for idx in range(len(expert_names)) if idx not in y_coords]
            if inactive_y:
                ax_matrix.scatter([i]*len(inactive_y), inactive_y, color='lightgray', s=130, zorder=2)

        ax_matrix.set_yticks(range(len(expert_names)))
        ax_matrix.set_yticklabels(expert_names, fontsize=16, fontweight='bold')
        ax_matrix.set_xticks(range(len(df_res)))
        ax_matrix.set_xticklabels([])
        ax_matrix.tick_params(axis='both', which='both', length=0)
        ax_matrix.grid(axis='y', linestyle='--', alpha=0.3)

        handles = [mpatches.Patch(color=color_map[s], label=f'{s} Experts') for s in range(1, 5)]
        ax_matrix.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=400)
        plt.close()
        
        logger.info(f"Honest UpSet study saved to {output_path}")

    def save_models(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models_list.items():
            joblib.dump(model, os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.joblib"))
        
        # Save the integer vote threshold
        joblib.dump({'vote_threshold': self.optimized_vote_threshold}, os.path.join(output_dir, "ensemble_metadata.joblib"))

    @classmethod
    def load_models(cls, model_dir):
        instance = cls(input_dims_dict=None, config=None)
        expert_names = ["XGBoost", "RandomForest", "SVM", "KNN"]
        instance.models_list = {n: joblib.load(os.path.join(model_dir, f"{n.lower().replace(' ', '_')}.joblib")) for n in expert_names}
        metadata = joblib.load(os.path.join(model_dir, "ensemble_metadata.joblib"))
        instance.optimized_vote_threshold = metadata['vote_threshold']
        return instance