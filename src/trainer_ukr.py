# src/trainer.py

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import model generators
from src.models import get_xgb_model, get_rf_model, get_svm_model, get_knn_model

logger = logging.getLogger("PROTAC_Pipeline.trainer")

class PROTACTrainer:
    def __init__(self, input_dims_dict=None, config=None):
        # Allow empty initialization for load_models classmethod
        if input_dims_dict is None or config is None:
            return
            
        self.config = config
        self.ablation_results = None 
        
        logger.info(f"Initializing Simplified Meta-Stacking Trainer. Dims: {input_dims_dict}")
        
        # Raw, uncalibrated experts to preserve their unique probability signatures
        self.models_list = {
            "XGBoost": get_xgb_model(config['models'].get('xgb')),
            "RandomForest": get_rf_model(config['models'].get('rf')),
            "SVM": get_svm_model(config['models'].get('svm')),
            "KNN": get_knn_model(config['models'].get('knn'))
        }
        
        # Simplified Linear Meta-Learner
        self.meta_learner = LogisticRegression(
            class_weight='balanced', 
            max_iter=2000,
            random_state=42
        )
        self.optimized_threshold = 0.5
        
        # State for strict blending splits
        self.idx_meta, self.idx_tune = None, None
        self.y_meta, self.y_tune = None, None

    def train_all(self, views_train, y_train, views_val=None, y_val=None, retrain_meta=True):
        logger.info("Stage 1: Training Heterogeneous Experts...")
        
        if retrain_meta:
            # 1. Strict Split of Validation Data to prevent threshold over-fitting
            indices = np.arange(len(y_val))
            self.idx_meta, self.idx_tune = train_test_split(
                indices, test_size=0.5, random_state=42, stratify=y_val
            )
            self.y_meta, self.y_tune = y_val[self.idx_meta], y_val[self.idx_tune]

            expert_meta_probs = []
            expert_tune_probs = []

        # 2. Expert Training & Probability Collection
        for name, model in self.models_list.items():
            logger.info(f"Training {name} Expert...")
            
            model.fit(views_train[name], y_train)
                
            if retrain_meta:
                val_probs = model.predict_proba(views_val[name])[:, 1]
                expert_meta_probs.append(val_probs[self.idx_meta])
                expert_tune_probs.append(val_probs[self.idx_tune])

        if not retrain_meta:
            logger.info("Stage 2 & 3 Skipped: Using pre-optimized Meta-Learner and Threshold.")
            return

        # Stage 2: Meta-Learner Blending
        logger.info("Stage 2: Blending Expert Probabilities...")
        
        X_meta_train = np.column_stack(expert_meta_probs)
        self.meta_learner.fit(X_meta_train, self.y_meta)
        
        # Stage 3: Threshold Optimization
        logger.info("Stage 3: Optimizing Decision Threshold...")
        
        X_meta_tune = np.column_stack(expert_tune_probs)
        tune_p = self.meta_learner.predict_proba(X_meta_tune)[:, 1]
        
        best_t, best_mcc = 0.5, -1
        for t in np.linspace(0.05, 0.95, 91):
            mcc = matthews_corrcoef(self.y_tune, (tune_p >= t).astype(int))
            if mcc > best_mcc: 
                best_mcc, best_t = mcc, t
                
        self.optimized_threshold = float(best_t)
        logger.info(f"Final Optimized Threshold: {self.optimized_threshold:.2f} (Val MCC: {best_mcc:.4f})")

    def ensemble_predict(self, views_test):
        expert_probs = []
        for name in self.models_list.keys():
            expert_probs.append(self.models_list[name].predict_proba(views_test[name])[:, 1])
        
        X_meta_test = np.column_stack(expert_probs)
        
        uncertainties = np.std(X_meta_test, axis=1)
        
        final_p = self.meta_learner.predict_proba(X_meta_test)[:, 1]
        
        return (final_p >= self.optimized_threshold).astype(int), final_p, uncertainties

    def evaluate(self, views_test, y_test):
        preds, probs, uncertainties = self.ensemble_predict(views_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        precision_pts, recall_pts, _ = precision_recall_curve(y_test, probs)
        
        conf_threshold = 0.20
        high_conf_mask = uncertainties < conf_threshold
        coverage = np.mean(high_conf_mask) * 100
        
        if len(y_test[high_conf_mask]) > 0 and len(np.unique(y_test[high_conf_mask])) > 1:
            high_conf_mcc = matthews_corrcoef(y_test[high_conf_mask], preds[high_conf_mask])
        else:
            high_conf_mcc = 0.0

        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "mcc": matthews_corrcoef(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "probabilities": probs.tolist(),
            "ground_truth": y_test.tolist(),
            "auprc": auc(recall_pts, precision_pts),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, 
            "selected_threshold": self.optimized_threshold,
            "meta_weights": self.meta_learner.coef_[0].tolist(),
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
                         label='Правильні передбачення', kde=True, stat='density', 
                         alpha=0.7, bins=20, edgecolor='black', linewidth=0.6,
                         line_kws={'linewidth': 2, 'color': 'black', 'zorder': 5})
            
        if np.any(~correct_mask):
            sns.histplot(uncertainties[~correct_mask], color=color_incorrect, 
                         label='Неправильні передбачення', kde=True, stat='density', 
                         alpha=0.7, bins=20, edgecolor='black', linewidth=0.6,
                         line_kws={'linewidth': 2, 'color': 'black', 'zorder': 5})
        
        plt.title('Епістемічна невизначеність ансамблю моделей', fontsize=20, fontweight='bold')
        plt.xlabel('Розбіжність оцінок експертів (стандартне відхилення ймовірностей)', fontsize=18)
        plt.ylabel('Щільність', fontsize=18)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 18})
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=400)
        plt.close()

    def plot_expert_correlations(self, views_test, y_test, output_path):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['axes.unicode_minus'] = False 

        preds = {}
        for name, model in self.models_list.items():
            try:
                preds[name] = model.predict_proba(views_test[name])[:, 1]
            except Exception: pass
            
        if len(preds) < 2: return
        
        df_preds = pd.DataFrame(preds)
        df_preds['True_Label'] = y_test
        
        df_actives = df_preds[df_preds['True_Label'] == 1].drop(columns=['True_Label'])
        df_inactives = df_preds[df_preds['True_Label'] == 0].drop(columns=['True_Label'])

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        heatmap_kwargs = {
            'annot': True,
            'cmap': "RdBu_r",
            'vmin': -1,
            'vmax': 1,
            'center': 0,
            'annot_kws': {"size": 16, "fontweight": "bold"}
        }

        # Побудова теплових карт
        sns.heatmap(df_inactives.corr(), ax=axes[0], **heatmap_kwargs)
        sns.heatmap(df_actives.corr(), ax=axes[1], **heatmap_kwargs)

        # Налаштування лівого графіка (Неактивні)
        cbar_0 = axes[0].collections[0].colorbar
        cbar_0.ax.tick_params(labelsize=16) 

        axes[0].set_title(f"А) Кореляція для неактивних сполук (n={len(df_inactives)})", 
                          fontsize=20, fontweight='bold', pad=15)
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        
        for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
            label.set_fontsize(14)
            label.set_fontweight('bold')

        # Налаштування правого графіка (Активні)
        cbar_1 = axes[1].collections[0].colorbar
        cbar_1.ax.tick_params(labelsize=16) 

        axes[1].set_title(f"Б) Кореляція для активних сполук (n={len(df_actives)})", 
                          fontsize=20, fontweight='bold', pad=15)
        axes[1].tick_params(axis='both', which='major', labelsize=16)
        
        for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
            label.set_fontsize(14)
            label.set_fontweight('bold')

        # Спільний заголовок
        plt.suptitle("Кореляція передбачень експертів", 
                     fontsize=24, y=0.98, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        plt.savefig(output_path, dpi=400)
        plt.close()
        
        logger.info(f"Saved expert correlation heatmaps: {output_path}")

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
        
        plt.title('Абляційний аналіз: Вплив вилучення окремих експертів', fontsize=20, fontweight='bold')
        plt.xlabel('Зниження тестового MCC (Більше значення = вища важливість експерта)', fontsize=18)
        
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
        
        plt.figure(figsize=(10, 8))
        
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Неактивні', 'Активні'], 
                    yticklabels=['Неактивні', 'Активні'],
                    annot_kws={"size": 18, "fontweight": "bold"})
        
        plt.title(f"Матриця помилок (MCC: {metrics['mcc']:.2f})", 
                  fontsize=20, fontweight='bold', pad=20)
        
        plt.ylabel('Істинний клас', fontsize=20, fontweight='bold')
        plt.xlabel('Передбачений клас', fontsize=20, fontweight='bold')
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=400)
        plt.close()

    def plot_meta_importance(self, output_path):
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
        
        plt.xlabel('Коефіцієнт мета-моделі (Важливість)', fontsize=18, fontweight='bold')
        plt.title('Внесок експертів у підсумковий ансамбль стекінгу', fontsize=20, fontweight='bold', pad=15)
        
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
        plt.close()

    def run_exhaustive_upset_study(self, views_val, y_val, views_test, y_test, output_path, logger):
        logger.info("--- RUNNING HONEST EXHAUSTIVE COMBINATORIAL (UPSET) STUDY ---")
        
        expert_names = list(self.models_list.keys())
        
        meta_probs, tune_probs, test_probs = {}, {}, {}
        for name, model in self.models_list.items():
            full_val_probs = model.predict_proba(views_val[name])[:, 1]
            meta_probs[name] = full_val_probs[self.idx_meta]
            tune_probs[name] = full_val_probs[self.idx_tune]
            test_probs[name] = model.predict_proba(views_test[name])[:, 1]

        results = []

        for r in range(1, len(expert_names) + 1):
            for combo in itertools.combinations(expert_names, r):
                logger.info(f"Evaluating Subset: {combo}")
                
                # Stack probabilities for the specific subset
                X_meta_subset = np.column_stack([meta_probs[n] for n in combo])
                X_tune_subset = np.column_stack([tune_probs[n] for n in combo])
                X_test_subset = np.column_stack([test_probs[n] for n in combo])
                
                # Retrain Meta-Learner exactly as the updated base Meta-Learner
                temp_meta = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)
                temp_meta.fit(X_meta_subset, self.y_meta)
                
                tune_p = temp_meta.predict_proba(X_tune_subset)[:, 1]
                best_t, best_mcc = 0.5, -1
                for t in np.linspace(0.05, 0.95, 91):
                    mcc = matthews_corrcoef(self.y_tune, (tune_p >= t).astype(int))
                    if mcc > best_mcc: 
                        best_mcc, best_t = mcc, t
                
                test_p = temp_meta.predict_proba(X_test_subset)[:, 1]
                preds = (test_p >= best_t).astype(int)
                final_mcc = matthews_corrcoef(y_test, preds)
                
                results.append({
                    'Size': r,
                    'Combination': combo,
                    'MCC': final_mcc
                })

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
        ax_bar.axhline(best_single, color='red', linestyle='--', linewidth=1.5, label='Найкращий окремий експерт')
        
        ax_bar.set_ylabel('MCC', fontsize=18, fontweight='bold')
        ax_bar.set_title('Комбінаторний абляційний аналіз', 
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

        handles = [mpatches.Patch(color=color_map[s], label=f'Експертів: {s}') for s in range(1, 5)]
        ax_matrix.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=400)
        plt.close()
        
        logger.info(f"Honest UpSet study saved to {output_path}")

    def save_models(self, output_dir):
        """Saves a trained ensemble and meta learner"""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models_list.items():
            path = os.path.join(output_dir, name.lower().replace(" ", "_"))
            joblib.dump(model, f"{path}.joblib")
        joblib.dump(self.meta_learner, os.path.join(output_dir, "meta_learner.joblib"))
        joblib.dump({'threshold': self.optimized_threshold}, os.path.join(output_dir, "ensemble_metadata.joblib"))

    @classmethod
    def load_models(cls, model_dir):
        """Loads a pre-trained ensemble and meta-learner directly from disk."""
        # Create an empty instance of the class
        instance = cls(input_dims_dict=None, config=None)
        
        # Reconstruct the models_list exactly how save_models saved them
        expert_names = ["XGBoost", "RandomForest", "SVM", "KNN"]
        instance.models_list = {}
        for name in expert_names:
            filename = f"{name.lower().replace(' ', '_')}.joblib"
            path = os.path.join(model_dir, filename)
            instance.models_list[name] = joblib.load(path)
            
        # Load the meta-learner
        meta_path = os.path.join(model_dir, "meta_learner.joblib")
        instance.meta_learner = joblib.load(meta_path)
        
        # Load the optimized threshold
        metadata_path = os.path.join(model_dir, "ensemble_metadata.joblib")
        metadata = joblib.load(metadata_path)
        instance.optimized_threshold = metadata['threshold']
        
        return instance