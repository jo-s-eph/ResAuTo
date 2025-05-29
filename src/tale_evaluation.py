#!/usr/bin/env python
"""Évaluation comparative spécialisée pour les contes.

Ce module évalue spécifiquement la performance de la méthode tale_summary()
d'Alexandre comparée aux méthodes générales sur le corpus de littérature
(contes et romans) et le nouveau fichier violon.txt.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from rouge import Rouge

# Ajouter le répertoire src au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import summarization_algorithms as sa

class TaleEvaluationFramework:
    """Framework d'évaluation spécialisé pour les contes."""
    
    def __init__(self, corpus_dir: str, results_dir: str):
        """Initialise le framework d'évaluation.
        
        Args:
            corpus_dir: Répertoire contenant les fichiers du corpus
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.corpus_dir = Path(corpus_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Méthodes à évaluer
        self.methods = {
            'basic': sa.basic_frequency_summary,
            'tfidf': sa.tfidf_summary,
            'textrank': sa.textrank_summary,
            'lsa': sa.lsa_summary,
            'tale': sa.tale_summary  # Méthode spécialisée d'Alexandre
        }
        
        # ROUGE scorer
        self.rouge = Rouge()
        
    def get_tale_files(self):
        """Récupère tous les fichiers de contes et textes narratifs."""
        tale_files = []
        
        # Nouveau corpus spécialisé contes (priorité)
        corpus_contes_dir = self.corpus_dir.parent / 'corpus_contes'
        if corpus_contes_dir.exists():
            conte_files = [
                # Contes de fées classiques
                'conte_cendrillon.txt',
                'conte_petitchaperon.txt', 
                'conte_blancheneige.txt',
                # Contes populaires/folkloriques
                'folklorique_barbe_bleue.txt',
                'folklorique_belle_bete.txt',
                # Fables
                'fable_cigale_fourmi.txt',
                'fable_corbeau_renard.txt',
                # Nouvelles
                'nouvelle_maupassant.txt',
                'nouvelle_alphonse_daudet.txt',
                # Légende
                'legende_arthur.txt'
            ]
            
            for filename in conte_files:
                text_file = corpus_contes_dir / filename
                ref_file = corpus_contes_dir / filename.replace('.txt', '_reference.txt')
                
                if text_file.exists() and ref_file.exists():
                    # Vérifier que les fichiers ne sont pas vides/placeholders
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                    with open(ref_file, 'r', encoding='utf-8') as f:
                        ref_content = f.read().strip()
                    
                    # Ignorer les placeholders
                    if (len(text_content) > 200 and 
                        "Instructions de remplissage" not in text_content and
                        len(ref_content) > 50 and 
                        "À compléter" not in ref_content):
                        
                        tale_files.append({
                            'name': filename.replace('.txt', ''),
                            'text_file': text_file,
                            'reference_file': ref_file,
                            'type': 'conte_specialise'
                        })
        
        # Fichiers du corpus de littérature (fallback)
        corpus_files = [
            'litterature_conte.txt',
            'litterature_roman.txt'
        ]
        
        for filename in corpus_files:
            text_file = self.corpus_dir / filename
            ref_file = self.corpus_dir / filename.replace('.txt', '_reference.txt')
            
            if text_file.exists() and ref_file.exists():
                tale_files.append({
                    'name': filename.replace('.txt', ''),
                    'text_file': text_file,
                    'reference_file': ref_file,
                    'type': 'corpus_litterature'
                })
        
        # Ajouter le fichier violon.txt avec sa référence
        violon_file = self.corpus_dir.parent / 'violon.txt'
        violon_ref = self.corpus_dir / 'violon_reference.txt'
        if violon_file.exists() and violon_ref.exists():
            tale_files.append({
                'name': 'violon',
                'text_file': violon_file,
                'reference_file': violon_ref,
                'type': 'alexandre'
            })
            
        return tale_files
    
    def evaluate_single_file(self, file_info: dict, ratio: float = 0.3):
        """Évalue un fichier avec toutes les méthodes.
        
        Args:
            file_info: Dictionnaire contenant les informations du fichier
            ratio: Ratio de résumé
            
        Returns:
            Dictionnaire des résultats d'évaluation
        """
        # Lire le texte
        with open(file_info['text_file'], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Lire la référence si disponible
        reference = None
        if file_info['reference_file']:
            with open(file_info['reference_file'], 'r', encoding='utf-8') as f:
                reference = f.read().strip()
        
        results = {
            'file': file_info['name'],
            'type': file_info['type'],
            'original_length': len(text.split()),
            'reference_length': len(reference.split()) if reference else None
        }
        
        # Générer les résumés avec chaque méthode
        for method_name, method_func in self.methods.items():
            try:
                print(f"  Évaluation {method_name}...")
                summary = method_func(text, ratio=ratio)
                summary_length = len(summary.split())
                
                results[f'{method_name}_summary'] = summary
                results[f'{method_name}_length'] = summary_length
                results[f'{method_name}_compression'] = summary_length / len(text.split())
                
                # Calcul ROUGE si référence disponible
                if reference:
                    try:
                        rouge_scores = self.rouge.get_scores(summary, reference)[0]
                        results[f'{method_name}_rouge1_f'] = rouge_scores['rouge-1']['f']
                        results[f'{method_name}_rouge2_f'] = rouge_scores['rouge-2']['f']
                        results[f'{method_name}_rougeL_f'] = rouge_scores['rouge-l']['f']
                    except Exception as e:
                        print(f"    Erreur ROUGE pour {method_name}: {e}")
                        results[f'{method_name}_rouge1_f'] = 0.0
                        results[f'{method_name}_rouge2_f'] = 0.0
                        results[f'{method_name}_rougeL_f'] = 0.0
                else:
                    # Pas de référence disponible
                    results[f'{method_name}_rouge1_f'] = None
                    results[f'{method_name}_rouge2_f'] = None
                    results[f'{method_name}_rougeL_f'] = None
                    
            except Exception as e:
                print(f"    Erreur avec {method_name}: {e}")
                results[f'{method_name}_summary'] = ""
                results[f'{method_name}_length'] = 0
                results[f'{method_name}_compression'] = 0.0
                results[f'{method_name}_rouge1_f'] = 0.0
                results[f'{method_name}_rouge2_f'] = 0.0
                results[f'{method_name}_rougeL_f'] = 0.0
        
        return results
    
    def run_tale_evaluation(self, ratio: float = 0.3):
        """Lance l'évaluation complète des contes.
        
        Args:
            ratio: Ratio de résumé à utiliser
            
        Returns:
            DataFrame avec tous les résultats
        """
        print("=== ÉVALUATION SPÉCIALISÉE DES CONTES ===")
        print(f"Ratio de résumé: {ratio}")
        
        tale_files = self.get_tale_files()
        print(f"Fichiers à évaluer: {len(tale_files)}")
        
        all_results = []
        
        for file_info in tale_files:
            print(f"\nÉvaluation de {file_info['name']}...")
            results = self.evaluate_single_file(file_info, ratio)
            all_results.append(results)
        
        # Créer DataFrame
        df = pd.DataFrame(all_results)
        
        # Sauvegarder les résultats
        results_file = self.results_dir / 'tale_evaluation_results.csv'
        df.to_csv(results_file, index=False)
        print(f"\nRésultats sauvegardés: {results_file}")
        
        return df
    
    def analyze_tale_performance(self, df: pd.DataFrame):
        """Analyse les performances spécifiques à la méthode tale.
        
        Args:
            df: DataFrame contenant les résultats d'évaluation
        """
        print("\n=== ANALYSE DE LA MÉTHODE TALE ===")
        
        # Filtrer les fichiers avec références (pour ROUGE)
        df_with_ref = df[df['reference_length'].notna()].copy()
        
        if len(df_with_ref) > 0:
            print("\nPerformances ROUGE-1 (avec référence):")
            rouge_cols = [col for col in df_with_ref.columns if col.endswith('_rouge1_f')]
            rouge_methods = [col.replace('_rouge1_f', '') for col in rouge_cols]
            
            for method in rouge_methods:
                if f'{method}_rouge1_f' in df_with_ref.columns:
                    score = df_with_ref[f'{method}_rouge1_f'].mean()
                    print(f"  {method.upper()}: {score:.3f}")
            
            # Comparaison spécifique tale vs autres
            tale_score = df_with_ref['tale_rouge1_f'].mean()
            other_scores = []
            for method in ['basic', 'tfidf', 'textrank', 'lsa']:
                if f'{method}_rouge1_f' in df_with_ref.columns:
                    other_scores.append(df_with_ref[f'{method}_rouge1_f'].mean())
            
            if other_scores:
                avg_others = np.mean(other_scores)
                improvement = ((tale_score - avg_others) / avg_others) * 100
                print(f"\nAméliorations de la méthode TALE:")
                print(f"  Score TALE: {tale_score:.3f}")
                print(f"  Score moyen autres: {avg_others:.3f}")
                print(f"  Amélioration: {improvement:+.1f}%")
        
        # Analyse des longueurs de résumé
        print(f"\nAnalyse des longueurs de résumé:")
        for method in self.methods.keys():
            if f'{method}_length' in df.columns:
                avg_length = df[f'{method}_length'].mean()
                avg_compression = df[f'{method}_compression'].mean()
                print(f"  {method.upper()}: {avg_length:.0f} mots (compression: {avg_compression:.1%})")
    
    def create_tale_visualization(self, df: pd.DataFrame):
        """Crée des visualisations spécifiques à l'évaluation des contes.
        
        Args:
            df: DataFrame contenant les résultats d'évaluation
        """
        # Configuration de style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure avec plusieurs sous-graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Évaluation Comparative: Méthode Tale vs Méthodes Standard', 
                     fontsize=16, fontweight='bold')
        
        # 1. Comparaison ROUGE-1 (si références disponibles)
        df_with_ref = df[df['reference_length'].notna()].copy()
        if len(df_with_ref) > 0:
            rouge_data = []
            for method in self.methods.keys():
                if f'{method}_rouge1_f' in df_with_ref.columns:
                    for _, row in df_with_ref.iterrows():
                        rouge_data.append({
                            'Méthode': method.upper(),
                            'ROUGE-1': row[f'{method}_rouge1_f'],
                            'Fichier': row['file']
                        })
            
            rouge_df = pd.DataFrame(rouge_data)
            sns.boxplot(data=rouge_df, x='Méthode', y='ROUGE-1', ax=axes[0,0])
            axes[0,0].set_title('Distribution des Scores ROUGE-1')
            axes[0,0].tick_params(axis='x', rotation=45)
        else:
            axes[0,0].text(0.5, 0.5, 'Pas de références\ndisponibles', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Scores ROUGE-1 (non disponibles)')
        
        # 2. Longueurs de résumé
        length_data = []
        for method in self.methods.keys():
            if f'{method}_length' in df.columns:
                for _, row in df.iterrows():
                    length_data.append({
                        'Méthode': method.upper(),
                        'Longueur': row[f'{method}_length'],
                        'Fichier': row['file']
                    })
        
        length_df = pd.DataFrame(length_data)
        sns.barplot(data=length_df, x='Méthode', y='Longueur', ax=axes[0,1])
        axes[0,1].set_title('Longueur Moyenne des Résumés')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Taux de compression
        compression_data = []
        for method in self.methods.keys():
            if f'{method}_compression' in df.columns:
                for _, row in df.iterrows():
                    compression_data.append({
                        'Méthode': method.upper(),
                        'Compression': row[f'{method}_compression'] * 100,
                        'Fichier': row['file']
                    })
        
        compression_df = pd.DataFrame(compression_data)
        sns.violinplot(data=compression_df, x='Méthode', y='Compression', ax=axes[1,0])
        axes[1,0].set_title('Taux de Compression (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap des performances par fichier (ROUGE-1)
        if len(df_with_ref) > 0:
            heatmap_data = []
            for method in self.methods.keys():
                row_data = []
                for _, file_row in df_with_ref.iterrows():
                    if f'{method}_rouge1_f' in file_row:
                        row_data.append(file_row[f'{method}_rouge1_f'])
                    else:
                        row_data.append(0.0)
                heatmap_data.append(row_data)
            
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=[m.upper() for m in self.methods.keys()],
                columns=df_with_ref['file'].tolist()
            )
            
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,1])
            axes[1,1].set_title('ROUGE-1 par Fichier et Méthode')
        else:
            axes[1,1].text(0.5, 0.5, 'Pas de données ROUGE\ndisponibles', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Heatmap ROUGE-1 (non disponible)')
        
        plt.tight_layout()
        
        # Sauvegarder
        viz_file = self.results_dir / 'tale_evaluation_visualization.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {viz_file}")
        plt.show()
    
    def generate_tale_report(self, df: pd.DataFrame):
        """Génère un rapport détaillé de l'évaluation des contes.
        
        Args:
            df: DataFrame contenant les résultats d'évaluation
        """
        report_file = self.results_dir / 'tale_evaluation_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT D'ÉVALUATION: MÉTHODE TALE vs MÉTHODES STANDARD\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CONTEXTE:\n")
            f.write("Ce rapport compare la méthode tale_summary() développée par Alexandre\n")
            f.write("avec les méthodes standard sur un corpus de textes narratifs.\n\n")
            
            f.write("MÉTHODE TALE - CARACTÉRISTIQUES:\n")
            f.write("• Pondération narrative (1.5x début/fin)\n")
            f.write("• Détection de personnages principaux\n")
            f.write("• Reconnaissance de formules de contes\n")
            f.write("• Base TF-IDF avec bonus spécialisés\n\n")
            
            f.write("CORPUS ÉVALUÉ:\n")
            for _, row in df.iterrows():
                ref_info = f" (réf: {row['reference_length']} mots)" if row['reference_length'] else " (sans réf)"
                f.write(f"• {row['file']}: {row['original_length']} mots{ref_info}\n")
            f.write("\n")
            
            # Performances ROUGE
            df_with_ref = df[df['reference_length'].notna()].copy()
            if len(df_with_ref) > 0:
                f.write("PERFORMANCES ROUGE-1 (fichiers avec référence):\n")
                for method in self.methods.keys():
                    if f'{method}_rouge1_f' in df_with_ref.columns:
                        score = df_with_ref[f'{method}_rouge1_f'].mean()
                        f.write(f"• {method.upper()}: {score:.3f}\n")
                
                tale_score = df_with_ref['tale_rouge1_f'].mean()
                other_scores = []
                for method in ['basic', 'tfidf', 'textrank', 'lsa']:
                    if f'{method}_rouge1_f' in df_with_ref.columns:
                        other_scores.append(df_with_ref[f'{method}_rouge1_f'].mean())
                
                if other_scores:
                    avg_others = np.mean(other_scores)
                    improvement = ((tale_score - avg_others) / avg_others) * 100
                    f.write(f"\nAMÉLIORAION DE LA MÉTHODE TALE:\n")
                    f.write(f"• Score TALE: {tale_score:.3f}\n")
                    f.write(f"• Score moyen autres: {avg_others:.3f}\n")
                    f.write(f"• Amélioration: {improvement:+.1f}%\n\n")
            
            # Analyse des longueurs
            f.write("LONGUEURS DE RÉSUMÉ (moyenne):\n")
            for method in self.methods.keys():
                if f'{method}_length' in df.columns:
                    avg_length = df[f'{method}_length'].mean()
                    avg_compression = df[f'{method}_compression'].mean()
                    f.write(f"• {method.upper()}: {avg_length:.0f} mots ({avg_compression:.1%} du texte original)\n")
            
            f.write("\nCONCLUSIONS:\n")
            f.write("1. La méthode tale_summary() est spécifiquement optimisée pour les textes narratifs\n")
            f.write("2. Elle intègre des éléments structurels et stylistiques des contes\n")
            f.write("3. Comparaison avec les méthodes générales nécessaire pour validation\n")
            f.write("4. Potentiel d'extension à d'autres genres littéraires\n")
        
        print(f"Rapport détaillé sauvegardé: {report_file}")


def main():
    """Fonction principale pour lancer l'évaluation des contes."""
    # Chemins
    corpus_dir = "/Users/josephmaouche/Desktop/fac/S6/ResAuTo/assets/corpus_evaluation"
    results_dir = "/Users/josephmaouche/Desktop/fac/S6/ResAuTo/results/tale_evaluation"
    
    # Créer le framework d'évaluation
    evaluator = TaleEvaluationFramework(corpus_dir, results_dir)
    
    # Lancer l'évaluation
    df = evaluator.run_tale_evaluation(ratio=0.3)
    
    # Analyser les performances
    evaluator.analyze_tale_performance(df)
    
    # Créer les visualisations
    evaluator.create_tale_visualization(df)
    
    # Générer le rapport
    evaluator.generate_tale_report(df)
    
    print("\n=== ÉVALUATION TERMINÉE ===")
    print(f"Résultats disponibles dans: {results_dir}")


if __name__ == "__main__":
    main()
