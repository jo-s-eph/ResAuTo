#!/usr/bin/env python
"""
Évaluation automatisée sur un corpus de textes.

Ce script permet d'évaluer les performances des différentes méthodes
de résumé sur un corpus de textes variés et de générer des rapports
d'évaluation complets.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import json

# Importer les modules existants
from evaluation import compare_methods, evaluate_summary
import summarization_algorithms as sa

def process_corpus(corpus_dir, output_dir, ratio=0.3, encoding='utf-8'):
    """
    Traite un corpus entier de textes et résumés de référence.
    
    Structure attendue du corpus:
    corpus_dir/
        text1.txt
        text1_reference.txt
        text2.txt
        text2_reference.txt
        ...
    
    Args:
        corpus_dir: Répertoire contenant les textes et résumés de référence
        output_dir: Répertoire de sortie pour les résultats
        ratio: Ratio de compression à utiliser
        encoding: Encodage des fichiers
    
    Returns:
        DataFrame contenant tous les résultats
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Trouver tous les textes (fichiers qui n'ont pas "_reference" dans leur nom)
    text_files = [f for f in corpus_dir.glob("*.txt") if "_reference" not in f.name]
    
    all_results = []
    
    for text_file in text_files:
        # Déterminer le chemin du résumé de référence
        reference_file = corpus_dir / f"{text_file.stem}_reference.txt"
        
        if not reference_file.exists():
            print(f"Avertissement: Pas de résumé de référence pour {text_file.name}, ignoré.")
            continue
        
        print(f"Traitement de {text_file.name}...")
        
        # Lire le texte et le résumé de référence
        try:
            with open(text_file, 'r', encoding=encoding) as f:
                text = f.read()
            
            with open(reference_file, 'r', encoding=encoding) as f:
                reference_summary = f.read()
        except Exception as e:
            print(f"Erreur lors de la lecture des fichiers: {e}")
            continue
        
        # Collecter des informations sur le type de document
        doc_type = "non_spécifié"
        if "_" in text_file.stem:
            doc_type = text_file.stem.split("_")[0]
        
        # Comparer les différentes méthodes
        results = compare_methods(text, reference_summary, ratio=ratio)
        
        # Sauvegarder les résultats pour ce document
        results_path = output_dir / f"{text_file.stem}_evaluation.txt"
        with open(results_path, 'w', encoding=encoding) as f:
            f.write(f"Évaluation des méthodes de résumé pour {text_file.name}\n")
            f.write(f"Type de document: {doc_type}\n")
            f.write(f"Ratio de compression cible: {ratio}\n")
            f.write("="*50 + "\n\n")
            
            for method, result in results.items():
                metrics = result['metrics']
                summary = result['summary']
                
                f.write(f"Méthode: {method}\n")
                f.write("-"*30 + "\n")
                f.write(f"ROUGE-1 (F1): {metrics['rouge-1-f']:.4f}\n")
                f.write(f"ROUGE-2 (F1): {metrics['rouge-2-f']:.4f}\n")
                f.write(f"ROUGE-L (F1): {metrics['rouge-l-f']:.4f}\n")
                f.write(f"Taux de compression: {metrics['compression_ratio']:.2f}\n")
                f.write(f"Phrases (original/résumé): {metrics['original_length']}/{metrics['generated_length']}\n\n")
                
                f.write("Résumé généré:\n")
                f.write(summary + "\n\n")
                f.write("="*50 + "\n\n")
        
        # Générer un graphique pour ce document
        graph_path = output_dir / f"{text_file.stem}_evaluation_graph.png"
        try:
            from evaluation import plot_comparison
            plot_comparison(results, output_path=graph_path)
            plt.close()  # Fermer la figure pour éviter les conflits
        except Exception as e:
            print(f"Erreur lors de la génération du graphique: {e}")
        
        # Collecter les résultats pour l'analyse globale
        for method, result in results.items():
            metrics = result['metrics']
            all_results.append({
                'document': text_file.stem,
                'type': doc_type,
                'method': method,
                'rouge1': metrics['rouge-1-f'],
                'rouge2': metrics['rouge-2-f'],
                'rougeL': metrics['rouge-l-f'],
                'compression': metrics['compression_ratio'],
                'original_length': metrics['original_length'],
                'summary_length': metrics['generated_length']
            })
    
    # Convertir en DataFrame pour l'analyse
    if all_results:
        return pd.DataFrame(all_results)
    else:
        print("Aucun résultat n'a pu être collecté.")
        return None

def generate_corpus_report(results_df, output_dir):
    """
    Génère un rapport global d'évaluation pour tout le corpus.
    
    Args:
        results_df: DataFrame contenant tous les résultats
        output_dir: Répertoire de sortie pour les rapports
    """
    if results_df is None or len(results_df) == 0:
        print("Pas de données disponibles pour le rapport.")
        return
    
    output_dir = Path(output_dir)
    
    # Créer un timestamp pour le rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"rapport_corpus_{timestamp}"
    report_dir.mkdir(exist_ok=True)
    
    # Sauvegarder les données brutes
    results_df.to_csv(report_dir / "resultats_complets.csv", index=False)
    
    # 1. Performance globale par méthode
    plt.figure(figsize=(12, 8))
    method_performance = results_df.groupby('method')[['rouge1', 'rouge2', 'rougeL']].mean()
    method_performance.plot(kind='bar', ylim=(0, 1))
    plt.title("Performance moyenne des méthodes sur l'ensemble du corpus")
    plt.ylabel("Score ROUGE (F1)")
    plt.tight_layout()
    plt.savefig(report_dir / "performance_globale_methodes.png")
    plt.close()
    
    # 2. Performance par type de document
    plt.figure(figsize=(15, 10))
    type_method_perf = results_df.pivot_table(
        index='type', 
        columns='method', 
        values='rouge1', 
        aggfunc='mean'
    )
    type_method_perf.plot(kind='bar', ylim=(0, 1))
    plt.title("Performance des méthodes par type de document (ROUGE-1)")
    plt.ylabel("Score ROUGE-1 (F1)")
    plt.tight_layout()
    plt.savefig(report_dir / "performance_par_type.png")
    plt.close()
    
    # 3. Heatmap des performances
    plt.figure(figsize=(12, 10))
    pivot_data = results_df.pivot_table(
        index='type', 
        columns='method', 
        values='rouge1', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.3f', vmin=0, vmax=1)
    plt.title("Heatmap des performances par type de document et méthode (ROUGE-1)")
    plt.tight_layout()
    plt.savefig(report_dir / "heatmap_performance.png")
    plt.close()
    
    # 4. Rapport textuel des performances
    with open(report_dir / "rapport_synthese.txt", 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE SYNTHÈSE - ÉVALUATION DES MÉTHODES DE RÉSUMÉ\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. PERFORMANCE GLOBALE\n")
        f.write("-"*70 + "\n")
        global_stats = results_df.groupby('method')[['rouge1', 'rouge2', 'rougeL']].mean()
        f.write(f"{global_stats.to_string()}\n\n")
        
        f.write("2. MEILLEURE MÉTHODE PAR TYPE DE DOCUMENT\n")
        f.write("-"*70 + "\n")
        for doc_type in results_df['type'].unique():
            type_data = results_df[results_df['type'] == doc_type]
            best_method = type_data.loc[type_data['rouge1'].idxmax()]
            f.write(f"Type: {doc_type}\n")
            f.write(f"Meilleure méthode: {best_method['method']}\n")
            f.write(f"Score ROUGE-1: {best_method['rouge1']:.4f}\n")
            f.write(f"Score ROUGE-2: {best_method['rouge2']:.4f}\n")
            f.write(f"Score ROUGE-L: {best_method['rougeL']:.4f}\n\n")
        
        f.write("3. STATISTIQUES PAR MÉTHODE\n")
        f.write("-"*70 + "\n")
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            f.write(f"Méthode: {method}\n")
            f.write(f"Score ROUGE-1 moyen: {method_data['rouge1'].mean():.4f} (±{method_data['rouge1'].std():.4f})\n")
            f.write(f"Score ROUGE-2 moyen: {method_data['rouge2'].mean():.4f} (±{method_data['rouge2'].std():.4f})\n")
            f.write(f"Score ROUGE-L moyen: {method_data['rougeL'].mean():.4f} (±{method_data['rougeL'].std():.4f})\n")
            f.write(f"Taux de compression moyen: {method_data['compression'].mean():.4f}\n\n")
    
    print(f"Rapport complet généré dans: {report_dir}")
    return report_dir

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Évaluer les méthodes de résumé sur un corpus de textes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('corpus_dir',
                       type=Path,
                       help='Répertoire contenant les textes et résumés de référence')
    parser.add_argument('--output_dir',
                       type=Path,
                       default='results',
                       help='Répertoire de sortie pour les résultats')
    parser.add_argument('--ratio',
                       type=float,
                       default=0.3,
                       help='Ratio de compression du texte (entre 0.0 et 1.0)')
    parser.add_argument('--encoding',
                       default='utf-8',
                       help='Encodage des fichiers texte')
    
    args = parser.parse_args()
    
    # Vérifier que le répertoire du corpus existe
    if not args.corpus_dir.exists() or not args.corpus_dir.is_dir():
        print(f"Erreur: Le répertoire '{args.corpus_dir}' n'existe pas ou n'est pas un dossier.")
        return 1
    
    # Traiter le corpus
    results_df = process_corpus(
        args.corpus_dir, 
        args.output_dir, 
        ratio=args.ratio, 
        encoding=args.encoding
    )
    
    if results_df is not None:
        # Générer le rapport global
        report_dir = generate_corpus_report(results_df, args.output_dir)
        print(f"Évaluation du corpus terminée. Consultez le rapport dans: {report_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
