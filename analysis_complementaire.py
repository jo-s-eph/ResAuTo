#!/usr/bin/env python
"""
Analyse complémentaire des résultats d'évaluation ResAuTo.
Génère des insights supplémentaires pour le rapport et la soutenance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(csv_path):
    """Analyse complémentaire des résultats d'évaluation."""
    
    # Charger les données
    df = pd.read_csv(csv_path)
    
    print("=== ANALYSE COMPLÉMENTAIRE RESAUTO ===\n")
    
    # 1. Analyse de la variance par méthode
    print("1. STABILITÉ DES MÉTHODES (coefficient de variation)")
    print("-" * 50)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        cv_rouge1 = method_data['rouge1'].std() / method_data['rouge1'].mean()
        cv_rouge2 = method_data['rouge2'].std() / method_data['rouge2'].mean()
        print(f"{method:20} | CV ROUGE-1: {cv_rouge1:.3f} | CV ROUGE-2: {cv_rouge2:.3f}")
    
    print("\n2. PERFORMANCE RELATIVE PAR TYPE DE DOCUMENT")
    print("-" * 60)
    
    # Calculer les performances relatives
    for doc_type in df['type'].unique():
        type_data = df[df['type'] == doc_type]
        print(f"\nType: {doc_type}")
        type_grouped = type_data.groupby('method')['rouge1'].mean().sort_values(ascending=False)
        
        best_score = type_grouped.iloc[0]
        for method, score in type_grouped.items():
            relative_perf = (score / best_score) * 100
            print(f"  {method:20}: {score:.3f} ({relative_perf:.1f}% du meilleur)")
    
    # 3. Analyse des types de documents les plus difficiles
    print("\n3. DIFFICULTÉS PAR TYPE DE DOCUMENT")
    print("-" * 50)
    
    type_difficulty = df.groupby('type')['rouge1'].agg(['mean', 'std', 'min', 'max'])
    type_difficulty['difficulty_score'] = 1 - type_difficulty['mean']  # Plus le score est bas, plus c'est difficile
    type_difficulty_sorted = type_difficulty.sort_values('difficulty_score', ascending=False)
    
    print("Classement par difficulté (du plus difficile au plus facile):")
    for doc_type, row in type_difficulty_sorted.iterrows():
        print(f"{doc_type:15}: Score moyen {row['mean']:.3f} (±{row['std']:.3f})")
    
    # 4. Recommandations pour un système hybride
    print("\n4. RECOMMANDATIONS SYSTÈME HYBRIDE")
    print("-" * 50)
    
    best_method_by_type = df.loc[df.groupby('type')['rouge1'].idxmax()]
    
    print("Sélection automatique de méthode recommandée:")
    for _, row in best_method_by_type.iterrows():
        print(f"Type '{row['type']}' -> {row['method']} (ROUGE-1: {row['rouge1']:.3f})")
    
    # 5. Calcul de l'amélioration potentielle
    print("\n5. GAIN POTENTIEL D'UN SYSTÈME HYBRIDE")
    print("-" * 50)
    
    # Performance actuelle de la meilleure méthode globale (TF-IDF)
    tfidf_mean = df[df['method'] == 'TF-IDF']['rouge1'].mean()
    
    # Performance potentielle avec sélection optimale par type
    optimal_scores = []
    for doc_type in df['type'].unique():
        type_data = df[df['type'] == doc_type]
        best_score = type_data['rouge1'].max()
        optimal_scores.append(best_score)
    
    hybrid_mean = np.mean(optimal_scores)
    improvement = ((hybrid_mean - tfidf_mean) / tfidf_mean) * 100
    
    print(f"TF-IDF actuel:        {tfidf_mean:.3f}")
    print(f"Système hybride:      {hybrid_mean:.3f}")
    print(f"Amélioration:         +{improvement:.1f}%")
    
    return df

def create_summary_chart(df, output_dir):
    """Crée un graphique de synthèse pour la soutenance."""
    
    # Graphique résumé pour la soutenance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance globale
    global_perf = df.groupby('method')['rouge1'].mean().sort_values(ascending=False)
    bars1 = ax1.bar(range(len(global_perf)), global_perf.values, 
                    color=['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C'])
    ax1.set_title('Performance Globale (ROUGE-1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score ROUGE-1')
    ax1.set_xticks(range(len(global_perf)))
    ax1.set_xticklabels(global_perf.index, rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars1, global_perf.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Variabilité des méthodes
    method_stats = df.groupby('method')['rouge1'].agg(['mean', 'std'])
    cv = method_stats['std'] / method_stats['mean']
    bars2 = ax2.bar(range(len(cv)), cv.values, color='lightcoral')
    ax2.set_title('Stabilité des Méthodes', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Coefficient de Variation')
    ax2.set_xticks(range(len(cv)))
    ax2.set_xticklabels(cv.index, rotation=45, ha='right')
    
    # 3. Performance par type (heatmap simplifiée)
    pivot_simple = df.groupby(['type', 'method'])['rouge1'].mean().unstack()
    im = ax3.imshow(pivot_simple.values, cmap='YlOrRd', aspect='auto')
    ax3.set_title('Performance par Type de Document', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(pivot_simple.columns)))
    ax3.set_xticklabels(pivot_simple.columns, rotation=45, ha='right')
    ax3.set_yticks(range(len(pivot_simple.index)))
    ax3.set_yticklabels(pivot_simple.index)
    
    # Ajouter les valeurs dans la heatmap
    for i in range(len(pivot_simple.index)):
        for j in range(len(pivot_simple.columns)):
            if not pd.isna(pivot_simple.iloc[i, j]):
                ax3.text(j, i, f'{pivot_simple.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold')
    
    # 4. Comparaison ROUGE-1 vs ROUGE-2
    ax4.scatter(df['rouge1'], df['rouge2'], c=df['method'].astype('category').cat.codes, 
               cmap='viridis', alpha=0.7, s=50)
    ax4.set_xlabel('ROUGE-1')
    ax4.set_ylabel('ROUGE-2')
    ax4.set_title('Corrélation ROUGE-1 vs ROUGE-2', fontsize=14, fontweight='bold')
    
    # Ajouter une ligne de régression
    z = np.polyfit(df['rouge1'], df['rouge2'], 1)
    p = np.poly1d(z)
    ax4.plot(df['rouge1'], p(df['rouge1']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synthese_soutenance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique de synthèse sauvegardé: {output_dir / 'synthese_soutenance.png'}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analysis_complementaire.py <chemin_vers_resultats_complets.csv>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    output_dir = csv_path.parent
    
    df = analyze_results(csv_path)
    create_summary_chart(df, output_dir)
