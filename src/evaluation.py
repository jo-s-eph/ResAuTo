"""
Module d'évaluation pour les algorithmes de résumé automatique.

Ce module permet d'évaluer et de comparer les différentes méthodes de
résumé automatique implémentées dans ResAuTo.
"""

import argparse
from pathlib import Path
import nltk
import numpy as np
from rouge import Rouge
import matplotlib.pyplot as plt
import summarization_algorithms as sa

# Téléchargement des ressources NLTK nécessaires
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def evaluate_summary(original_text: str, reference_summary: str, generated_summary: str):
    """Évalue un résumé généré en le comparant à un résumé de référence.
    
    Args:
        original_text: Texte original
        reference_summary: Résumé de référence (humain)
        generated_summary: Résumé généré automatiquement
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    # Initialisation de ROUGE
    rouge = Rouge()
    
    # Calcul des scores ROUGE
    try:
        scores = rouge.get_scores(generated_summary, reference_summary)[0]
    except ValueError:
        # Gérer le cas où le résumé ou la référence est vide
        scores = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }
    
    # Calculer la longueur des textes
    original_sentences = nltk.sent_tokenize(original_text)
    reference_sentences = nltk.sent_tokenize(reference_summary)
    generated_sentences = nltk.sent_tokenize(generated_summary)
    
    compression_ratio = len(generated_sentences) / len(original_sentences) if len(original_sentences) > 0 else 0
    
    # Retourner les métriques
    return {
        'rouge-1-f': scores['rouge-1']['f'],
        'rouge-2-f': scores['rouge-2']['f'],
        'rouge-l-f': scores['rouge-l']['f'],
        'compression_ratio': compression_ratio,
        'original_length': len(original_sentences),
        'reference_length': len(reference_sentences),
        'generated_length': len(generated_sentences)
    }

def compare_methods(text: str, reference_summary: str, ratio: float = 0.3):
    """Compare différentes méthodes de résumé sur un même texte.
    
    Args:
        text: Texte original à résumer
        reference_summary: Résumé de référence (humain)
        ratio: Ratio de compression
        
    Returns:
        Dictionnaire des résultats par méthode
    """
    methods = {
        'Fréquence (Basique)': sa.basic_frequency_summary,
        'TF-IDF': sa.tfidf_summary,
        'TextRank': sa.textrank_summary,
        'LSA': sa.lsa_summary
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        # Générer le résumé
        summary = method_func(text, ratio=ratio)
        
        # Évaluer le résumé
        metrics = evaluate_summary(text, reference_summary, summary)
        
        # Stocker les résultats
        results[method_name] = {
            'summary': summary,
            'metrics': metrics
        }
    
    return results

def plot_comparison(results, output_path=None):
    """Génère un graphique comparatif des méthodes de résumé.
    
    Args:
        results: Résultats de la comparaison
        output_path: Chemin pour sauvegarder le graphique
    """
    methods = list(results.keys())
    rouge1_scores = [results[method]['metrics']['rouge-1-f'] for method in methods]
    rouge2_scores = [results[method]['metrics']['rouge-2-f'] for method in methods]
    rougel_scores = [results[method]['metrics']['rouge-l-f'] for method in methods]
    
    x = np.arange(len(methods))  # Positions sur l'axe x
    width = 0.25  # Largeur des barres
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, rouge1_scores, width, label='ROUGE-1')
    rects2 = ax.bar(x, rouge2_scores, width, label='ROUGE-2')
    rects3 = ax.bar(x + width, rougel_scores, width, label='ROUGE-L')
    
    # Personnalisation du graphique
    ax.set_title('Comparaison des méthodes de résumé (Scores ROUGE)')
    ax.set_ylabel('Score F1')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points de décalage vertical
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def main():
    """Fonction principale pour l'évaluation des résumés."""
    
    parser = argparse.ArgumentParser(
        description='Évaluer et comparer les méthodes de résumé automatique',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file',
                       type=Path,
                       help='Chemin du fichier texte à résumer')
    parser.add_argument('reference_file',
                       type=Path,
                       help='Chemin du fichier contenant le résumé de référence')
    parser.add_argument('--output_dir',
                       type=Path,
                       default='results',
                       help='Dossier de sortie pour les résultats et graphiques')
    parser.add_argument('--ratio',
                       type=float,
                       default=0.3,
                       help='Ratio de compression du texte (entre 0.0 et 1.0)')
    parser.add_argument('--encoding',
                       default='utf-8',
                       help='Encodage des fichiers texte')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Lecture du fichier d'entrée
        with open(args.input_file, 'r', encoding=args.encoding) as f:
            text = f.read()
            
        # Lecture du résumé de référence
        with open(args.reference_file, 'r', encoding=args.encoding) as f:
            reference_summary = f.read()
            
    except UnicodeDecodeError:
        print(f"Erreur: Impossible de lire les fichiers avec l'encodage {args.encoding}")
        return
    
    # Comparer les méthodes
    results = compare_methods(text, reference_summary, ratio=args.ratio)
    
    # Préparation des chemins de sortie
    input_filename = Path(args.input_file).stem
    results_path = output_dir / f"{input_filename}_evaluation.txt"
    graph_path = output_dir / f"{input_filename}_evaluation_graph.png"
    
    # Écrire les résultats dans un fichier
    with open(results_path, 'w', encoding=args.encoding) as f:
        f.write(f"Évaluation des méthodes de résumé pour {args.input_file.name}\n")
        f.write(f"Ratio de compression cible: {args.ratio}\n")
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
    
    # Générer le graphique
    plot_comparison(results, output_path=graph_path)
    
    print(f"Évaluation terminée. Résultats sauvegardés dans:")
    print(f"- {results_path}")
    print(f"- {graph_path}")

if __name__ == "__main__":
    main()
