#!/usr/bin/env python
"""Outil de résumé automatique de texte

Ce script permet de résumer automatiquement un texte en utilisant différentes 
approches d'extraction (TF-IDF, TextRank, LSA) et offre la possibilité
d'intégrer les analyses émotionnelles dans le processus de résumé.
"""

import argparse
import re
import importlib.util
import sys
import os
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import summarization_algorithms as sa

# Téléchargement des ressources NLTK nécessaires
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

def get_emotion_parser_path():
    """Trouve le chemin du module d'analyse d'émotions."""
    current_dir = Path(__file__).parent
    
    # Le chemin correct est dans le dossier src/emotion_parser
    emotion_parser_path = current_dir / "emotion_parser" / "parseur.py"
    if emotion_parser_path.exists():
        return emotion_parser_path
    return None

def import_emotion_parser():
    """Importe dynamiquement le module d'analyse d'émotions."""
    emotion_parser_path = get_emotion_parser_path()
    if not emotion_parser_path:
        return None
    
    spec = importlib.util.spec_from_file_location("parseur", emotion_parser_path)
    parseur = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parseur)
    return parseur

def main():
    """Fonction principale pour traiter les arguments en ligne de commande."""
    
    # Parser les arguments
    parser = argparse.ArgumentParser(
        description='Générer un résumé automatique d\'un texte',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file',
                       type=Path,
                       help='Chemin du fichier texte à résumer')
    parser.add_argument('--output_dir',
                       type=Path,
                       default='results',
                       help='Dossier de sortie pour le texte résumé')
    parser.add_argument('--ratio',
                       type=float,
                       default=0.3,
                       help='Ratio de compression du texte (entre 0.0 et 1.0)')
    parser.add_argument('--method',
                       type=str,
                       choices=['basic', 'tfidf', 'textrank', 'lsa', 'emotion', 'tale'],
                       default='tfidf',
                       help='Méthode de résumé à utiliser')
    parser.add_argument('--emotion_method',
                       type=str,
                       choices=['basic', 'tfidf', 'textrank', 'lsa'],
                       default='tfidf',
                       help='Méthode de base à utiliser avec l\'analyse émotionnelle')
    parser.add_argument('--encoding',
                       default='utf-8',
                       help='Encodage du fichier texte')
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Afficher les messages de progression')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Lecture du fichier d'entrée
        with open(args.input_file, 'r', encoding=args.encoding) as f:
            text = f.read()
            if args.verbose:
                print(f"Lu {len(text)} caractères depuis {args.input_file}")
    except UnicodeDecodeError:
        print(f"Erreur: Impossible de lire le fichier avec l'encodage {args.encoding}")
        return
    
    # Génération du résumé avec la méthode choisie
    if args.method == 'basic':
        summary = sa.basic_frequency_summary(text, ratio=args.ratio)
    elif args.method == 'tfidf':
        summary = sa.tfidf_summary(text, ratio=args.ratio)
    elif args.method == 'textrank':
        summary = sa.textrank_summary(text, ratio=args.ratio)
    elif args.method == 'lsa':
        summary = sa.lsa_summary(text, ratio=args.ratio)
    elif args.method == 'emotion':
        # Importation et utilisation du module d'analyse d'émotions
        parseur = import_emotion_parser()
        if not parseur:
            print("Erreur: Module d'analyse d'émotions introuvable")
            return
        
        # Annoter le texte avec les émotions
        annotated_text = parseur.annotate_text(text)
        
        # Utiliser le résumé avec pondération émotionnelle
        summary = sa.get_summary_with_emotion_weighting(
            text, 
            annotated_text, 
            ratio=args.ratio, 
            method=args.emotion_method
        )
    elif args.method == 'tale':
        summary = sa.tale_summary(text, ratio=args.ratio)
    else:
        print(f"Erreur: Méthode de résumé inconnue: {args.method}")
        return
    
    # Préparation du chemin de sortie
    input_filename = Path(args.input_file).stem
    output_path = output_dir / f"{input_filename}_resume_{args.method}.txt"
    
    # Sauvegarde du résumé
    with open(output_path, 'w', encoding=args.encoding) as f:
        f.write(summary)
    
    if args.verbose:
        print(f"Traitement terminé avec succès")
    print(f"Résumé sauvegardé dans: {output_path}")
    
    # Afficher des statistiques basiques
    sentences_original = len(sent_tokenize(text))
    sentences_summary = len(sent_tokenize(summary))
    compression_ratio = sentences_summary / sentences_original if sentences_original > 0 else 0
    
    print(f"Statistiques:")
    print(f"- Phrases dans le texte original: {sentences_original}")
    print(f"- Phrases dans le résumé: {sentences_summary}")
    print(f"- Taux de compression réel: {compression_ratio:.2f} ({compression_ratio*100:.1f}%)")

if __name__ == "__main__":
    main()
