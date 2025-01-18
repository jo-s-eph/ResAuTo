"""Proof of concept : résumé automatique de texte utilisant l'analyse de fréquence"""

from typing import Dict, List, Set
from pathlib import Path
import argparse
from string import punctuation
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les caractères spéciaux.
    
    Args:
        text: Texte d'entrée à nettoyer
        
    Returns:
        Texte nettoyé
    """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_word_frequencies(
    text: str, 
    stop_words: Set[str]
) -> Dict[str, float]:
    """Calcule la fréquence normalisée des mots dans le texte.
    
    Args:
        text: Texte d'entrée
        stop_words: Ensemble des mots vides à exclure
        
    Returns:
        Dictionnaire des fréquences des mots
    """
    words = word_tokenize(text.lower())
    freq_table: Dict[str, float] = {}
    
    for word in words:
        if word not in stop_words and word not in punctuation and word.isalnum():
            freq_table[word] = freq_table.get(word, 0) + 1
            
    max_freq = max(freq_table.values()) if freq_table else 1
    for word in freq_table:
        freq_table[word] = freq_table[word]/max_freq
            
    return freq_table

def score_sentences(
    sentences: List[str], 
    freq_table: Dict[str, float]
) -> Dict[str, float]:
    """Calcule le score de chaque phrase selon les fréquences des mots.
    
    Args:
        sentences: Liste des phrases
        freq_table: Dictionnaire des fréquences des mots
        
    Returns:
        Dictionnaire des scores des phrases
    """
    sentence_scores: Dict[str, float] = {}
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        word_count = len([word for word in words if word.isalnum()])
        
        if word_count > 0:
            for word in words:
                if word in freq_table:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq_table[word]
                    else:
                        sentence_scores[sentence] += freq_table[word]
            
            sentence_scores[sentence] = sentence_scores[sentence] / word_count
            
    return sentence_scores

def generate_summary(text: str, ratio: float = 0.3) -> str:
    """Génère un résumé du texte par extraction de phrases.
    
    Args:
        text: Texte d'entrée à résumer
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        
    Returns:
        Texte résumé
    """
    cleaned_text = clean_text(text)
    stop_words = set(stopwords.words('french'))
    sentences = sent_tokenize(cleaned_text)
    
    if len(sentences) <= 2:
        return cleaned_text
    
    freq_table = get_word_frequencies(cleaned_text, stop_words)
    sentence_scores = score_sentences(sentences, freq_table)
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def main():
    """Fonction principale pour traiter les arguments en ligne de commande."""
    # Parser les args
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

    # Génération du résumé
    summary = generate_summary(text, ratio=args.ratio)

    # Préparation du chemin de sortie
    input_filename = Path(args.input_file).stem
    output_path = output_dir / f"{input_filename}_resumee.txt"

    # Sauvegarde du résumé
    with open(output_path, 'w', encoding=args.encoding) as f:
        f.write(summary)

    if args.verbose:
        print("Traitement terminé avec succès")
    print(f"Résumé sauvegardé dans: {output_path}")

if __name__ == "__main__":
    main()