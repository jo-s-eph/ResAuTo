import argparse
import spacy
import nltk
from pathlib import Path

nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load('fr_core_news_sm') # Charger le modèle français
import fr_core_news_sm

# Augmenter la limite de caractères pour les textes longs
nlp.max_length = 3000000 

def detect_emotion(text: str) -> tuple:
    """Detecter l'émotion et l'intensité d'un texte. 
    (une phrase dans notre cas)
    """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    print(text + " | " + str(scores))
    
    # Déterminer l'émotion et l'intensité en fonction du score de compound
    if scores['compound'] >= 0.05:
        if scores['pos'] > 0.5:
            emotion = 'Satisfaction'
        else:
            emotion = 'Agreement'
        intensity = min(9, int(scores['compound'] * 9))
    elif scores['compound'] <= -0.05:
        if scores['neg'] > 0.5:
            emotion = 'Dissatisfaction'
        else:
            emotion = 'Disagreement'
        intensity = min(9, int(abs(scores['compound'] * 9)))
    else:
        return None, None
    return emotion, intensity

def annotate_text(text: str) -> str:
    """Annoter le texte avec les émotions et l'intensité détectées
    dan le format de balises XML.
    """
    doc = nlp(text)
    annotated_text = text
    
   # Pour chaque phrase,
    for sent in doc.sents:
        emotion, intensity = detect_emotion(sent.text)
        if emotion and intensity:
            # Création de l'annotation
            annotated_version = f"<{emotion} int={intensity}>{sent.text.strip()}</{emotion}>"
            annotated_text = annotated_text.replace(sent.text.strip(), annotated_version)
    
    return annotated_text

def main():
    # Parser les args
    parser = argparse.ArgumentParser(
        description='Annoter les émotions et l\'intensité dans un texte',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', 
                       help='Chemin du fichier texte à annoter')
    parser.add_argument('--output_dir', 
                       default='results', 
                       help='Dossier de sortie pour le texte annoté')
    parser.add_argument('--max_length', 
                       type=int, 
                       default=3000000,
                       help='Longueur maximale du texte (Spacy)')
    parser.add_argument('--encoding', 
                       default='utf-8',
                       help='Encodage du fichier texte')
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Afficher les messages de progression')
    
    args = parser.parse_args()
    
    # Fixer la taille maximale du texte pour Spacy
    nlp.max_length = args.max_length
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(args.input_file, 'r', encoding=args.encoding) as f:
            text = f.read()
            if args.verbose:
                print(f"Read {len(text)} characters from {args.input_file}")
    except UnicodeDecodeError:
        print(f"Error: Could not read file with {args.encoding} encoding")
        return
    
    # Annoter le texte
    annotated_text = annotate_text(text)
    
    input_filename = Path(args.input_file).stem
    output_path = output_dir / f"{input_filename}_annotation.txt"
    
    # Sauvegarder le texte annoté
    with open(output_path, 'w', encoding=args.encoding) as f:
        f.write(annotated_text)
    
    if args.verbose:
        print(f"Processing completed successfully")
    print(f"Annotated text saved to: {output_path}")

if __name__ == "__main__":
    main()