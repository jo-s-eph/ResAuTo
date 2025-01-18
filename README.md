# ResAuTo

## Structure du Projet

```
ResAuTo/
├── assets/
│
├── results/
│
├── src/
│   ├── emotion_parser/
│   │   └── parseur.py
│   │
│   └── poc/
│       └── pco.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Créer l'environnement

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Pour Mac/Linux :
source venv/bin/activate
# Pour Windows :
.\venv\Scripts\activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt

# Télécharger le modèle de langue en français pour SpaCy
python -m spacy download fr_core_news_sm
```

## Utilisation

### Exécuter le Parseur

Le parseur annote le texte avec des émotions (Satisfaction, Insatisfaction, Accord, Désaccord) avec un niveaux d'intensité comprit entre 1 et 9.

Cas simple :
```bash
python src/emotion_parser/parseur.py assets/test1.txt
```

Avec un répertoire de sortie personnalisé :
```bash
python src/emotion_parser/parseur.py input.txt --output_dir mon/dossier/perso
```

### Format de Sortie

Le parseur génère des annotations XML avec comme attribut le type d'émotion et son intensité. Par exemple :

```
<Satisfaction int=6>C'est vraiment excellent !</Satisfaction>
<Disagreement int=4>Je ne suis pas certain de cela.</Disagreement>
```

### Notes

- Le fichier d'entrée doit être un fichier texte (.txt), il faut spécifer l'encodage si ce n'est pas UTF-8
- Le répertoire de sortie par défaut est "results"
- Les fichiers annotés sont sauvegardés avec le suffixe "_annotation"

### Exécuter la proof of concept

La PoC résume automatiquement un texte court en utilisant l'analyse de fréquence.

Cas simple :
```bash
python src/poc/poc.py assets/test1.txt
```

Avec un répertoire de sortie personnalisé :
```bash
python src/poc/poc.py assets/test1.txt --output_dir mon/dossier/perso
```

### Format de Sortie

Le parseur génère un fichier texte contenant le résumé generée. Par exemple :

```
Il a montré une satisfaction immense lorsqu'il a pu transporter des charges plus légères. 
De plus, le concept d'instinct est souvent mal compris. 
Les Ânes sont souvent perçus comme des symboles de bêtise.
```

### Notes

- Le fichier d'entrée doit être un fichier texte (.txt), il faut spécifer l'encodage si ce n'est pas UTF-8
- Le répertoire de sortie par défaut est "results"
- Les fichiers annotés sont sauvegardés avec le suffixe "_resumee"