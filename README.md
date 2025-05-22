# ResAuTo

ResAuTo est un outil de résumé automatique de textes en français, qui utilise différentes approches d'extraction pour identifier les phrases les plus importantes d'un document.

## Structure du Projet

```
ResAuTo/
├── assets/
│   └── test1.txt            # Exemple de fichier texte à résumer
│
├── results/
│   ├── test1_annotation.txt # Annotations émotionnelles
│   └── test1_resume_*.txt   # Résumés générés avec différentes méthodes
│
├── src/
│   ├── emotion_parser/
│   │   └── parseur.py       # Module d'analyse des émotions
│   │
│   ├── poc/
│   │   └── poc.py           # Preuve de concept initiale
│   │
│   ├── resauto.py           # Script principal de résumé automatique
│   ├── summarization_algorithms.py # Implémentation des algorithmes de résumé
│   └── evaluation.py        # Outils d'évaluation des résumés générés
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

### Résumé Automatique

ResAuTo propose plusieurs méthodes de résumé automatique :

#### 1. Méthode basée sur la fréquence (méthode de base)

```bash
python src/resauto.py assets/test1.txt --method basic
```

#### 2. Méthode TF-IDF

```bash
python src/resauto.py assets/test1.txt --method tfidf
```

#### 3. Algorithme TextRank

```bash
python src/resauto.py assets/test1.txt --method textrank
```

#### 4. Analyse Sémantique Latente (LSA)

```bash
python src/resauto.py assets/test1.txt --method lsa
```

#### 5. Méthode intégrant l'analyse émotionnelle

```bash
python src/resauto.py assets/test1.txt --method emotion --emotion_method tfidf
```

#### Options additionnelles

```bash
# Définir un ratio de compression différent (default: 0.3)
python src/resauto.py assets/test1.txt --method tfidf --ratio 0.5

# Spécifier un dossier de sortie personnalisé
python src/resauto.py assets/test1.txt --output_dir mon/dossier/perso

# Activer le mode verbeux pour plus d'informations pendant le traitement
python src/resauto.py assets/test1.txt --verbose
```

### Analyse d'Émotions

Le module d'analyse d'émotions annote le texte avec des émotions (Satisfaction, Dissatisfaction, Accord, Désaccord) avec un niveau d'intensité compris entre 1 et 9.

```bash
python src/emotion_parser/parseur.py assets/test1.txt
```

### Évaluation des Méthodes de Résumé

Pour comparer les différentes méthodes de résumé, un outil d'évaluation est disponible. Il nécessite un texte original et un résumé de référence (généralement créé manuellement) :

```bash
python src/evaluation.py assets/test1.txt reference_summary.txt
```

## Algorithmes Implémentés

1. **Basique (Fréquence des mots)** : Méthode simple basée sur la fréquence des mots dans le document.
2. **TF-IDF** : Utilise la métrique Term Frequency-Inverse Document Frequency pour identifier les phrases importantes.
3. **TextRank** : Algorithme inspiré de PageRank, modélise les phrases comme un graphe de similarité.
4. **LSA (Latent Semantic Analysis)** : Utilise la décomposition en valeurs singulières pour identifier les concepts sémantiques latents.
5. **Émotionnel** : Intègre l'analyse émotionnelle pour pondérer les phrases en fonction de leur charge émotionnelle.

## Format de Sortie

Les résumés générés sont sauvegardés au format texte dans le dossier `results` par défaut avec le suffixe `_resume_[method].txt` où `[method]` est la méthode utilisée.

Le parseur d'émotions génère des annotations XML avec comme attribut le type d'émotion et son intensité, par exemple :

```
<Satisfaction int=6>C'est vraiment excellent !</Satisfaction>
<Disagreement int=4>Je ne suis pas certain de cela.</Disagreement>
```

## Notes

- Les fichiers d'entrée doivent être des fichiers texte (.txt)
- Il faut spécifier l'encodage si ce n'est pas UTF-8 avec l'option `--encoding`
- Le répertoire de sortie par défaut est "results"