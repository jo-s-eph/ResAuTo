"""Module d'algorithmes de résumé automatique de texte.

Ce module fournit différentes implémentations d'algorithmes pour le résumé automatique
de textes basés sur des approches d'extraction.
"""
from typing import Dict, List, Set, Tuple
import re
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

def get_sentences(text: str) -> List[str]:
    """Segmente le texte en phrases.
    
    Args:
        text: Texte d'entrée
        
    Returns:
        Liste des phrases du texte
    """
    cleaned_text = clean_text(text)
    return sent_tokenize(cleaned_text)

def basic_frequency_summary(text: str, ratio: float = 0.3) -> str:
    """Génère un résumé en utilisant une méthode basique basée sur la fréquence des mots.
    
    Args:
        text: Texte d'entrée à résumer
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        
    Returns:
        Texte résumé
    """
    sentences = get_sentences(text)
    
    if len(sentences) <= 2:
        return text
    
    # Préparation des stopwords
    stop_words = set(stopwords.words('french'))
    
    # Calcul des fréquences
    words = word_tokenize(text.lower())
    freq_table = {}
    
    for word in words:
        if word not in stop_words and word not in punctuation and word.isalnum():
            freq_table[word] = freq_table.get(word, 0) + 1
            
    max_freq = max(freq_table.values()) if freq_table else 1
    for word in freq_table:
        freq_table[word] = freq_table[word]/max_freq
    
    # Calcul des scores de phrases
    sentence_scores = {}
    
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
    
    # Sélection des phrases pour le résumé
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def tfidf_summary(text: str, ratio: float = 0.3) -> str:
    """Génère un résumé en utilisant la méthode TF-IDF.
    
    Args:
        text: Texte d'entrée à résumer
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        
    Returns:
        Texte résumé
    """
    sentences = get_sentences(text)
    
    if len(sentences) <= 2:
        return text
    
    # Paramétrage du vectoriseur TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('french'),
        min_df=1,
        max_features=5000
    )
    
    # Création de la matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calcul des scores de phrases
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        # Le score est la somme des scores TF-IDF des termes dans la phrase
        sentence_scores[sentence] = np.sum(tfidf_matrix[i].toarray())
    
    # Sélection des phrases pour le résumé
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def textrank_summary(text: str, ratio: float = 0.3) -> str:
    """Génère un résumé en utilisant l'algorithme TextRank.
    
    Args:
        text: Texte d'entrée à résumer
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        
    Returns:
        Texte résumé
    """
    sentences = get_sentences(text)
    
    if len(sentences) <= 2:
        return text
    
    # Création de la matrice de similarité
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Normalisation de la matrice
    np.fill_diagonal(similarity_matrix, 0)  # Ignorer la similarité avec soi-même
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    if row_sums.all():  # Éviter la division par zéro
        norm_similarity_matrix = similarity_matrix / row_sums
    else:
        norm_similarity_matrix = similarity_matrix
        
    # Initialisation des scores
    scores = np.ones(len(sentences)) / len(sentences)
    
    # Itération pour la convergence (principe de TextRank)
    damping = 0.85  # Facteur d'amortissement
    iterations = 30
    
    for _ in range(iterations):
        scores = (1 - damping) + damping * (norm_similarity_matrix.T @ scores)
    
    # Association des scores aux phrases
    sentence_scores = {sentence: score for sentence, score in zip(sentences, scores)}
    
    # Sélection des phrases pour le résumé
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def lsa_summary(text: str, ratio: float = 0.3, n_components: int = 3) -> str:
    """Génère un résumé en utilisant l'analyse sémantique latente (LSA).
    
    Args:
        text: Texte d'entrée à résumer
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        n_components: Nombre de dimensions sémantiques à conserver
        
    Returns:
        Texte résumé
    """
    sentences = get_sentences(text)
    
    if len(sentences) <= 2:
        return text
    
    # Ajustement du nombre de composantes pour les petits textes
    n_components = min(n_components, len(sentences)-1) if len(sentences) > 1 else 1
    
    # Création de la matrice TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Application de la décomposition SVD (base de LSA)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_matrix)
    
    # Transformation des phrases dans l'espace LSA
    transformed_sentences = svd.transform(tfidf_matrix)
    
    # Calcul des scores: distance entre chaque phrase et le premier vecteur singulier
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = abs(transformed_sentences[i, 0])
    
    # Sélection des phrases pour le résumé
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def get_summary_with_emotion_weighting(text: str, annotated_text: str, ratio: float = 0.3, method: str = 'tfidf') -> str:
    """Génère un résumé en tenant compte des émotions annotées dans le texte.
    
    Args:
        text: Texte d'entrée original à résumer
        annotated_text: Texte annoté avec les émotions (format XML)
        ratio: Proportion du texte original à conserver (entre 0.0 et 1.0)
        method: Méthode de résumé à utiliser ('basic', 'tfidf', 'textrank', 'lsa')
        
    Returns:
        Texte résumé
    """
    import re
    
    # Extraction des phrases avec émotions et leurs intensités
    emotion_pattern = r'<(Satisfaction|Dissatisfaction|Agreement|Disagreement) int=(\d)>(.*?)</\1>'
    emotion_sentences = re.findall(emotion_pattern, annotated_text)
    
    # Création d'un dictionnaire des phrases avec leurs poids émotionnels
    emotion_weights = {}
    for emotion, intensity, sentence in emotion_sentences:
        # Normalisation de l'intensité entre 0 et 1
        normalized_intensity = float(intensity) / 9.0
        emotion_weights[sentence.strip()] = normalized_intensity
    
    # Génération du résumé avec la méthode spécifiée
    sentences = get_sentences(text)
    
    if len(sentences) <= 2:
        return text
    
    # Obtention des scores de base selon la méthode choisie
    if method == 'basic':
        base_summary = basic_frequency_summary(text, ratio=1.0)  # On récupère tous les scores
        base_sentences = get_sentences(base_summary)
        base_scores = {sent: 1.0 - i/len(base_sentences) for i, sent in enumerate(base_sentences)}
    elif method == 'tfidf':
        # Utilisation de TF-IDF pour les scores de base
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        base_scores = {sentence: np.sum(tfidf_matrix[i].toarray()) for i, sentence in enumerate(sentences)}
    elif method == 'textrank':
        # Utilisation de TextRank pour les scores de base
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        if row_sums.all():
            norm_similarity_matrix = similarity_matrix / row_sums
        else:
            norm_similarity_matrix = similarity_matrix
        scores = np.ones(len(sentences)) / len(sentences)
        damping = 0.85
        iterations = 30
        for _ in range(iterations):
            scores = (1 - damping) + damping * (norm_similarity_matrix.T @ scores)
        base_scores = {sentence: score for sentence, score in zip(sentences, scores)}
    elif method == 'lsa':
        # Utilisation de LSA pour les scores de base
        n_components = min(3, len(sentences)-1) if len(sentences) > 1 else 1
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(tfidf_matrix)
        transformed_sentences = svd.transform(tfidf_matrix)
        base_scores = {sentence: abs(transformed_sentences[i, 0]) for i, sentence in enumerate(sentences)}
    else:
        raise ValueError(f"Méthode de résumé inconnue: {method}")
    
    # Normalisation des scores de base
    max_base_score = max(base_scores.values()) if base_scores else 1.0
    for sentence in base_scores:
        base_scores[sentence] /= max_base_score
    
    # Combinaison des scores de base avec les poids émotionnels
    final_scores = {}
    emotion_weight_factor = 0.5  # Poids de l'émotion dans le score final
    
    for sentence in sentences:
        sentence_clean = sentence.strip()
        base_score = base_scores.get(sentence, 0.0)
        emotion_score = emotion_weights.get(sentence_clean, 0.0)
        
        # Score final: combinaison du score de base et du poids émotionnel
        final_scores[sentence] = (1 - emotion_weight_factor) * base_score + emotion_weight_factor * emotion_score
    
    # Sélection des phrases pour le résumé
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:select_length]])
    
    return summary

def tale_summary(text: str, ratio: float = 0.3) -> str:
    """Résumé spécialisé pour les contes, avec pondération narrative, détection de personnages principaux,
    et valorisation des formules traditionnelles du conte.

    Args:
        text: Texte du conte à résumer
        ratio: Proportion du texte original à conserver

    Returns:
        Résumé du conte
    """
    sentences = get_sentences(text)
    if len(sentences) <= 2:
        return text

    # Étape 1 : détection des personnages (entités nommées par majuscule répétée)
    words = word_tokenize(text)
    capitalized_words = [w for w in words if w.istitle() and w.lower() not in stopwords.words('french')]

    name_freq = {}
    for word in capitalized_words:
        if word not in punctuation and len(word) > 1:
            name_freq[word] = name_freq.get(word, 0) + 1

    main_characters = {name for name, freq in name_freq.items() if freq >= 2}

    # Étape 2 : TF-IDF classique
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('french'),
        min_df=1,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    base_scores = {sentence: np.sum(tfidf_matrix[i].toarray()) for i, sentence in enumerate(sentences)}

    # Liste des formules classiques de contes
    tale_formulas = [
        "il était une fois", "dans un royaume lointain", "au temps jadis",
        "dans un pays lointain", "un jour", "il y a fort longtemps",
        "chaque jour", "tout à coup", "un beau matin",
        "ils vécurent heureux", "jusqu'à la fin des temps", "fin", "la morale de cette histoire",
        "et c’est ainsi que", "et depuis ce jour"
    ]

    def contains_formula(sentence: str) -> bool:
        sentence_lower = sentence.lower()
        return any(formula in sentence_lower for formula in tale_formulas)

    # Étape 3 : pondération narrative + bonus
    def narrative_weight(index: int, total: int) -> float:
        position = index / total
        return (
            1.5 if position < 0.15 else
            1.2 if position > 0.85 else
            1.0
        )

    total_sentences = len(sentences)
    weighted_scores = {}

    for i, sent in enumerate(sentences):
        base_score = base_scores.get(sent, 0.0)
        weight = narrative_weight(i, total_sentences)

        char_bonus = 1.0
        for name in main_characters:
            if name in sent:
                char_bonus += 0.2

        formule_bonus = 1.3 if contains_formula(sent) else 1.0

        final_score = base_score * weight * char_bonus * formule_bonus
        weighted_scores[sent] = final_score

    # Étape 4 : sélection des meilleures phrases
    select_length = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    selected_sentences = [s[0] for s in sorted_sentences[:select_length]]

    # Rétablir l'ordre narratif original
    sentence_order = {sent: i for i, sent in enumerate(sentences)}
    selected_sentences.sort(key=lambda s: sentence_order.get(s, 0))
    summary = " ".join(selected_sentences)

    return summary
