# Corpus d'Évaluation Spécialisé : Méthode Tale vs Méthodes Standard

## Objectif
Évaluer la performance de la méthode `tale_summary()` comparée aux méthodes standard (TF-IDF, TextRank, LSA, Fréquence) sur un corpus de textes narratifs variés.

## Hypothèse à tester
**Hypothèse A** : "Tale" est généralement meilleur sur les contes
→ Focus sur métriques ROUGE + corpus varié

## Structure du corpus (10 textes + références)

### Contes de fées classiques (3 textes)
- `conte_cendrillon.txt` + `conte_cendrillon_reference.txt`
- `conte_petitchaperon.txt` + `conte_petitchaperon_reference.txt`  
- `conte_blancheneige.txt` + `conte_blancheneige_reference.txt`

### Contes populaires/folkloriques (2 textes)
- `folklorique_barbe_bleue.txt` + `folklorique_barbe_bleue_reference.txt`
- `folklorique_belle_bete.txt` + `folklorique_belle_bete_reference.txt`

### Fables (2 textes)
- `fable_cigale_fourmi.txt` + `fable_cigale_fourmi_reference.txt`
- `fable_corbeau_renard.txt` + `fable_corbeau_renard_reference.txt`

### Nouvelles courtes (2 textes)
- `nouvelle_maupassant.txt` + `nouvelle_maupassant_reference.txt`
- `nouvelle_alphonse_daudet.txt` + `nouvelle_alphonse_daudet_reference.txt`

### Légende (1 texte)
- `legende_arthur.txt` + `legende_arthur_reference.txt`

## Critères de sélection des textes
- **Longueur** : Entre 200-1000 mots (optimal pour résumé 30%)
- **Structure narrative claire** : début, milieu, fin
- **Présence de personnages identifiables**
- **Formules traditionnelles** (pour contes/fables)
- **Domaine public** (droits libres)

## Critères pour les références manuelles
1. **Préserver la structure narrative** (début-milieu-fin)
2. **Inclure les personnages principaux**
3. **Garder les formules importantes** ("Il était une fois", morale, etc.)
4. **Conserver les actions clés**
5. **Ratio cible : ~30%** du texte original

## Métriques d'évaluation
- **ROUGE-1, ROUGE-2, ROUGE-L** (comparaison avec références)
- **Taux de compression** (longueur résumé / longueur original)
- **Analyse statistique** (moyennes, écarts-types, significativité)

## Usage
```bash
python src/tale_evaluation.py
```

---
**Note** : Tous les fichiers textes seront remplis manuellement avec du contenu de qualité et leurs références correspondantes.
