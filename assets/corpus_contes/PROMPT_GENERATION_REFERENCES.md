# Prompt pour Génération de Références de Résumé - Évaluation Méthode "Tale"

## Prompt Principal

```
Tu es un expert en résumé de textes narratifs. Ton rôle est de créer un résumé de référence de haute qualité pour évaluer une méthode de résumé automatique spécialisée dans les contes et textes narratifs.

**CONTEXTE D'ÉVALUATION :**
- Ce résumé servira de référence pour évaluer une méthode appelée "tale_summary()" qui privilégie :
  * La structure narrative (début-milieu-fin)
  * Les personnages principaux
  * Les formules traditionnelles des contes
  * La pondération des éléments d'ouverture et de clôture

**CONSIGNES STRICTES :**

1. **LONGUEUR CIBLE :** Exactement 30% de la longueur du texte original (si le texte fait 400 mots, le résumé doit faire ~120 mots)

2. **STRUCTURE NARRATIVE OBLIGATOIRE :**
   - Préserver l'ordre chronologique
   - Inclure le début (situation initiale)
   - Inclure le milieu (développement, conflit)
   - Inclure la fin (résolution, dénouement)

3. **ÉLÉMENTS À PRÉSERVER ABSOLUMENT :**
   - Tous les personnages principaux (noms et rôles)
   - Les formules d'ouverture ("Il était une fois", "Dans un royaume...")
   - Les formules de clôture ("ils vécurent heureux", "fin", morales)
   - Les actions clés qui font avancer l'intrigue
   - Les éléments magiques/merveilleux s'il y en a

4. **STYLE ET COHÉRENCE :**
   - Maintenir le style narratif du texte original
   - Préserver le ton (merveilleux, réaliste, moral selon le type)
   - Conserver les temps de narration
   - Garder la fluidité narrative

5. **CRITÈRES DE QUALITÉ :**
   - Le résumé doit être auto-suffisant (compréhensible sans le texte original)
   - Respecter la progression dramatique
   - Éviter les détails secondaires mais garder l'essence
   - Maintenir l'émotion et l'atmosphère du récit

**FORMAT DE RÉPONSE :**
Fournis uniquement le résumé de référence, sans introduction ni explication. Le résumé doit être un texte fluide, pas une liste à puces.

**TEXTE À RÉSUMER :**
[INSÉRER ICI LE TEXTE ORIGINAL]

**RÉSUMÉ DE RÉFÉRENCE (30% de la longueur originale) :**
```

## Prompt Spécialisé par Type de Texte

### Pour les Contes de Fées Classiques :
```
INSTRUCTIONS SUPPLÉMENTAIRES pour contes de fées :
- Conserver impérativement "Il était une fois" si présent
- Préserver la formule de fin heureuse
- Inclure tous les éléments magiques (transformations, objets enchantés)
- Mentionner les épreuves principales du héros/héroïne
- Garder la dimension morale implicite
```

### Pour les Fables :
```
INSTRUCTIONS SUPPLÉMENTAIRES pour fables :
- Conserver la situation initiale avec les personnages animaux
- Préserver le conflit/problème central
- Inclure obligatoirement la morale explicite (textuellement)
- Maintenir la leçon de vie
- Respecter la brièveté caractéristique du genre
```

### Pour les Nouvelles/Récits :
```
INSTRUCTIONS SUPPLÉMENTAIRES pour nouvelles :
- Préserver l'atmosphère et le cadre
- Maintenir la psychologie des personnages
- Conserver le climax/point de tension
- Inclure la chute caractéristique si présente
- Respecter le style de l'auteur (réaliste, romantique, etc.)
```

### Pour les Légendes :
```
INSTRUCTIONS SUPPLÉMENTAIRES pour légendes :
- Conserver les éléments héroïques et nobles
- Préserver les valeurs (honneur, courage, sacrifice)
- Inclure les éléments merveilleux/épiques
- Maintenir la dimension exemplaire du récit
- Respecter le registre soutenu
```

## Exemple d'Usage Pratique

**Pour automatiser le processus :**

```bash
# 1. Identifier le type de texte
TYPE="conte_classique"  # ou "fable", "nouvelle", "legende"

# 2. Calculer la longueur cible (30%)
LONGUEUR_ORIGINALE=$(wc -w < texte.txt)
LONGUEUR_CIBLE=$((LONGUEUR_ORIGINALE * 3 / 10))

# 3. Utiliser le prompt adapté avec ces informations
echo "Longueur originale: $LONGUEUR_ORIGINALE mots"
echo "Longueur cible: $LONGUEUR_CIBLE mots"
```

## Validation de la Référence Générée

**Checklist post-génération :**
- [ ] Longueur = 30% ± 5% du texte original
- [ ] Structure narrative complète (début-milieu-fin)
- [ ] Tous les personnages principaux présents
- [ ] Formules d'ouverture/clôture conservées
- [ ] Actions clés incluses
- [ ] Cohérence narrative maintenue
- [ ] Style et ton préservés

**Commande de validation automatique :**
```bash
python src/corpus_validation.py
```

---

**Note :** Ce prompt est optimisé pour générer des références de qualité qui permettront d'évaluer efficacement la méthode "tale_summary()" d'Alexandre versus les méthodes standard.
