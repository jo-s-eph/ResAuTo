#!/usr/bin/env python
"""Script de validation du corpus de contes.

Vérifie que tous les fichiers nécessaires sont présents et prêts pour l'évaluation.
"""

from pathlib import Path
import sys

def validate_corpus():
    """Valide la complétude du corpus de contes."""
    
    corpus_dir = Path("/Users/josephmaouche/Desktop/fac/S6/ResAuTo/assets/corpus_contes")
    
    # Liste des fichiers attendus (sans les instructions de remplissage)
    expected_files = [
        # Contes de fées classiques
        ("conte_cendrillon.txt", "conte_cendrillon_reference.txt"),
        ("conte_petitchaperon.txt", "conte_petitchaperon_reference.txt"),
        ("conte_blancheneige.txt", "conte_blancheneige_reference.txt"),
        
        # Contes populaires/folkloriques
        ("folklorique_barbe_bleue.txt", "folklorique_barbe_bleue_reference.txt"),
        ("folklorique_belle_bete.txt", "folklorique_belle_bete_reference.txt"),
        
        # Fables
        ("fable_cigale_fourmi.txt", "fable_cigale_fourmi_reference.txt"),
        ("fable_corbeau_renard.txt", "fable_corbeau_renard_reference.txt"),
        
        # Nouvelles
        ("nouvelle_maupassant.txt", "nouvelle_maupassant_reference.txt"),
        ("nouvelle_alphonse_daudet.txt", "nouvelle_alphonse_daudet_reference.txt"),
        
        # Légende
        ("legende_arthur.txt", "legende_arthur_reference.txt")
    ]
    
    print("=== VALIDATION DU CORPUS DE CONTES ===")
    print(f"Répertoire: {corpus_dir}")
    print(f"Fichiers attendus: {len(expected_files)} paires")
    
    missing_files = []
    empty_files = []
    ready_files = []
    
    for text_file, ref_file in expected_files:
        text_path = corpus_dir / text_file
        ref_path = corpus_dir / ref_file
        
        # Vérifier existence
        if not text_path.exists():
            missing_files.append(text_file)
            continue
        if not ref_path.exists():
            missing_files.append(ref_file)
            continue
        
        # Vérifier contenu (non vide et pas juste instructions)
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_content = f.read().strip()
        
        # Considérer comme vide si contient seulement des instructions
        text_is_placeholder = (
            len(text_content) < 200 or 
            "Instructions de remplissage" in text_content or
            text_content.count('#') > len(text_content.split('\n')) / 2
        )
        ref_is_placeholder = (
            len(ref_content) < 50 or
            "À compléter" in ref_content or
            ref_content.count('#') > len(ref_content.split('\n')) / 2
        )
        
        if text_is_placeholder:
            empty_files.append(f"{text_file} (placeholder)")
        elif ref_is_placeholder:
            empty_files.append(f"{ref_file} (placeholder)")
        else:
            ready_files.append((text_file, len(text_content.split()), len(ref_content.split())))
    
    # Rapport
    print(f"\n✅ FICHIERS PRÊTS: {len(ready_files)}")
    for text_file, text_words, ref_words in ready_files:
        compression = ref_words / text_words if text_words > 0 else 0
        print(f"  • {text_file.replace('.txt', '')}: {text_words} mots → {ref_words} mots ({compression:.1%})")
    
    if empty_files:
        print(f"\n⚠️  FICHIERS À REMPLIR: {len(empty_files)}")
        for file in empty_files:
            print(f"  • {file}")
    
    if missing_files:
        print(f"\n❌ FICHIERS MANQUANTS: {len(missing_files)}")
        for file in missing_files:
            print(f"  • {file}")
    
    # Status global
    total_expected = len(expected_files) * 2  # texte + référence
    total_ready = len(ready_files) * 2
    completion_rate = total_ready / total_expected
    
    print(f"\n📊 TAUX DE COMPLÉTION: {completion_rate:.1%} ({total_ready}/{total_expected} fichiers)")
    
    if completion_rate == 1.0:
        print("🎉 CORPUS COMPLET! Prêt pour l'évaluation.")
        return True
    else:
        print(f"📝 Encore {total_expected - total_ready} fichiers à compléter.")
        return False

def show_next_steps():
    """Affiche les prochaines étapes."""
    print("\n=== PROCHAINES ÉTAPES ===")
    print("1. Remplir les fichiers textes vides avec du contenu de qualité")
    print("2. Créer les résumés de référence manuels (30% du texte original)")
    print("3. Lancer l'évaluation comparative:")
    print("   python src/tale_evaluation.py")
    print("\n💡 CONSEILS POUR LES RÉFÉRENCES:")
    print("   • Préserver la structure narrative (début-milieu-fin)")
    print("   • Inclure personnages principaux et actions clés")
    print("   • Garder les formules importantes ('Il était une fois', morales)")
    print("   • Viser ~30% de la longueur originale")

if __name__ == "__main__":
    is_complete = validate_corpus()
    if not is_complete:
        show_next_steps()
