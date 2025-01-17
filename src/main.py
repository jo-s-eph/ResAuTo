import argparse
from pathlib import Path
from emotion_parser import EmotionParser

def main():
    parser = argparse.ArgumentParser(
        description='Annoter les émotions et l\'intensité dans un texte',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ...existing argument parser code...
    
    args = parser.parse_args()
    
    emotion_parser = EmotionParser(max_length=args.max_length)
    
    # ...existing file processing code...

if __name__ == "__main__":
    main()
