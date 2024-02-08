# import trained model
# here you can do translation tasks

from model import Translator
import torch
from language_processing import text_transform, vocab_transform

m = torch.load("trained_model/model.pth")

en_to_fr = Translator(m, text_transform, vocab_transform)

# SRC, GT pairs from the validation set.
infer_sentences = [
    ["Take a seat.", "Prends place !"],
    ["I'm not scared to die", "Je ne crains pas de mourir."],
    ["You'd better make sure that it is true.", "Tu ferais bien de t'assurer que c'est vrai."],
    ["The clock has stopped.", "L'horloge s'est arrêtée."],
    ["Take any two cards you like.", "Prends deux cartes de ton choix."]
]
for sentence in infer_sentences:
    print(f"SRC: {sentence[0]}")
    print(f"GT: {sentence[1]}")
    print(f"PRED: {en_to_fr.translate(sentence[0])}\n")