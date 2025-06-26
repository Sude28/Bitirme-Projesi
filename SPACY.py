import spacy
from spacy.training.example import Example
from final_ner_token_format_v6 import TRAIN_DATA  # Eğitim verini bu dosyada tuttuğunu varsayıyoruz

# Boş İngilizce model başlat
nlp = spacy.blank("en")

# NER pipeline'ı ekle
ner = nlp.add_pipe("ner")

# Etiketleri tanımla
for tokens in TRAIN_DATA:
    for _, label in tokens:
        if label != "O":
            ner.add_label(label)

# Eğitim döngüsü
optimizer = nlp.begin_training()

for i in range(60):
    losses = {}
    for tokens in TRAIN_DATA:
        text = " ".join([t for t, _ in tokens])
        annotations = {"entities": []}
        start = 0
        for token, label in tokens:
            end = start + len(token)
            if label != "O":
                annotations["entities"].append((start, end, label))
            start = end + 1  # boşluk dahil
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.3, losses=losses)
    print(f"Iteration {i+1}, Losses: {losses}")

# Kaydet
nlp.to_disk("tech_ner_model_v12")
print("✅ Model başarıyla 'tech_ner_model_v12' klasörüne kaydedildi.")
