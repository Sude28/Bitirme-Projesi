import spacy
from spacy.training.example import Example
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import random
import numpy as np

# Veri seti
from final_ner_token_format_v6 import TRAIN_DATA

# %80 eğitim - %20 doğrulama ayır
random.shuffle(TRAIN_DATA)
split = int(len(TRAIN_DATA) * 0.8)
train_data = TRAIN_DATA[:split]
val_data = TRAIN_DATA[split:]

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Etiketleri tanımla
for tokens in TRAIN_DATA:
    for _, label in tokens:
        if label != "O":
            ner.add_label(label)

optimizer = nlp.begin_training()

train_losses = []
val_f1_scores = []

for epoch in range(60):
    losses = {}
    random.shuffle(train_data)
    for tokens in train_data:
        text = " ".join([t for t, _ in tokens])
        annotations = {"entities": []}
        start = 0
        for token, label in tokens:
            end = start + len(token)
            if label != "O":
                annotations["entities"].append((start, end, label))
            start = end + 1
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.3, losses=losses)

    train_losses.append(losses["ner"])

    # Validation verisinde f1-score hesapla
    true_labels = []
    pred_labels = []

    for tokens in val_data:
        text = " ".join([t for t, _ in tokens])
        expected_tech_count = sum(1 for _, label in tokens if label == "TECH")

        doc = nlp(text)
        predicted_tech_count = sum(1 for ent in doc.ents if ent.label_ == "TECH")

        # Her cümle için min eşleşmeye göre ikisini dengele
        count = min(expected_tech_count, predicted_tech_count)
        true_labels.extend([1] * count)
        pred_labels.extend([1] * count)

        # Ekstra farklar varsa sıfırla tamamla
        if expected_tech_count > count:
            true_labels.extend([1] * (expected_tech_count - count))
            pred_labels.extend([0] * (expected_tech_count - count))
        elif predicted_tech_count > count:
            pred_labels.extend([1] * (predicted_tech_count - count))
            true_labels.extend([0] * (predicted_tech_count - count))

    f1 = f1_score(true_labels, pred_labels, average='binary') if pred_labels else 0
    val_f1_scores.append(f1)

    print(f"Epoch {epoch+1} - Loss: {losses['ner']:.4f} - Val F1: {f1:.4f}")

# Grafik çizimi
fig, ax1 = plt.subplots(figsize=(10, 6))  # <--- Bu satır eksikse NameError oluşur

ax1.plot(range(1, 61), train_losses, label="Train Loss", color="red")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss", color="red")
ax1.tick_params(axis='y', labelcolor='red')

# Validation F1 - ikinci eksen
ax2 = ax1.twinx()
ax2.plot(range(1, 61), val_f1_scores, label="Validation F1", color="blue")
ax2.set_ylabel("Validation F1", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("spaCy Modeli - Epoch Bazlı Performans")
fig.tight_layout()
plt.savefig("spacy_train_val_graph_dual_axis.png")
plt.show()
# Modeli kaydet
nlp.to_disk("tech_ner_model_tracked")
