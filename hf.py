from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from sklearn.metrics import f1_score
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import random
import os
import transformers



from final_ner_token_format_v6 import TRAIN_DATA

label2id = {"O": 0, "TECH": 1}
id2label = {0: "O", 1: "TECH"}

# HF Dataset'e dönüşüm
hf_data = []
for example in TRAIN_DATA:
    tokens = [token for token, _ in example]
    labels = [label2id[label] for _, label in example]
    hf_data.append({"tokens": tokens, "labels": labels})

random.shuffle(hf_data)
split = int(len(hf_data) * 0.8)
train_dataset = Dataset.from_list(hf_data[:split])
test_dataset = Dataset.from_list(hf_data[split:])

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

train_tokenized = train_dataset.map(tokenize_and_align)
test_tokenized = test_dataset.map(tokenize_and_align)

# Model
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=2, id2label=id2label, label2id=label2id
)

# Metriğik hesaplama
train_losses = []
val_f1_scores = []

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            train_losses.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            predictions = trainer.predict(test_tokenized)
            preds = np.argmax(predictions.predictions, axis=2)
            labels = predictions.label_ids

            true_labels, true_preds = [], []
            for pred, lab in zip(preds, labels):
                for p_, l_ in zip(pred, lab):
                    if l_ != -100:
                        true_preds.append(p_)
                        true_labels.append(l_)

            f1 = f1_score(true_labels, true_preds, average="binary")
            val_f1_scores.append(f1)

# Training args
args = TrainingArguments(
    output_dir="hf_model_output_tracked",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs_tracked",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    callbacks=[CustomCallback()]
)

trainer.train()

# Grafik çiz
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss (HF)", marker='o')
plt.plot(range(1, len(val_f1_scores)+1), val_f1_scores, label="Validation F1 (HF)", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Değer")
plt.title("Hugging Face Modeli - Epoch Bazlı Performans")
plt.legend()
plt.grid(True)
plt.savefig("hf_epoch_graph.png")
plt.show()




Eğitim;
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np
import evaluate
from sklearn.metrics import classification_report, accuracy_score
import random
from final_ner_token_format_v6 import TRAIN_DATA
import transformers
print(transformers.__version__)



label2id = {"O": 0, "TECH": 1}
id2label = {0: "O", 1: "TECH"}


hf_data = []
for example in TRAIN_DATA:
    tokens = [token for token, _ in example]
    labels = [label2id[label] for _, label in example]
    hf_data.append({"tokens": tokens, "labels": labels})

random.shuffle(hf_data)
split = int(len(hf_data) * 0.8)
train_dataset = Dataset.from_list(hf_data[:split])
test_dataset = Dataset.from_list(hf_data[split:])


tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)  # Only first token gets label
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized


train_tokenized = train_dataset.map(tokenize_and_align)
test_tokenized = test_dataset.map(tokenize_and_align)


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=2, id2label=id2label, label2id=label2id
)


metric = evaluate.load("accuracy")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []

    for pred, lab in zip(predictions, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)

    report = classification_report(true_labels, true_preds, target_names=["O", "TECH"])
    with open("hf_model_classification_report_epoch5.txt", "w") as f:
        f.write(report)
    return {
        "accuracy": accuracy_score(true_labels, true_preds)
    }



args = TrainingArguments(
    output_dir="hf_model_output",
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()
