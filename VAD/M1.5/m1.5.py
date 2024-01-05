# %%
import sys
import os
sys.path.append(os.path.abspath("."))

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from time import localtime, strftime
from VAD.D2.data_selector import DataSelector

SEED = 42

# %%
# tmp_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
# fake_batch = {
#     "input_ids": torch.ones(32, 512, dtype=torch.long).to("cuda"), 
#     "attention_mask": torch.ones(32, 512, dtype=torch.long).to("cuda")
# }
# fake_labels = torch.zeros(32, 3).to("cuda")
# tmp_model.to("cuda")
# logits = tmp_model(**fake_batch).logits
# loss = ((logits - fake_labels) ** 2).mean()
# loss.backward()

# %%
ds = DataSelector("VAD/D2/data.csv")
data = ds.select_data(30000, length=5, scale_sd=30, seed=SEED)
data['conversation'] = data['conversation'].astype(str)

# %%
data.hist(bins=20)

# %%
raw_dataset = Dataset.from_pandas(data)
train_testval = raw_dataset.train_test_split(test_size=0.2, seed=SEED)
test_val = train_testval['test'].train_test_split(test_size=0.5, seed=SEED)

dataset = DatasetDict({
    'train': train_testval['train'],
    'test': test_val['test'],
    'val': test_val['train']}
)

# %%
dataset['train'][0]

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def mask_data(mask_ratio):
    def mask_function(examples):
        mask_idx = np.random.choice(len(examples["text"]), int(mask_ratio * len(examples["text"])), replace=False)
        convs = examples["text"].copy()
        for i in mask_idx:
            convs[i][-1] = tokenizer.mask_token
        
        return {"text": convs}

    return mask_function

def preprocess_data(examples):
    role_names = ("speaker", "respondent")

    convs = [
        tokenizer.apply_chat_template(
            [{"role": role_names[i % 2], "content": x} for i, x in enumerate(m)], 
            tokenize=False)
        for m in examples["text"]
    ]

    encoding = tokenizer(convs, max_length=512, truncation=True)
    encoding["labels"] = list(zip(examples["V"], examples["A"], examples["D"]))

    return encoding


dataset["train"] = dataset["train"].map(mask_data(1), batched=True, batch_size=100000)
dataset["masked_val"] = dataset["val"].map(mask_data(1), batched=True, batch_size=100000)
dataset["masked_test"] = dataset["test"].map(mask_data(1), batched=True, batch_size=100000)
tokenized_dataset = dataset.map(preprocess_data, batched=True, batch_size=100000, 
                                remove_columns=['text', 'conversation', 'source', 'V', 'A', 'D', '__index_level_0__'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
def get_model():
    return AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = 4/(1+np.exp(-logits)) + 1

    metrics = {
        "mae": mean_absolute_error(labels, logits, multioutput="raw_values").tolist(),
        "mse": mean_squared_error(labels, logits, multioutput="raw_values").tolist(),
        "pearsonr": [pearsonr(logits[:, i], labels[:, i])[0] for i in range(len(logits[0]))],
        "r_squared": r2_score(labels, logits, multioutput="raw_values").tolist(),
    }

    return {f"{m}_{s}": metrics[m][i] for i, s in enumerate("VAD") for m in metrics}

# %%
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs['logits']) * 4 + 1
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


# %%
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class MaskEvalCallback(TrainerCallback):
    def __init__(self):
        training_args = TrainingArguments(
            output_dir="/tmp/tmp_trainer",
            per_device_eval_batch_size=32,
            report_to="none",
        )

        self.trainer = RegressionTrainer(
            model=get_model(),
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )


    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self.trainer.model = model
        print(self.trainer.evaluate(tokenized_dataset["masked_val"]))


# %%
dir_name = strftime("%b-%d-%y-%H:%M:%S", localtime())
# dir_name = "Jan-01-24-14:55:56"
dir_name = "Jan-02-24-08:40:31"
# dir_name = "test"

training_args = TrainingArguments(
    output_dir=f"VAD/M1.5/results/{dir_name}",
    logging_dir=f"VAD/M1.5/results/{dir_name}/runs",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-6,
    max_grad_norm=1.0,
    num_train_epochs=20,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    logging_strategy="epoch",
    report_to="none",
    save_strategy="epoch",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    disable_tqdm=False
)

trainer = RegressionTrainer(
    model_init=get_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['masked_val'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # callbacks=[MaskEvalCallback()]
)

trainer.train(resume_from_checkpoint=True)