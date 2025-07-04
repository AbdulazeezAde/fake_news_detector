import pandas as pd
import re
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import evaluate


def load_and_preprocess(true_path: Path, fake_path: Path) -> pd.DataFrame:
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    df_true['label'] = 1
    df_fake['label'] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.lower()
    df['content'] = df['content'].str.replace(r'<[^>]+>', ' ', regex=True)
    df['content'] = df['content'].str.replace(r'http\S+|www\.\S+', ' ', regex=True)
    df['content'] = df['content'].str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['content'].str.len() > 0].reset_index(drop=True)
    return df


# # Baseline with XGBoost
# def train_baseline(df: pd.DataFrame, model_path: Path):
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['content'], df['label'], test_size=0.2, random_state=42
#     )

#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8)),
#         ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
#     ])

#     params = {
#         'tfidf__max_features': [10000, 20000],
#         'xgb__n_estimators': [100, 200],
#         'xgb__max_depth': [4, 6]
#     }

#     grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='accuracy')
#     grid.fit(X_train, y_train)

#     best = grid.best_estimator_
#     preds = best.predict(X_test)

#     print("Baseline XGBoost Accuracy:", accuracy_score(y_test, preds))
#     print(classification_report(y_test, preds))

#     joblib.dump(best, model_path / 'baseline_xgb.pkl')
#     return best


# Fine-tuning with DistilBERT

from transformers import BertTokenizerFast, BertForSequenceClassification

def train_transformer(df: pd.DataFrame, model_dir: Path):
    # Prepare dataset
    ds = Dataset.from_pandas(df[['content', 'label']])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_fn(batch):
        return tokenizer(batch['content'], padding=True, truncation=True)

    ds = ds.train_test_split(test_size=0.2)
    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir / 'bert_fake_news'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        #evaluation_strategy='epoch',
        #save_strategy='epoch',
        logging_dir=str(model_dir / 'logs'),
        logging_steps=50,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save full pipeline as joblib
    model.save_pretrained(model_dir / 'bert_fake_news')
    tokenizer.save_pretrained(model_dir / 'bert_fake_news')
    joblib.dump((model, tokenizer), model_dir / 'bert_pipeline.pkl')
    print("Training complete, saving model...")

    return model


if __name__ == '__main__':
    # Paths
    DATA_DIR = Path(r'C:\Users\abdul\OneDrive\Documents\fake_news_detector')
    MODEL_DIR = Path('model')
    MODEL_DIR.mkdir(exist_ok=True)

    # Load
    df = load_and_preprocess(DATA_DIR / 'True.csv', DATA_DIR / 'Fake.csv')
    print(f"Dataset size after cleaning: {len(df)}")

    # # Train baseline
    # print("Training baseline XGBoost...")
    # train_baseline(df, MODEL_DIR)

    # Fine-tune transformer
    print("Fine-tuning DistilBERT...")
    train_transformer(df, MODEL_DIR)