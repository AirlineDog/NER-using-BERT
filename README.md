# NER-using-BERT

This project implements a Named Entity Recognition (NER) model using BERT combined with Part-of-Speech (POS) tag embeddings. The model is built using PyTorch and Hugging Face's Transformers library.

## Features

- Fine-tunes `bert-base-cased` model for NER tasks.
- Integrates POS tag embeddings with BERT token embeddings.
- Performs hyperparameter tuning using grid search.
- Evaluates with SeqEval metrics and generates confusion matrices.

## 🔄 Step-by-Step Workflow

The `NER-using-BERT.ipynb` notebook guides you through a complete NER pipeline using BERT and POS embeddings. Below is a breakdown of the key steps:

1. **Library Imports & Configurations**  
   Loads all necessary libraries including PyTorch, Hugging Face Transformers, Scikit-learn, SeqEval, and more. Sets up configurations like device selection (CPU/GPU).

2. **Data Preprocessing**  
   - Reads and cleans the dataset.
   - Splits the dataset into training and test sets.

3. **Tokenization**  
   - Uses `BertTokenizerFast` to tokenize text data.
   - Aligns input features such as `input_ids`, `attention_mask`, and POS tags.

4. **Model Definition**  
   - Defines a custom `BERTNERWithPOS` model.
   - Combines BERT embeddings with POS tag embeddings.
   - Outputs token-level label predictions via a linear classifier.

5. **Training**  
   - Defines loss function, optimizer (AdamW), and learning rate scheduler.
   - Trains the model across several epochs while tracking the training loss.

6. **Hyperparameter Tuning**  
   - Performs grid search across hyperparameters like learning rate, batch size, dropout rate, and POS embedding dimensions.
   - Selects the best-performing model based on F1-score.

7. **Evaluation**  
   - Predicts labels on the test set.
   - Generates detailed metrics: F1-score, Precision, Recall using `SeqEval`.
   - Creates confusion matrices with and without the 'O' label to visualize model performance.

8. **Visualization**  
   - Plots side-by-side confusion matrices using `matplotlib` for a clear comparison of entity-specific predictions.

This step-by-step process ensures a full pipeline from raw data to insightful model evaluation and visualization.

