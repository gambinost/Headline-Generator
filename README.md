# Headline Generation

Short: A simple sequence-to-sequence LSTM model and transformer that generates short summaries from article text.

## Quick project flow
1. **Preprocess**: lowercase, basic cleanup; add `startseq` / `endseq` to each summary.  
2. **Tokenize & limit vocab**: fit tokenizers on training data, cap vocab sizes (articles, summaries).  
3. **Convert & pad**: texts → integer sequences → pad/truncate to fixed lengths.  
4. **Prepare decoder data**: `decoder_input` = summary without last token; `decoder_target` = summary without first token.  
5. **Train**: encoder (Embedding → LSTM) → decoder (Embedding → LSTM → Dense softmax). Use `sparse_categorical_crossentropy` and Adam.  
6. **Evaluate**: decode test set (greedy) and compute BLEU scores.

## Files
- `notebook.ipynb` — data prep, model build, training, evaluation.  
- `best_seq2seq.keras` — saved model weights (trained).  
- (optional) `tokenizers.pkl` — saved `article_tok` and `summary_tok`.

## How to run (high level)
 
# open the notebook and run cells: preprocessing → training → evaluation

