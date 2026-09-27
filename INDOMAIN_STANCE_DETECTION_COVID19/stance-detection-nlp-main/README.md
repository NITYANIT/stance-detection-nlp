# COVID-19 Stance Detection with BERTweet

Fine-tuned BERTweet model for stance detection on COVID-19 tweets.
Classifies text as FAVOR, AGAINST, or NONE toward a given topic.

## Results (COVID-19 dataset)
| Class    | F1    |
|----------|-------|
| AGAINST  | 0.685 |
| FAVOR    | 0.714 |
| NEUTRAL  | 0.718 |
| **Overall F1-macro** | **0.706** |

## Setup
```bash
pip install transformers torch tweet-preprocessor wordninja pandas
```

## Training
```bash
python train_model_v2.py \
    -c config/config-roberta_large.txt \
    -s 42 -d 0.1 \
    -train data/ -dev data/ -test data/ \
    -dataset covid19 -leave_one_out 0 \
    -lr1 1e-5 -lr2 1e-4
```

## Inference
```bash
python inference.py \
    --model_path RoBERTa_seed42.pt \
    --topic "face masks" \
    --text "Everyone should wear masks to protect others."
```

## Model Architecture
- Base: BERTweet (vinai/bertweet-base)
- Classifier: Separate mean pooling of text and topic tokens → Linear(1536→768) → Linear(768→3)
- Dataset: COVID-19 Stance Dataset (4533 train / 800 val / 800 test)

## Credits
Based on ZeroStance (Chen et al.) — https://github.com/chenyez/ZeroStance
