# Twitter Sentiment Analysis — Naive Bayes

A multiclass sentiment classifier that labels tweets as **Positive**, **Negative**, or **Neutral** using TF-IDF vectorization and a Multinomial Naive Bayes model.

---

## Project Scope

This project takes the raw text of a tweet and predicts the sentiment expressed in it. The pipeline covers everything end to end:

- Loading and cleaning a real-world tweet dataset
- Text preprocessing tailored for Twitter language (URLs, mentions, hashtags, HTML noise)
- Feature extraction using TF-IDF with unigrams and bigrams
- Training a probabilistic classifier (Multinomial Naive Bayes)
- Evaluating performance with both train/test split and a held-out validation set

The goal is to build a reusable model that can be loaded later and used to score new tweets one at a time.

---

## Use Cases

A tweet sentiment classifier has practical applications in several places:

- **Brand monitoring** — automatically scan mentions of a product or company to flag negative sentiment early
- **Customer feedback analysis** — classify support tweets to prioritize responses to unhappy customers
- **Market research** — measure public reaction to a launch, campaign, or news event in real time
- **Content moderation** — surface negative or hostile content for human review
- **Trend analysis** — track sentiment movement over time toward a topic, brand, or entity

---

## Approach

### 1. Data preparation
The raw dataset is the *Twitter Entity Sentiment Analysis* corpus from Kaggle (~74K tweets across four classes: Positive, Negative, Neutral, Irrelevant).

- The `Irrelevant` class is dropped — it doesn't represent a sentiment, so it would only confuse the model
- The `tweet_id` and `entity` columns are dropped — entity tells us *what* the tweet is about, not *how* the writer feels, so including it would let the model learn topic-based shortcuts that don't generalize
- Class balancing is applied via stratified sampling so the model isn't biased toward whichever class has more rows
- Sentiment labels are encoded as `Negative=0, Neutral=1, Positive=2`

### 2. Text preprocessing
A single `text_preprocessing()` function applies these steps in order:

- Lowercasing
- Removing URLs (`http://...`, `www....`, `xyz.com/...`)
- Removing `@mentions`
- Removing HTML-style tags
- Stripping the `#` symbol (keeping the hashtag word itself)
- Removing all special characters except `!` and `?`
- Tokenization
- Stopword removal — but **keeping negation words** like `not`, `didn't`, `wasn't` because they flip sentiment meaning
- Porter stemming to reduce words to their root form

### 3. Feature extraction — TF-IDF
- `TfidfVectorizer(max_features=20000, ngram_range=(1, 2))`
- Bigrams capture short phrases like `not good` or `didn't enjoy` as single features, which preserves negation context that single words would miss
- TF-IDF weights words by importance — common words across all tweets get low weight, distinctive words get high weight

### 4. Model — Multinomial Naive Bayes
The project initially used **GaussianNB**, but two issues surfaced:

- GaussianNB assumes features follow a bell curve, which is wrong for sparse TF-IDF vectors
- It required a dense matrix, which caused memory crashes (61K × 20K dense array ≈ several GB)

The switch to **MultinomialNB** solved both: it's designed specifically for word-frequency-style features, accepts sparse matrices directly, and removes the need for a `MinMaxScaler` step entirely.

### 5. Evaluation
- 80/20 train/test split for development metrics
- A separate `twitter_validation.csv` file is used as a true held-out validation set
- The validation notebook loads the saved model and runs prediction without any retraining — this is the honest measure of how the model would behave on truly unseen data
- Metrics reported: accuracy, classification report (precision/recall/F1 per class), confusion matrix

---

## Folder Structure

```
twitter-sentiment-analysis-naivebayes/
│
├── data/
│   ├── twitter_training.csv              # Training dataset
│   └── twitter_validation.csv            # Held-out validation set
│
├── Model/
│   └── Twitter_Sentiment_Analysis_TfIdf.pkl   # Saved model + vectorizer
│
├── src/
│   ├── twitter_senti_analysis.ipynb           # Training notebook
│   └── twitter_sentment_validation.ipynb      # Validation notebook
│
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.10 or above. Install dependencies with:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
wordcloud
mlxtend
joblib
jupyter
```

After installing, download the NLTK stopwords corpus once:
```python
import nltk
nltk.download("stopwords")
```

---

## How to Run

### Training
1. Place `twitter_training.csv` inside the `data/` folder
2. Open `src/twitter_senti_analysis.ipynb` in Jupyter
3. Run all cells in order
4. The trained model gets saved to `Model/Twitter_Sentiment_Analysis_TfIdf.pkl`

### Validation
1. Make sure `twitter_validation.csv` is in the `data/` folder
2. Open `src/twitter_sentment_validation.ipynb`
3. Run all cells — it will load the saved model and report validation accuracy, classification report, and confusion matrix

### Single-tweet prediction
Inside the notebook, the `real_time_prediction()` function takes a raw tweet string and returns the predicted sentiment label:

```python
real_time_prediction("This is the best update they've shipped in years!")
# -> "Positive"
```

---

## Key Decisions and Trade-offs

| Decision | Why |
|---|---|
| Drop `Irrelevant` class | It's not a sentiment — keeping it would confuse the model |
| Drop `entity` column | Entity is metadata about the topic, not the sentiment. Including it would create topic-based shortcuts that don't generalize |
| Keep negation stopwords | Removing `not`, `didn't` would flip sentiment meaning (`didn't like` → `like`) |
| TF-IDF with bigrams | Captures short phrases like `not good` as a single feature |
| MultinomialNB over GaussianNB | Designed for word-frequency features, handles sparse matrices, no scaler needed |
| Skip MinMaxScaler | TF-IDF is already non-negative; scaling forced dense conversion which crashed memory |
| Validation in a separate notebook | Prevents accidentally tuning on validation data, which would defeat its purpose |

---

## Future Improvements

- Try **Logistic Regression** or **Linear SVM** as alternative classifiers — they often outperform Naive Bayes on text
- Experiment with **stratified train_test_split** to keep class proportions consistent
- Add **GridSearchCV** to tune `max_features`, `min_df`, `max_df`, and `alpha`
- Build a **Streamlit interface** to make the model accessible to non-technical users
- Use **pre-trained embeddings** (Word2Vec, GloVe, or transformer models like BERT) for richer representations

---

## Dataset Source

Twitter Entity Sentiment Analysis — [Kaggle link](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
