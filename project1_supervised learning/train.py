import pandas as pd
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Final_Dataset_Combined.csv", encoding="utf-8-sig")

# Keep only rows with valid labels
df = df.dropna(subset=["Content", "Label", "Sentiment"]).copy()
df["Content"] = df["Content"].astype(str).str.strip()

print("=" * 60)
print("Full dataset overview")
print("=" * 60)
print("Label counts:")
print(df["Label"].value_counts())
print("\nSentiment counts:")
print(df["Sentiment"].value_counts())

# -----------------------------
# 2. Chinese tokenizer
# -----------------------------
def jieba_tokenizer(text):
    return jieba.lcut(text)

# -----------------------------
# 3. Helper functions
# -----------------------------
def can_use_stratify(y_series):
    """Return True if every class has at least 2 samples."""
    return y_series.value_counts().min() >= 2

def get_cv(y_series, n_splits=5):
    """
    Use StratifiedKFold if every class has enough samples.
    Otherwise fall back to plain KFold.
    """
    min_count = y_series.value_counts().min()
    if min_count >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif min_count >= 2:
        safe_splits = min(min_count, n_splits)
        return StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 4. Build model pipelines
# -----------------------------
rf_pipeline_candidate = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=1000)),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ))
])

svm_pipeline_candidate = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=1000)),
    ("clf", LinearSVC(class_weight="balanced", random_state=42))
])

rf_pipeline_sentiment = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=1000)),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    ))
])

svm_pipeline_sentiment = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=1000)),
    ("clf", LinearSVC(class_weight="balanced", random_state=42))
])

# -----------------------------
# 5. Candidate classification
# Remove "Other" because this task is 3-class only
# -----------------------------
candidate_df = df[df["Label"] != "Other"].copy()

print("\n" + "=" * 60)
print("Candidate dataset overview (after removing 'Other')")
print("=" * 60)
print(candidate_df["Label"].value_counts())

X_candidate = candidate_df["Content"]
y_candidate = candidate_df["Label"]

candidate_cv = get_cv(y_candidate, n_splits=5)

if can_use_stratify(y_candidate):
    X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(
        X_candidate, y_candidate,
        test_size=0.2,
        random_state=42,
        stratify=y_candidate
    )
    print("\nCandidate split mode: stratified")
else:
    X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(
        X_candidate, y_candidate,
        test_size=0.2,
        random_state=42
    )
    print("\nCandidate split mode: non-stratified (insufficient class samples)")

print("\n" + "=" * 60)
print("Task A: Candidate Classification (3-Class)")
print("=" * 60)

# RF CV
rf_cv_can = cross_val_score(
    rf_pipeline_candidate, X_candidate, y_candidate,
    cv=candidate_cv, scoring="accuracy"
)
print(f"Random Forest CV Accuracy: {rf_cv_can.mean():.4f} ± {rf_cv_can.std():.4f}")

# SVM CV
svm_cv_can = cross_val_score(
    svm_pipeline_candidate, X_candidate, y_candidate,
    cv=candidate_cv, scoring="accuracy"
)
print(f"Linear SVM CV Accuracy:    {svm_cv_can.mean():.4f} ± {svm_cv_can.std():.4f}")

# RF test
rf_pipeline_candidate.fit(X_train_can, y_train_can)
y_pred_can_rf = rf_pipeline_candidate.predict(X_test_can)
rf_test_acc_can = accuracy_score(y_test_can, y_pred_can_rf)

print("\n[Random Forest Test Accuracy]")
print(f"{rf_test_acc_can:.4f}")
print("\n[Random Forest Classification Report]")
print(classification_report(y_test_can, y_pred_can_rf, digits=2))

# SVM test
svm_pipeline_candidate.fit(X_train_can, y_train_can)
y_pred_can_svm = svm_pipeline_candidate.predict(X_test_can)
svm_test_acc_can = accuracy_score(y_test_can, y_pred_can_svm)

print("\n[Linear SVM Test Accuracy]")
print(f"{svm_test_acc_can:.4f}")
print("\n[Linear SVM Classification Report]")
print(classification_report(y_test_can, y_pred_can_svm, digits=2))

# Candidate confusion matrix (RF)
candidate_labels = ["DPP_Su", "KMT_Lee", "TPP_Huang"]
cm_candidate = confusion_matrix(y_test_can, y_pred_can_rf, labels=candidate_labels)

disp_can = ConfusionMatrixDisplay(confusion_matrix=cm_candidate, display_labels=candidate_labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp_can.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Candidate Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_candidate.png", dpi=300)
plt.close()

print("Saved figure: cm_candidate.png")

# -----------------------------
# 6. Sentiment analysis
# Keep all rows with valid sentiment labels
# -----------------------------
sentiment_df = df[df["Sentiment"].isin(["Negative", "Positive"])].copy()

print("\n" + "=" * 60)
print("Sentiment dataset overview")
print("=" * 60)
print(sentiment_df["Sentiment"].value_counts())

X_sentiment = sentiment_df["Content"]
y_sentiment = sentiment_df["Sentiment"]

sentiment_cv = get_cv(y_sentiment, n_splits=5)

if can_use_stratify(y_sentiment):
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sentiment, y_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=y_sentiment
    )
    print("\nSentiment split mode: stratified")
else:
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sentiment, y_sentiment,
        test_size=0.2,
        random_state=42
    )
    print("\nSentiment split mode: non-stratified (insufficient class samples)")

print("\n" + "=" * 60)
print("Task B: Sentiment Analysis (Binary)")
print("=" * 60)

# RF CV
rf_cv_sent = cross_val_score(
    rf_pipeline_sentiment, X_sentiment, y_sentiment,
    cv=sentiment_cv, scoring="accuracy"
)
print(f"Random Forest CV Accuracy: {rf_cv_sent.mean():.4f} ± {rf_cv_sent.std():.4f}")

# SVM CV
svm_cv_sent = cross_val_score(
    svm_pipeline_sentiment, X_sentiment, y_sentiment,
    cv=sentiment_cv, scoring="accuracy"
)
print(f"Linear SVM CV Accuracy:    {svm_cv_sent.mean():.4f} ± {svm_cv_sent.std():.4f}")

# RF test
rf_pipeline_sentiment.fit(X_train_sent, y_train_sent)
y_pred_sent_rf = rf_pipeline_sentiment.predict(X_test_sent)
rf_test_acc_sent = accuracy_score(y_test_sent, y_pred_sent_rf)

print("\n[Random Forest Test Accuracy]")
print(f"{rf_test_acc_sent:.4f}")
print("\n[Random Forest Classification Report]")
print(classification_report(y_test_sent, y_pred_sent_rf, digits=2))

# SVM test
svm_pipeline_sentiment.fit(X_train_sent, y_train_sent)
y_pred_sent_svm = svm_pipeline_sentiment.predict(X_test_sent)
svm_test_acc_sent = accuracy_score(y_test_sent, y_pred_sent_svm)

print("\n[Linear SVM Test Accuracy]")
print(f"{svm_test_acc_sent:.4f}")
print("\n[Linear SVM Classification Report]")
print(classification_report(y_test_sent, y_pred_sent_svm, digits=2))

# Sentiment confusion matrix (RF)
sentiment_labels = ["Negative", "Positive"]
cm_sentiment = confusion_matrix(y_test_sent, y_pred_sent_rf, labels=sentiment_labels)

disp_sent = ConfusionMatrixDisplay(confusion_matrix=cm_sentiment, display_labels=sentiment_labels)
fig, ax = plt.subplots(figsize=(6, 5))
disp_sent.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Sentiment Analysis Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_sentiment.png", dpi=300)
plt.close()

print("Saved figure: cm_sentiment.png")

# -----------------------------
# 7. Summary for report writing
# -----------------------------
print("\n" + "=" * 60)
print("Summary for Report")
print("=" * 60)
print(f"Candidate RF CV Accuracy: {rf_cv_can.mean():.4f}")
print(f"Candidate SVM CV Accuracy: {svm_cv_can.mean():.4f}")
print(f"Candidate RF Test Accuracy: {rf_test_acc_can:.4f}")
print(f"Candidate SVM Test Accuracy: {svm_test_acc_can:.4f}")
print(f"Sentiment RF CV Accuracy: {rf_cv_sent.mean():.4f}")
print(f"Sentiment SVM CV Accuracy: {svm_cv_sent.mean():.4f}")
print(f"Sentiment RF Test Accuracy: {rf_test_acc_sent:.4f}")
print(f"Sentiment SVM Test Accuracy: {svm_test_acc_sent:.4f}")