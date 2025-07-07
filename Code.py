import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load CSV file
df = pd.read_csv("reviews.csv")

# Define prompt template
def build_prompt(review):
    return f"Extract product features from this review: \"{review}\""

# Prepare pipeline
feature_extractor = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Extract features
def extract_features(review):
    prompt = build_prompt(review)
    result = feature_extractor(prompt, max_length=64, do_sample=False)
    return result[0]['generated_text']

df["features"] = df["review"].apply(extract_features)

# Save the results
df.to_csv("extracted_features.csv", index=False)
print("Features saved to extracted_features.csv")

from collections import Counter

# Tokenize features
all_features = []
for feats in df["features"]:
    all_features.extend([f.strip().lower() for f in feats.split(",") if f.strip()])

# Count frequencies
feature_counts = Counter(all_features)
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter

# Load the model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load dataset
def load_reviews(filepath):
    df = pd.read_csv(filepath)
    if 'review' not in df.columns:
        raise ValueError("CSV must contain a 'review' column.")
    return df

# Prompt engineering function
def build_prompt(review):
    return (
        f"Extract product features like 'battery life' or 'camera quality' from this review:\n"
        f"{review}\n"
        f"List the features (comma-separated):"
    )

# Feature extraction using FLAN-T5
def extract_features_from_review(review):
    prompt = build_prompt(review)
    response = pipe(prompt, max_length=64, do_sample=False)
    output = response[0]['generated_text'].strip()

    # To avoid full review being repeated
    if review.lower().strip() in output.lower().strip():
        return "N/A"  
    return output


# Apply extraction to all reviews
def process_reviews(df):
    df['features'] = df['review'].apply(extract_features_from_review)
    return df

# Summarize most/least mentioned features
def summarize_features(df):
    all_features = []
    for line in df['features']:
        features = [f.strip().lower() for f in line.split(',') if f.strip()]
        all_features.extend(features)
    counter = Counter(all_features)
    return counter.most_common(10)

# Save the result to CSV
def save_output(df, out_path="extracted_features.csv"):
    df.to_csv(out_path, index=False)
    print(f"Output saved to {out_path}")

# Main

if __name__ == "__main__":
    input_path = "reviews.csv"  
    df = load_reviews(input_path)
    df = process_reviews(df)
    save_output(df)

    # Summary
    print("\nTop 10 Extracted Features:")
    all_features = []
    for feats in df["features"]:
        all_features.extend([f.strip().lower() for f in feats.split(",") if f.strip()])

    from collections import Counter
    feature_counts = Counter(all_features)

    for feat, count in feature_counts.most_common(10):
        print(f"{feat} - {count} times")

    # Top 5 appreciated features
    print("\nMost Mentioned Features:")
    for feat, count in feature_counts.most_common(5):
        print(f"{feat} - {count} times")
