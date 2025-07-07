# 🧠 Product Feature Extraction using FLAN-T5

This project uses Google's **FLAN-T5** model to automatically extract product features (e.g., *camera quality*, *battery life*) from customer reviews using prompt-based generation. It helps businesses identify what aspects of their products users care about the most.

---

## 🚀 Project Overview

📌 **Objective**:  
To extract and summarize product features mentioned in textual reviews using a pre-trained generative AI model (FLAN-T5).

📌 **Key Features**:
- Prompt-based feature extraction from free-text reviews
- Uses Hugging Face Transformers and FLAN-T5
- Outputs a structured CSV of reviews and extracted features
- Summarizes top-mentioned product aspects

---

## 🧰 Tech Stack

| Component         | Details                                  |
|------------------|-------------------------------------------|
| Language          | Python                                    |
| Model             | `google/flan-t5-base` (via Hugging Face) |
| Libraries         | Transformers, Pandas                     |
| Platform Tested   | VS Code + Local Machine (CPU)            |
| Dataset           | Custom `reviews.csv` with product reviews |

---

