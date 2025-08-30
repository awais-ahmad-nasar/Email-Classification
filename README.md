# Email-Classification

A sophisticated Python-based email classification system designed to categorize emails into eight distinct categories: Updates, Personal, Promotions, Forums, Purchases, Travel, Spam, and Social. This project combines state-of-the-art pretrained transformer models (DistilBERT, BERT, RoBERTa) with an advanced rule-based classification system, optimized for both CPU and GPU environments. It employs comprehensive keyword pattern matching, regex-based structural analysis, and contextual indicators to achieve high accuracy and robustness.

# Features


> Multi-Model Support: Seamlessly integrates pretrained transformer models (cardiffnlp/twitter-roberta-base-sentiment-latest) for sentiment-based classification, with fallback to a robust rule-based classifier if model loading fails or for lightweight operation.

> Enhanced Keyword Analysis: Includes ultra-comprehensive keyword dictionaries for each category (e.g., spam, promotions, personal) with weighted scoring for critical indicators like urgency, financial terms, or personal greetings. Regular expressions enhance detection of email structures specific to each category.

Device Optimization: Automatically detects and selects the optimal device (CPU or CUDA-enabled GPU) based on available memory (minimum 1.5GB GPU memory required for CUDA). Provides fallback to CPU if GPU resources are insufficient.

Batch Processing: Efficiently processes multiple emails with progress tracking, suitable for large-scale email analysis.

Detailed Analysis: Offers in-depth insights through category scoring, keyword matching counts, and pattern detection, enabling transparency in classification decisions.

Robust Error Handling: Gracefully handles model loading failures, CUDA errors, and long input texts (truncates at 1000 characters) to ensure reliability.

Contextual Indicators: Incorporates sender domain analysis, subject line prefixes, and content length heuristics to refine classification accuracy.

Extensive Category Support: Classifies emails into eight categories with detailed keyword groups, such as spam (urgent words, scam indicators), promotions (sales terms, discounts), and personal (greetings, emotions).
