# Email-Classification

A robust Python-based email classification system leveraging pretrained transformer models (DistilBERT, BERT, RoBERTa) and enhanced rule-based methods to categorize emails into eight categories: Updates, Personal, Promotions, Forums, Purchases, Travel, Spam, and Social. Optimized for both CPU and NVIDIA T500 2GB GPU, it features comprehensive keyword pattern matching and contextual analysis for accurate classification.

# Features

Multi-Model Support: Utilizes DistilBERT, BERT, or RoBERTa with a fallback to a rule-based classifier.
Enhanced Keyword Analysis: Extensive keyword dictionaries and regex patterns for precise category detection.
Device Optimization: Automatically selects CPU or GPU based on available resources.
Batch Processing: Efficiently classifies multiple emails with progress tracking.
Detailed Analysis: Provides in-depth category scoring and keyword matching insights.
