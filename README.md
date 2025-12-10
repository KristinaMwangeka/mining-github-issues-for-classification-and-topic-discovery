# Github Issues Classification and Topic Discovery:

## Project Overview

Efficient management and triage of client issues is critical, especially in modern software development with open-source collaboration. GitHub Issues is a core feature of GitHub that serves as a collaborative system for users  (developers, contributors, and end-users) to track tasks, bugs, feature requests, and general questions related to a software repository.
GitHub. I aim to automate three core tasks: Issue type (Bug, Feature, Documentation, etc.), & priority (High, Normal, Low) classification, and topic discovery. The primary challenges encountered were the inherent high-dimensional noise of GitHub issues text (code snippets and URLs) and a severe class imbalance dominated by 'Bug' issues. 

## Project Goals

*   **Problem Classification**: Accurately categorize GitHub issues into predefined types (e.g., Bug, Feature, Documentation, Enhancement, Question) using machine learning.
*   **Priority Classification**: Assign appropriate priority levels (High, Normal, Low) to issues based on their content and associated labels.
*   **Topic Discovery**: Uncover underlying themes and subjects within issue descriptions, especially for new or uncategorized issues, to gain deeper insights and identify emerging trends.
*   **Automation Potential**: Explore the combination of these models for an automated issue triaging and analysis system.

## Dataset

The project utilizes the `github_issues_tickets.json` dataset, which contains a collection of GitHub issues with various attributes such as title, body, labels, creation date, and more. This dataset is available on [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/helpdesk-github-tickets?select=github_issues_tickets.json).

## Methodology

### 1. Data Loading & Initial Exploration

*   The `github_issues_tickets.json` file was loaded into a Pandas DataFrame (`df_raw`).
*   Initial exploration included checking the dataset's shape, column information, data types, date ranges (`created_at`), and average length of the `body` column.

### 2. Data Cleaning & Preprocessing

*   **Handling Missing Values**: Missing values in the `body` column were filled with empty strings.
*   **Text Cleaning**: A `clean_text_body` function was developed to:
    *   Remove code blocks (````` ``` `````) and replace them with `[CODE_BLOCK]`.
    *   Remove inline code (`` `...` ``) and replace them with `[INLINE_CODE]`.
    *   Remove URLs and replace them with `[URL]`.
    *   Normalize whitespace.
*   **Feature Engineering**: 
    *   `code_block_count`: Count of code blocks in the issue body.
    *   `combined_text`: Concatenation of the issue `title` and `clean_body`.
    *   `issue_type_3` & `issue_type_4`: Two taxonomies for issue categorization based on `labels` (3-class and 4-class).
    *   The 4-class taxonomy (`issue_type_4`) was selected as the final `issue` type due to its granularity.

### 3. Issue Taxonomy & Priority Classification

*   **Issue Type**: Issues were categorized into 'Bug', 'Feature', 'Documentation', 'Enhancement', 'Question', or 'Other' based on keywords in their labels.
*   **Priority Classification**: A `get_priority_level` function was implemented to assign 'High', 'Normal', or 'Low' priorities using a rule-based approach derived from established literature on GitHub priority labels (Caddy & Treude, 2024). Keywords like 'critical', 'urgent', 'high' indicate 'High' priority, while 'low', 'stale', 'wontfix' indicate 'Low' priority.

### 4. Supervised Classification Models

Issues categorized as 'Other' were filtered out (`df_model`) for supervised learning.

*   **TF-IDF Vectorization**: `TfidfVectorizer` (with `max_features=5000` and `stop_words='english'`) was used to convert `combined_text` into numerical features.
*   **Train-Test Split**: Data was split into 80% training and 20% testing sets, stratified by the `issue` column (`random_state=42`).

#### a. Support Vector Machine (SVM - `LinearSVC`)

*   **Purpose**: Classify issues into predefined types ('Bug', 'Documentation', 'Enhancement', 'Feature', 'Question').
*   **Hyperparameter Tuning (`GridSearchCV`)**: Optimized for `f1_macro` with parameters `C` and `class_weight`.
    *   **Best Parameters**: `{'C': 1, 'class_weight': 'balanced'}`.
*   **Performance**: Achieved **91% accuracy** and a macro-averaged F1-score of **0.46**. Excellent performance on 'Bug' (F1: 0.95), but struggled with minority classes like 'Question' (F1: 0.06).

#### b. Random Forest Classifier

*   **Purpose**: Alternative classifier, integrating TF-IDF features with `code_block_count`.
*   **Hyperparameter Tuning (`GridSearchCV`)**: Optimized for `f1_macro` with parameters `n_estimators`, `max_depth`, and `class_weight`.
    *   **Best Parameters**: `{'class_weight': 'balanced_subsample', 'max_depth': 20, 'n_estimators': 100}`.
*   **Performance**: Achieved **78% accuracy** and a macro-averaged F1-score of **0.35**. Performed well on 'Bug' (F1: 0.88), but struggled more severely with minority classes, failing to predict any 'Question' issues (F1: 0.00).

### 5. Unsupervised Topic Modeling

#### a. Latent Dirichlet Allocation (LDA)

*   **Purpose**: Discover latent topics based on word frequency.
*   **Vectorization**: `CountVectorizer` (with custom GitHub stop words) was used.
*   **Hyperparameter Tuning (`GridSearchCV`)**: Optimized for log likelihood with parameters `n_components` and `learning_decay`.
    *   **Best Parameters**: `{'learning_decay': 0.5, 'n_components': 3}`.
*   **Discovered Topics**: Identified 3 broad topics, with keywords reflecting general themes.

#### b. BERTopic

*   **Purpose**: Advanced topic modeling leveraging transformer embeddings for semantic understanding.
*   **Methodology**: Uses `all-MiniLM-L6-v2` for embeddings, UMAP for dimensionality reduction, HDBSCAN for clustering (`min_topic_size=30`), and c-TF-IDF for topic representation.
*   **Discovered Topics**: Identified 86 distinct topics. Provided coherent topics (e.g., `table, php, database, query` and `dark, background, theme, mode`).
*   **Noise Cluster**: A significant portion, **47.1%** of documents, were classified as noise (Topic ID -1), indicating issues that didn't form dense, coherent clusters.

## Key Findings & Conclusions

*   **Classification Models (SVM vs. Random Forest)**:
    *   The **Tuned SVM Model performed better overall**, demonstrating higher accuracy (91% vs. 78%) and a superior macro-averaged F1-score (0.46 vs. 0.35). SVM showed more robust performance across minority classes, outperforming Random Forest, especially for the 'Question' class.
*   **Topic Models (LDA vs. BERTopic)**:
    *   **BERTopic offered superior topic discovery** by understanding semantic nuances, making it more suitable for discerning client issues. LDA grouped topics by frequency, which was less effective for specific issue identification.
    *   A very low Adjusted Rand Index (ARI) of **0.020** indicated minimal agreement between LDA and BERTopic topic assignments, highlighting their fundamentally different approaches to topic identification.
*   **Automation Potential**: The project confirms that these models can be combined for automation. SVM can perform the initial classification of well-defined issue types, while BERTopic can provide valuable, semantically rich topic discovery for less structured or emerging issues, even potentially integrating with zero-shot classification for companies with dynamic issue categories.

## Reproducibility

To ensure the reproducibility of the analysis and results presented in this notebook, please refer to the detailed [Reproducibility Section](#reproducibility) within the notebook.

## AI Disclosure

This project utilized Google Gemini to generate portions of the code, which was then manually verified for logic and correctness.
