# Pokémon Battle Winner Prediction (Gen 1 OU)

This repository contains the solutions for the **[FDS: Pokemon Battles prediction 2025](https://kaggle.com/competitions/fds-pokemon-battles-prediction-2025)** Kaggle challenge. The goal was to build a classification model that predicts the winner of a competitive Pokémon battle from the perspective of "Player 1," using a dataset of real human-vs-human battles from the Pokémon Showdown simulator.

## 🚀 Project Overview
The core of this challenge was to leverage a rich dataset containing initial team compositions and a detailed turn-by-turn battle timeline to predict the final outcome. Success was measured by prediction accuracy on a hidden test set.

The problem required significant feature engineering to translate the complex, turn-based dynamics and imperfect information of a Pokémon battle into a format suitable for a machine learning model. This repository includes the notebooks and code used for data exploration, feature creation, model training, and generating our final submissions.

## 🏆 Our Approach: Three Submissions
This repository contains the code for three distinct submission approaches, each building upon the last.

### Submission 1: Expected Damage & Stacked Ensemble (`submission_1.ipynb`)
This notebook represents a comprehensive, feature-heavy approach focused on predicting the battle outcome by calculating the *theoretical or expected* impact of moves and team compositions.

**Key Features Engineered:**
- **Expected Damage:** Calculates the expected damage for every move in the battle timeline based on the classic Gen 1 damage formula, including stats, boosts, STAB, and type effectiveness.
- **Cumulative Stats:** Aggregates team-level statistics (mean, variance, sum) for HP, Attack, Defense, etc., to model the overall power and balance of each team.
- **Behavioral Counts:** Generates a wide array of features by counting events like status moves, physical/special attacks, type effectiveness, and status conditions inflicted.
- **"Diff" Features:** For almost every feature, a corresponding "difference" feature (e.g., `mean_spe_diff`) was created to directly model the imbalance between Player 1 and Player 2.
  
**Modeling Approach:**
- **Stacked Ensemble:** A `StackingClassifier` combines the predictions of multiple base models (XGBoost, RandomForest, CatBoost) and feeds them into a final meta-estimator (Logistic Regression).
- **Hyperparameter Tuning:** `HalvingGridSearchCV` was used for efficient hyperparameter tuning of the base models.
- **Probability Calibration:** A `CalibratedClassifierCV` was trained on the final stacked model to improve the reliability of its predicted probabilities.
- **Dimensionality Reduction:** The notebook evaluates the impact of applying PCA, which showed a slight improvement in accuracy.

### Submission 2: Actual Damage & Meta-Learner Comparison (`submission_2.ipynb`)
This version evolves the first by refining the feature engineering from *theoretical* damage to *actual* observed damage and experimenting with the stacked ensemble's architecture.

**Key Feature Changes:**
- **Actual Damage Calculation:** Replaced "Expected Damage" with a more accurate "Actual Damage" feature, calculated by tracking the `hp_pct_drop` of a Pokémon between turns.
- **Expanded "Effects" Features:** Explicitly counts the occurrences of specific in-battle effects like `substitute`, `reflect`, and `confusion` for a more granular view of the battle state.

**Modeling Experimentation:**
- **Meta-Learner Comparison:** This notebook trains and compares two `StackingClassifier` versions: one with **Logistic Regression** and another with a tuned **GradientBoostingClassifier** as the final estimator. The Gradient Boosting model proved to be slightly superior.

### Submission 3: Strategic Archetypes & Blended Ensemble (`submission_3.ipynb`)
This notebook takes a different approach, modeling the battle based on higher-level strategic archetypes and the end-of-timeline game state.

**Key Features Engineered:**
- **Strategic Move Counts:** Focuses on the *type* of moves used, counting key strategic moves like `Sleep Powder` (sleep), `Swords Dance` (setup), and `Recover` (recovery) to classify a team's playstyle.
- **End-of-Timeline State:** Creates powerful features summarizing the game state *after* 30 turns, including average remaining HP, number of fainted Pokémon, and net stat boosts.
- **Pre-Battle Analysis:** Includes features based purely on the initial team matchup, such as aggregate team stats, overall type advantage, and a count of "top-tier" meta-defining Pokémon.
- **Refined Stat Scaling:** Uses a more precise, official Gen 1 stat calculation formula that accounts for DVs and EVs.

**Modeling Approach:**
- **Simple Blending Ensemble:** This version moves away from stacking and instead trains three powerful gradient boosting models (**XGBoost**, **LightGBM**, and **CatBoost**) independently. The final prediction is a simple, robust average (blend) of the probabilities from these three models.

## 📊 About the Data
The dataset is provided in a `.jsonl` format, where each line represents a battle. Key data points include:
- **Initial Team Details:** Full teams for Player 1 and the lead for Player 2.
- **Battle Timeline:** A turn-by-turn summary of the first 30 active turns.
- **Outcome (`player_won`):** The target variable (true or false).

## 📂 Repository Structure
```
.
├── data/                     # (Create this folder for the dataset)
│   ├── train.jsonl
│   └── test.jsonl
├── submission_1.ipynb        # Expected Damage & Stacking
├── submission_2.ipynb        # Actual Damage & Meta-Learner Comparison
├── submission_3.ipynb        # Strategic Archetypes & Blending
├── requirements.txt          # Project dependencies
└── README.md
```

## ⚙️ How to Run
To reproduce these results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Set up the environment:**
    It's recommended to create a virtual environment. Then, install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(Key libraries include: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`)*

3.  **Place the data:**
    Download the competition data (`train.jsonl`, `test.jsonl`) from the [Kaggle competition page](https://kaggle.com/competitions/fds-pokemon-battles-prediction-2025) and place it in the `data/` directory.

4.  **Run the notebooks:**
    Execute the Jupyter Notebooks (`submission_1.ipynb`, `submission_2.ipynb`, `submission_3.ipynb`) to see the data processing, model training, and prediction generation for each approach.

## 🙏 Acknowledgments
-   Thanks to the organizers of the challenge for providing a fun and complex problem.
-   Data was sourced from the [Pokémon Showdown](https://pokemonshowdown.com/) battle simulator.

---

## 📜 Citation
If you use or refer to this project, please cite the original competition organizers:

> Spinelli, I., Rocci, L., & Facchiano, S. (2025). *FDS: Pokemon Battles prediction 2025*. Kaggle. Retrieved from https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025

<details>
<summary>BibTeX Entry</summary>

```bibtex
@misc{fds-pokemon-battles-prediction-2025,
    author = {Indro Spinelli and Leonardo Rocci and Simone Facchiano},
    title = {FDS: Pokemon Battles prediction 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/fds-pokemon-battles-prediction-2025}},
    note = {Kaggle}
}
```
</details>
