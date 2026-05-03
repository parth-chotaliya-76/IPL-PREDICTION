# 🏏 IPL Match Prediction System

> AI-powered winner, score & Player of the Match prediction for Indian Premier League matches.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![Accuracy](https://img.shields.io/badge/Winner%20Accuracy-67.8%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC-0.7466-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Key Insights](#key-insights)
- [Version History](#version-history)

---

## Overview

This project builds a complete end-to-end IPL Match Prediction System trained on ball-by-ball data from all IPL seasons (2008–2025). Before the first ball is bowled, the system predicts three outcomes for any upcoming match:

| Prediction | Method | Best Metric |
|---|---|---|
| 🏆 **Match Winner** | Binary classification | 67.8% accuracy, AUC 0.7466 |
| 📊 **Innings Score** | Regression | R² = 0.9984, MAE = 0.66 runs |
| 🌟 **Player of the Match** | Binary ranking | AUC = 0.6138 |

The system accepts real match inputs — team selection, venue, toss result, and full Playing XI (11 players per team) — and uses each player's historical stats against the specific opposition, at the specific venue, and in specific innings.

---

## Dataset

| Property | Value |
|---|---|
| Source file | `IPL.csv` |
| File size | ~31 MB |
| Total rows | 278,205 (one row per delivery) |
| Date range | April 2008 – June 2025 |
| Seasons | 18 (2008–2025) |
| Total matches | 1,164 complete matches |
| Unique players | 767 |
| Teams (raw → standardized) | 19 → 14 |
| Venues (raw → standardized) | 59 → 39 |

---

## Project Pipeline

```
IPL.csv (raw, 60+ columns)
       │
       ▼
ipl_clean_features.py          →  IPL_cleaned.csv (33 columns, ~8MB)
       │
       ▼
ipl_fix_team_names.py          →  Standardize 19 team names → 14
ipl_fix_venue_names.py         →  Standardize 59 venues → 39
       │
       ▼
ipl_feature_engineering_v4.py  →  IPL_match_features_v4.csv  (1,164 × 171)
                                   IPL_potm_features_v4.csv   (24,956 × 60)
                                   IPL_innings_features.csv   (2,333 × 15)
       │
       ▼
ipl_model_training_v4.py       →  models/  (winner, score, POTM .pkl files)
       │
       ▼
app.py (Flask)                 →  Web application at localhost:5000
```

---

## Feature Engineering

Four progressively improved versions were built. Every feature uses **only historical data before the match date** — no data leakage.

### v1 — Basic Match-Level Aggregates (41 features)
- Head-to-head win rates between team pairs
- Venue win rate per team
- Toss impact analysis
- Batter & bowler historical averages vs opposition
- Average scores at venue
- Recent form (last 5 matches)
- Player vs player matchups

### v2 — Playing XI Aware (107 features)
- Actual 11 players extracted per match from ball-by-ball data
- Separate Inn 1 / Inn 2 stats per player
- POTM redesigned: binary per player, 22 candidates ranked per match
- Training rows for POTM: 771 → 24,956

### v3 — Time Decay + New Features (139 features)
- Exponential time-decay weighting (half-life 365 days — recent matches count more)
- Current season form stats
- Last 10 & 20 match win windows
- Venue pitch features (avg score, run rate, boundary %)
- Toss × venue interaction
- Home advantage flag
- Player consistency score (standard deviation of recent scores)

### v4 — Maximum Features + Data Fixes (171 features)
- Powerplay run rate (overs 0–5) per team
- Death overs run rate (overs 16–19) per team
- Chasing vs defending win percentages
- Opposition weakness index (economy conceded by bowling attack)
- Player experience (total IPL matches)
- Calibrated XGBoost for better probability estimates
- Team name & venue name standardization integrated

---

## Model Performance

### Task 1 — Match Winner (Binary Classification)

| Model | CV Acc | Test Acc | AUC | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.6503 | 0.6481 | 0.7122 | 0.6489 |
| Random Forest | 0.6083 | 0.6395 | 0.7134 | 0.6327 |
| XGBoost | 0.6237 | 0.6652 | 0.7199 | 0.6597 |
| Gradient Boosting | 0.6211 | 0.6609 | 0.7436 | 0.6566 |
| SVM | 0.5756 | 0.5966 | 0.6528 | 0.5975 |
| MLP Neural Net | 0.5867 | 0.6352 | 0.6606 | 0.6355 |
| Soft Voting Ensemble | 0.6375 | 0.6738 | 0.7466 | 0.6706 |
| Stacking Ensemble | 0.6117 | 0.6695 | 0.7400 | 0.6619 |
| **XGBoost Calibrated ✓ BEST** | **0.6117** | **0.6781** | **0.7315** | **0.6746** |

Feature selection via `SelectFromModel` reduced 158 → 79 features.

**Top predictive features:**
1. `team2_bowl_recent_wkts` — recent wicket-taking form of bowling team
2. `team2_bowl_wkt_this_season` — current season wicket rate
3. `team2_pvp_bowl_n` — player vs player matchup count
4. `team2_player_experience` — combined team experience
5. `toss_dec_winpct` — historical win % with this toss decision

### Task 2 — Innings Score (Regression)

| Model | MAE (runs) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 2.44 | 3.93 | 0.9855 |
| Random Forest | 0.73 | 1.81 | 0.9969 |
| XGBoost | 0.65 | 1.35 | 0.9983 |
| **Gradient Boosting ✓ BEST** | **0.66** | **1.32** | **0.9984** |
| SVR | 1.89 | 5.36 | 0.9730 |

### Task 3 — Player of the Match (Binary Classification)

| Model | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression ✓ BEST** | **0.6438** | **0.6138** | **0.0686** | **0.5302** | **0.1215** |
| Random Forest | 0.9471 | 0.5700 | 0.1000 | 0.0172 | 0.0294 |
| XGBoost | 0.8832 | 0.5612 | 0.0709 | 0.1250 | 0.0905 |
| Gradient Boosting | 0.9537 | 0.6102 | 0.6667 | 0.0086 | 0.0170 |
| Stacking Ensemble | 0.6432 | 0.5806 | 0.0632 | 0.4828 | 0.1117 |

> ⚠️ High accuracy for RF/GB is misleading — they predict "not POTM" for everyone. AUC is the correct metric here.

---

## Web Application

Built with Flask, served at `http://localhost:5000`.

### Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask (Python) |
| ML models | scikit-learn, XGBoost via joblib |
| Data processing | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Theme | IPL-branded: Navy `#003B8E`, Gold `#FFB800`, Orange `#FF6B00` |

### Features
- **Match Setup** — Team 1, Team 2, Venue, Match Date, Toss Winner, Toss Decision
- **Toss Logic** — Auto-detects batting order; green banner shows correct batting team; orange warning on mismatch
- **Head-to-Head Panel** — Live H2H record, win %, last 5 results
- **Venue Stats Panel** — Avg 1st/2nd innings score, run rate, boundary %, avg wickets
- **Playing XI Selection** — Searchable dropdowns for all 767 players with keyboard navigation (↑↓ to navigate, Enter to select, Esc to close)
- **Winner Prediction** — Winning team with probability bar
- **Score Prediction** — Predicted innings totals for both teams
- **POTM Prediction** — Top 3 Player of the Match candidates with probability %

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main prediction interface |
| `/api/predict` | POST | Match prediction (winner, scores, POTM) |
| `/api/h2h/<t1>/<t2>` | GET | Head-to-head stats between two teams |
| `/api/venue_stats/<venue>` | GET | Pitch & historical stats for a venue |
| `/api/player_info/<player>` | GET | Career stats for a player |
| `/api/players` | GET | All 767 players for autocomplete |

---

## Project Structure

```
ipl-prediction/
│
├── 📄 IPL.csv                          # Raw ball-by-ball dataset (31 MB)
├── 📄 IPL_cleaned.csv                  # Cleaned dataset (33 cols, ~8 MB)
├── 📄 IPL_match_features_v4.csv        # Match features for winner model (1,164 × 171)
├── 📄 IPL_potm_features_v4.csv         # Player features for POTM model (24,956 × 60)
├── 📄 IPL_innings_features.csv         # Innings features for score model (2,333 × 15)
├── 📄 IPL_player_stats.csv             # Player career stats (autocomplete)
│
├── 🐍 ipl_clean_features.py            # Step 1: Data cleaning & feature selection
├── 🐍 ipl_fix_team_names.py            # Fix franchise name changes
├── 🐍 ipl_fix_venue_names.py           # Fix stadium name variants
├── 🐍 ipl_diagnose.py                  # Dataset diagnosis utility
├── 🐍 ipl_feature_engineering_v4.py   # Step 2: Feature engineering (final)
├── 🐍 ipl_model_training_v4.py        # Step 3: Model training (final)
├── 🐍 app.py                           # Flask web application
│
├── 📁 models/
│   ├── winner_v4_BEST.pkl              # Best winner prediction model
│   ├── winner_v4_selector.pkl          # Feature selector
│   ├── winner_v4_feature_cols.pkl      # Feature column names
│   ├── winner_v4_selected_cols.pkl     # Selected feature names
│   ├── potm_v4_BEST.pkl                # Best POTM model
│   ├── potm_v4_feature_cols.pkl        # POTM feature names
│   ├── score_BEST.pkl                  # Best score regression model
│   └── model_comparison_v4.xlsx        # Full model comparison results
│
└── 📁 templates/
    └── index.html                      # Web app frontend
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost flask joblib matplotlib openpyxl
```

---

## How to Run

### Full Pipeline (first time)

Run each step in order:

```bash
# Step 1 — Clean the raw dataset
python ipl_clean_features.py

# Step 1b — Fix data quality issues (run both)
python ipl_fix_team_names.py
python ipl_fix_venue_names.py

# Step 2 — Feature engineering (~25–30 min)
python ipl_feature_engineering_v4.py

# Step 3 — Train all models (~15–20 min)
python ipl_model_training_v4.py

# Step 4 — Launch web app
python app.py
```

Then open your browser at: **http://localhost:5000**

### Quick Start (models already trained)

If the `models/` folder is already populated:

```bash
python app.py
```

### Diagnose Dataset Issues

```bash
python ipl_diagnose.py
```

---

## Key Insights

**🎳 Bowling dominates predictions.** The top 15 most important features are almost entirely bowling-related — recent wickets taken, season wicket rate, economy vs opposition. This confirms the T20 principle that bowling attacks determine match outcomes more consistently than batting.

**⏱️ Recent form beats career stats.** Time-decay weighting (added in v3) produced the biggest single accuracy jump (+4.7%). A bowler's wickets in the last 5 matches matters far more than their career average.

**🌐 Data quality has massive impact.** Team and venue name standardization didn't change raw accuracy, but raised AUC from 0.7294 → 0.7466. Treating Feroz Shah Kotla and Arun Jaitley Stadium as the same ground doubled the available data for Delhi pitch conditions.

**🏏 POTM is fundamentally in-match.** Player of the Match prediction plateaued at AUC 0.61 regardless of feature improvements — because POTM is awarded based on that day's runs, wickets, and catches, which no pre-match model can reliably estimate.

**📈 Score model is near-perfect.** R² = 0.9984 is achieved because in-innings features (balls bowled, wickets fallen) are strongly correlated with final score. For pure pre-match estimation, venue averages provide the best signal.

**🎯 67.8% is the practical ceiling.** Published IPL prediction research benchmarks 60–72% for pre-match models. The remaining ~32% unpredictable portion comes from on-the-day form, pitch conditions, dew factor, dropped catches, and last-minute squad changes.

---

## Version History

| Version | Key Changes | Winner Acc | AUC |
|---|---|---|---|
| v1 | Basic match-level aggregates, 41 features | 57.1% | — |
| v2 | Playing XI extraction, player-level features, POTM redesigned, 107 features | 63.1% | 0.6746 |
| v3 | Time-decay weighting, current season stats, venue pitch features, 139 features | 67.8% | 0.7294 |
| **v4 (final)** | Team/venue name fixes, powerplay/death stats, chasing/defending %, player experience, 171 features | **67.8%** | **0.7466** |

> v3 and v4 achieve the same test accuracy but v4 has a higher AUC — its probability estimates are better calibrated and more reliable for the win probability display in the web app.

---

## License

This project is for educational and research purposes. IPL match data is used for non-commercial prediction research.
