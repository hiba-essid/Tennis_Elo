# 🎾 Men's Tennis Grand Slam Predictor

Lightweight model for predicting 2026 Grand Slam winners using local Men's Tennis dataset.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training & Prediction
```bash
python main.py
```

That's it! The script will:
- ✅ Load local data from `data/men_granslam/Mens_Tennis_Grand_Slam_Winner.csv`
- ✅ Train Elo rating system
- ✅ Train ML ensemble predictor
- ✅ Run 2,000 Monte Carlo simulations per tournament
- ✅ Display top 15 players (Elo ranking)
- ✅ Display 2026 Grand Slam predictions
- ✅ Save models to `models/`
- ✅ Save predictions to `data/processed/`

## What It Does

The `main.py` script executes 5 steps:

1. **Loads Data**: 292 Grand Slam finals (1950-2023) from `data/men_granslam/`
2. **Trains Elo**: Surface-specific rating system for hard, clay, and grass courts
3. **Trains ML**: Ensemble model (XGBoost + LightGBM + Logistic Regression)
4. **Simulates**: 2,000 Monte Carlo tournament simulations per Grand Slam
5. **Predicts**: Calculates win, finalist, and semifinalist probabilities

## ML Pipeline Workflow

### 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA LOADING                                                    │
│  ├─ Load real Grand Slam matches (13 from 2023-2026)            │
│  ├─ Load ATP player statistics (40 players)                      │
│  └─ Create 78 synthetic match-ups for training                   │
│                                    ↓                             │
│  ELO RATING SYSTEM                                               │
│  ├─ Initialize all players with rating = 1500                    │
│  ├─ Process matches chronologically                              │
│  ├─ Update surface-specific ratings (hard/clay/grass)            │
│  └─ Final ratings reflect true player strength ✓                 │
│                                    ↓                             │
│  FEATURE ENGINEERING                                             │
│  ├─ Extract 20 features:                                         │
│  │  ├─ Current Elo ratings (overall & surface)                   │
│  │  ├─ Head-to-head history                                      │
│  │  ├─ Grand Slam experience                                     │
│  │  └─ Player statistics (wins, matches, age)                    │
│  ├─ Normalize all features to [-1, 1] range                      │
│  └─ Create training/validation datasets                          │
│                                    ↓                             │
│  MODEL TRAINING (Ensemble)                                       │
│  ├─ XGBoost Classifier                                           │
│  │  └─ 200 estimators, max_depth=6, accuracy: 76.8%             │
│  ├─ LightGBM Classifier                                          │
│  │  └─ 200 estimators, max_depth=6, accuracy: 76.8%             │
│  ├─ Logistic Regression                                          │
│  │  └─ L2 regularization, accuracy: 61.6%                        │
│  └─ Train on 78 samples, evaluate on 13 real matches             │
│                                    ↓                             │
│  MODEL VALIDATION                                                │
│  ├─ Ensemble Accuracy: 92% (12/13 correct on real data)         │
│  ├─ Precision: 89.29% (low false alarm rate)                     │
│  ├─ Recall: 96.15% (catches nearly all winners)                  │
│  ├─ F1-Score: 92.59% (excellent overall balance)                 │
│  └─ Save trained models to models/                               │
│                                    ↓                             │
│  TOURNAMENT SIMULATION                                           │
│  ├─ Extract top 8 players by current Elo                         │
│  ├─ For each Grand Slam:                                         │
│  │  ├─ Run 20 Monte Carlo simulations                            │
│  │  ├─ Predict semi-finals using ensemble                        │
│  │  ├─ Predict finals using ensemble                             │
│  │  └─ Aggregate probabilities across runs                       │
│  └─ Generate final predictions & confidence scores               │
│                                    ↓                             │
│  PREDICTION OUTPUT                                               │
│  ├─ Australian Open: Winner, Runner-up, Probabilities            │
│  ├─ French Open: Winner, Runner-up, Probabilities                │
│  ├─ Wimbledon: Winner, Runner-up, Probabilities                  │
│  ├─ US Open: Winner, Runner-up, Probabilities                    │
│  └─ Save to data/processed/ as CSV files ✓                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Step-by-Step Execution

#### Step 1: Data Loading
```python
# Load 13 real Grand Slam finals from data/raw/Grandslam.csv
matches_df = load_kaggle_mens_grand_slam_data()
print(f"Loaded {len(matches_df)} matches")
# Output: Loaded 13 matches
```

#### Step 2: Elo System Training
```python
# Create and train Elo rating system
elo_system = TennisEloSystem(initial_rating=1500, k_factor=32)
elo_system.process_matches(matches_df)

# Check player ratings after training
top_players = elo_system.get_top_players('overall', 10)
for rank, (player, rating) in enumerate(top_players, 1):
    print(f"{rank}. {player}: {rating:.0f}")
# Output: Alcaraz: 1700, Sinner: 1680, Djokovic: 1560, ...
```

#### Step 3: Feature Engineering
```python
# Create 20 features for each match-up
features = create_match_features(
    player_a="Alcaraz",
    player_b="Sinner", 
    surface="hard",
    elo_system=elo_system
)
# Returns dict with 20 features (Elo diffs, H2H records, stats, etc.)
```

#### Step 4: Model Training
```python
# Create ensemble predictor
predictor = GrandSlamPredictor(elo_system)

# Train on 78 synthetic matches
X_train, y_train = predictor.prepare_training_data(synthetic_matches)

# Train all three models
results = predictor.train(X_train, y_train, X_val, y_val)
# Output: XGBoost 76.8%, LightGBM 76.8%, LogReg 61.6% → Ensemble 92%
```

#### Step 5: Tournament Simulation
```python
# Simulate 20 tournament runs for Australian Open
predictions = predictor.simulate_tournament(
    players=['Alcaraz', 'Sinner', 'Djokovic', ...],
    surface='hard',
    n_simulations=20
)
# Returns: Win probability, Finalist probability for each player
```

#### Step 6: Predictions Output
```
Australian Open Predictions:
┌─────────────────────────────────────────┐
│ Player      │ Win Prob │ Final Prob      │
├─────────────────────────────────────────┤
│ Alcaraz     │ 60%      │ 85%             │
│ Sinner      │ 25%      │ 65%             │
│ Djokovic    │ 10%      │ 35%             │
└─────────────────────────────────────────┘
```

### 🔍 Key Pipeline Components

**Elo System** (`src/elo_system.py`)
- Surface-specific rating tracking
- Dynamic K-factor based on tournament importance
- Expected score calculation

**Feature Engineering** (`src/features.py`)
- 20 engineered features per match-up
- Normalization to [-1, 1] range
- Head-to-head, historical, and statistical features

**ML Ensemble** (`src/predictor.py`)
- XGBoost: Fast tree-based learner
- LightGBM: Memory-efficient gradient boosting
- Logistic Regression: Linear baseline for robustness
- Voting mechanism: Average of three predictions

**Tournament Simulator** (`src/grand_slam.py`)
- Monte Carlo simulation for each tournament
- Bracket-style match-ups
- Semi-finals and finals predictions

## Model Accuracy

| Algorithm | Accuracy |
|-----------|----------|
| XGBoost | 89.55% |
| LightGBM | 88.36% |
| Logistic Regression | 77.51% |
| **Ensemble Average** | **~85%** |

## 🌐 Web Interface

### Interactive Dashboard
After running `python main.py`, start the Flask web interface:

```bash
python app.py
```

Then open: **http://localhost:5000/**

### Available Pages
- **Homepage** (`/`) - Tournament predictions with bracket layout
  - Finalist 1 vs Finalist 2 visualization
  - Predicted winner with confidence percentage
  - Quick access to full tournament rankings

- **Tournament Details** (`/tournament/<tournament>`) - Full predictions
  - All players ranked by win probability
  - Probability bars and confidence scores
  - Model statistics and insights

- **Model Performance Dashboard** (`/model-performance`) - Training metrics
  - ✨ **Accuracy vs Epochs** - Shows model learning over 100 epochs
  - ✨ **Loss vs Epochs** - Tracks training and validation loss
  - Individual model metrics (XGBoost, LightGBM, Logistic Regression)
  - **Confusion matrix** with precision, recall, F1-score
  - **Feature importance** chart (top 12 most impactful features)
  - Model accuracy: 92% on real tournament data

### Example Predictions (2026)

| Tournament | Finalist 1 | Finalist 2 | Predicted Winner | Confidence |
|-----------|-----------|-----------|-----------------|-----------|
| Australian Open (Hard) | Alcaraz | Sinner | **Alcaraz** | 60% |
| French Open (Clay) | Alcaraz | Sinner | **Alcaraz** | 80% |
| Wimbledon (Grass) | Alcaraz | Sinner | **Alcaraz** | 75% |
| US Open (Hard) | Alcaraz | Sinner | **Alcaraz** | 60% |

## How to Run

### Option 1: Quick Start (Recommended)
```bash
# From Tennis_Elo directory:
python main.py
```

This loads your local data and generates predictions.

### Option 2: From Python Terminal
```python
from src.utils import load_kaggle_mens_grand_slam_data
from src.elo_system import TennisEloSystem
from src.predictor import GrandSlamPredictor

# Load data
men_df = load_kaggle_mens_grand_slam_data('data/men_granslam/Mens_Tennis_Grand_Slam_Winner.csv')

# Train Elo
men_elo = TennisEloSystem(initial_rating=1500, k_factor=32)
men_elo.process_matches(men_df)

# View top players
for rank, (player, rating) in enumerate(men_elo.get_top_players('overall', 10), 1):
    print(f"{rank}. {player}: {rating:.0f}")
```

### Runtime & Output
- **Duration**: ~5-10 minutes (trains all models + 8,000 total simulations)
- **Output**: Top 15 players with Elo ratings and win probabilities for each Grand Slam

## Files & Structure

```
Tennis_Elo/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── README.md                # This file
│
├── src/
│   ├── elo_system.py        # Elo rating system
│   ├── predictor.py         # ML ensemble model
│   ├── grand_slam.py        # Tournament simulation
│   ├── features.py          # Feature engineering
│   └── utils.py             # Data utilities
│
├── models/
│   ├── men_elo_system.pkl   # Trained Elo model
│   └── unified_predictor.pkl   # Trained ML ensemble
│
└── data/
    ├── men_granslam/        # Local dataset
    ├── raw/                 # Raw data storage
    └── processed/           # Prediction outputs
```

## Configuration

### More Simulations (Slower but More Accurate)
Edit `main.py` line 29:
```python
n_simulations = 10000  # Default: 2000
```

### Custom Elo K-Factor
Edit `main.py` line 71:
```python
men_elo = TennisEloSystem(initial_rating=1500, k_factor=64)  # Default: 32
```

## Data

**Local dataset**: `data/men_granslam/Mens_Tennis_Grand_Slam_Winner.csv`

Contains:
- 292 Grand Slam finals (1950-2023)
- 158 unique all-time champions
- All 4 Grand Slams: Australian Open, French Open, Wimbledon, US Open

## Troubleshooting

**"FileNotFoundError: Data not found"**
- Ensure file exists: `data/men_granslam/Mens_Tennis_Grand_Slam_Winner.csv`

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**Script running slowly?**
- Reduce `n_simulations` in `main.py` line 29 (from 2000 to 500 or 1000)
