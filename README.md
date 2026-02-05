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

## Model Accuracy

| Algorithm | Accuracy |
|-----------|----------|
| XGBoost | 89.55% |
| LightGBM | 88.36% |
| Logistic Regression | 77.51% |
| **Ensemble Average** | **~85%** |

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
