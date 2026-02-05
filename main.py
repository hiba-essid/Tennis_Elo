"""
Tennis Elo Prediction System - Men Only
Using local data from: data/raw/ATP Stats.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from src.elo_system import TennisEloSystem
from src.predictor import GrandSlamPredictor
from src.grand_slam import GrandSlamTournament
from src.utils import load_kaggle_mens_grand_slam_data, save_model, load_model


def main():
    print("=" * 70)
    print("🎾 TENNIS ELO PREDICTION SYSTEM - MEN'S GRAND SLAMS (LOCAL DATA)")
    print("=" * 70)
    print("🏆 Predicting Winners & Finalists for:")
    print("   • Australian Open (Hard)")
    print("   • French Open (Clay)")
    print("   • Wimbledon (Grass)")
    print("   • US Open (Hard)")
    print("=" * 70)
    
    # Configuration
    data_path = 'data/raw/ATP Stats.csv'
    n_simulations = 20  # More = more accurate (but slower)
    
    # ========================================================================
    # STEP 1: LOAD LOCAL DATA
    # ========================================================================
    print("\n📊 STEP 1: Loading Men's Grand Slam Data...")
    print("-" * 70)
    
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        return
    
    men_df = load_kaggle_mens_grand_slam_data(data_path)
    
    if men_df is None or len(men_df) == 0:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(men_df):,} Grand Slam matches")
    
    # ========================================================================
    # STEP 2: TRAIN ELO SYSTEM (MEN ONLY)
    # ========================================================================
    print("\n⚡ STEP 2: Training Men's Elo System...")
    print("-" * 70)
    print("👨 Processing {0:,} matches...".format(len(men_df)))
    
    men_elo = TennisEloSystem(initial_rating=1500, k_factor=32)
    men_elo.process_matches(men_df)
    
    print("✅ Elo system trained!")
    
    # Save Elo system
    save_model(men_elo, 'models/men_elo_system.pkl')
    print("💾 Saved: models/men_elo_system.pkl")
    
    # ========================================================================
    # STEP 3: GET TOP PLAYERS
    # ========================================================================
    print("\n🏆 STEP 3: Identifying Top Players...")
    print("-" * 70)
    
    # Get top 128 for each surface
    men_players_hard = [p[0] for p in men_elo.get_top_players('hard', 128)]
    men_players_clay = [p[0] for p in men_elo.get_top_players('clay', 128)]
    men_players_grass = [p[0] for p in men_elo.get_top_players('grass', 128)]
    
    print(f"✅ Hard court: {len(men_players_hard)} players")
    print(f"✅ Clay court: {len(men_players_clay)} players")
    print(f"✅ Grass court: {len(men_players_grass)} players")
    
    # Display top 15 overall
    print("\n👨 TOP 15 PLAYERS (Overall Elo):")
    for rank, (player, rating) in enumerate(men_elo.get_top_players('overall', 15), 1):
        print(f"   {rank:2}. {player:<35} {rating:.0f}")
    
    # ========================================================================
    # STEP 4: TRAIN ML PREDICTOR
    # ========================================================================
    print("\n🤖 STEP 4: Training ML Predictor (Men Only)...")
    print("-" * 70)
    print("Creating features from match data...")
    
    predictor = GrandSlamPredictor(men_elo)
    
    # Prepare training data for each surface
    X_hard, y_hard = predictor.prepare_training_data(men_df, surface='hard', verbose=True)
    X_clay, y_clay = predictor.prepare_training_data(men_df, surface='clay', verbose=True)
    X_grass, y_grass = predictor.prepare_training_data(men_df, surface='grass', verbose=True)
    
    # Combine all surfaces (using numpy concatenation)
    X_all = np.vstack([X_hard, X_clay, X_grass])
    y_all = np.hstack([y_hard, y_clay, y_grass])
    
    print(f"\n📊 Total training samples: {len(X_all):,}")
    
    # Train the predictor
    print("\n🔧 Training ensemble (XGBoost + LightGBM + Logistic Regression)...")
    predictor.train(X_all, y_all)
    
    print("✅ ML predictor trained!")
    
    # Save predictor
    save_model(predictor, 'models/unified_predictor.pkl')
    print("💾 Saved: models/unified_predictor.pkl")
    
    # ========================================================================
    # STEP 5: SIMULATE 2026 GRAND SLAMS
    # ========================================================================
    print("\n🎲 STEP 5: Simulating 2026 Grand Slams...")
    print("-" * 70)
    print(f"Running {n_simulations:,} simulations per tournament...\n")
    
    slams = {
        'Australian Open': 'hard',
        'French Open': 'clay',
        'Wimbledon': 'grass',
        'US Open': 'hard'
    }
    
    all_predictions = {}
    
    for slam_name, surface in slams.items():
        print(f"🎾 {slam_name} ({surface})...", end=' ', flush=True)
        
        # Get top 128 players for this surface
        top_players = [p[0] for p in men_elo.get_top_players(surface, 128)]
        
        # Simulate tournament
        tournament = GrandSlamTournament(predictor, slam_name, 'men', men_elo)
        predictions = tournament.simulate_tournament(top_players, n_simulations)
        
        all_predictions[slam_name] = predictions
        
        # Show top 5
        print(f"✅ ({len(predictions)} players)")
        print(f"   Top 5 predictions:")
        for i, row in predictions.head(5).iterrows():
            print(f"      {i+1:2}. {row['Player']:<30} Win: {row['Win_Probability']*100:5.1f}% Final: {row['Final_Probability']*100:5.1f}%")
    
    # ========================================================================
    # STEP 6: SUMMARY REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("📋 2026 GRAND SLAM PREDICTIONS - SUMMARY")
    print("=" * 70)
    
    for slam_name in slams.keys():
        predictions = all_predictions[slam_name]
        print(f"\n🏆 {slam_name}")
        print("-" * 50)
        print("Top 5 Contenders:")
        for i, row in predictions.head(5).iterrows():
            print(f"   {i+1}. {row['Player']:<35} {row['Win_Probability']*100:5.1f}%")
    
    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================
    print("\n💾 Saving predictions...")
    os.makedirs('data/processed', exist_ok=True)
    
    for slam_name, predictions in all_predictions.items():
        safe_name = slam_name.lower().replace(' ', '_')
        filepath = f'data/processed/predictions_{safe_name}_2026.csv'
        predictions.to_csv(filepath, index=False)
        print(f"   ✅ {filepath}")
    
    print("\n" + "=" * 70)
    print("✨ PREDICTIONS COMPLETE!")
    print("=" * 70)
    print("\n🌐 To view predictions in web interface:")
    print("   python app.py")
    print("   Then visit: http://localhost:5000")
    print()


if __name__ == '__main__':
    main()
