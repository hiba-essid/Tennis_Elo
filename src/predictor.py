"""
Grand Slam prediction model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb

from .features import create_match_features


class GrandSlamPredictor:
    def __init__(self, elo_system):
        """
        Initialize Grand Slam predictor
        
        Args:
            elo_system: TennisEloSystem instance
        """
        self.elo_system = elo_system
        self.feature_names = None
        
        # Ensemble of models
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            ),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        
    def prepare_training_data(self, matches_df, surface='hard', verbose=True):
        """Convert matches to feature vectors"""
        X = []
        y = []
        
        if verbose:
            print(f"Preparing training data from {len(matches_df)} matches...")
        
        for idx, match in matches_df.iterrows():
            if verbose and idx % 500 == 0:
                print(f"  Processing {idx}/{len(matches_df)}")
            
            features = create_match_features(
                match['winner'], 
                match['loser'],
                surface,
                self.elo_system
            )
            
            if self.feature_names is None:
                self.feature_names = list(features.keys())
            
            X.append(list(features.values()))
            y.append(1)
            
            features_rev = create_match_features(
                match['loser'],
                match['winner'],
                surface,
                self.elo_system
            )
            X.append(list(features_rev.values()))
            y.append(0)
        
        if verbose:
            print(f"✅ Created {len(X)} samples with {len(self.feature_names)} features")
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Train all models in ensemble"""
        results = {}
        
        for name, model in self.models.items():
            if verbose:
                print(f"\nTraining {name}...")
            
            model.fit(X_train, y_train)
            
            train_pred = model.predict_proba(X_train)[:, 1]
            train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
            train_logloss = log_loss(y_train, train_pred)
            
            if verbose:
                print(f"  Train Accuracy: {train_acc:.4f}")
                print(f"  Train LogLoss: {train_logloss:.4f}")
            
            if X_val is not None and y_val is not None:
                val_pred = model.predict_proba(X_val)[:, 1]
                val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
                val_logloss = log_loss(y_val, val_pred)
                
                if verbose:
                    print(f"  Val Accuracy: {val_acc:.4f}")
                    print(f"  Val LogLoss: {val_logloss:.4f}")
                
                results[name] = {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_logloss': train_logloss,
                    'val_logloss': val_logloss
                }
        
        return results
    
    def predict_match(self, player_a, player_b, surface='hard', 
                     recent_matches_df=None, h2h_df=None):
        """Predict probability that player_a beats player_b"""
        features = create_match_features(
            player_a, player_b, surface, self.elo_system,
            recent_matches_df, h2h_df
        )
        
        X = np.array([list(features.values())])
        
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict_proba(X)[0][1])
        
        ensemble_prob = np.mean(predictions)
        
        return {
            'probability_a_wins': ensemble_prob,
            'probability_b_wins': 1 - ensemble_prob,
            'individual_predictions': dict(zip(self.models.keys(), predictions))
        }
    
    def simulate_tournament(self, players, surface='hard', n_simulations=10000, verbose=True):
        """Monte Carlo simulation of single-elimination tournament"""
        if len(players) & (len(players) - 1) != 0:
            raise ValueError("Number of players must be a power of 2")
        
        winner_counts = {player: 0 for player in players}
        finalist_counts = {player: 0 for player in players}
        semifinalist_counts = {player: 0 for player in players}
        
        if verbose:
            print(f"Running {n_simulations} tournament simulations...")
        
        for sim in range(n_simulations):
            if verbose and sim % 1000 == 0 and sim > 0:
                print(f"  Completed {sim}/{n_simulations}")
            
            current_round = players.copy()
            
            while len(current_round) > 1:
                next_round = []
                
                for i in range(0, len(current_round), 2):
                    player_a = current_round[i]
                    player_b = current_round[i + 1]
                    
                    prob_a = self.predict_match(player_a, player_b, surface)['probability_a_wins']
                    winner = player_a if np.random.random() < prob_a else player_b
                    next_round.append(winner)
                
                if len(current_round) == 4:
                    for player in current_round:
                        semifinalist_counts[player] += 1
                
                if len(current_round) == 2:
                    for player in current_round:
                        finalist_counts[player] += 1
                
                current_round = next_round
            
            winner_counts[current_round[0]] += 1
        
        winner_probs = {k: v/n_simulations for k, v in winner_counts.items()}
        finalist_probs = {k: v/n_simulations for k, v in finalist_counts.items()}
        semifinalist_probs = {k: v/n_simulations for k, v in semifinalist_counts.items()}
        
        results_df = pd.DataFrame({
            'Player': players,
            'Win_Probability': [winner_probs[p] for p in players],
            'Final_Probability': [finalist_probs[p] for p in players],
            'Semi_Probability': [semifinalist_probs[p] for p in players]
        }).sort_values('Win_Probability', ascending=False)
        
        return results_df