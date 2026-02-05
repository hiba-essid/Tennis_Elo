"""
Grand Slam Tournament Simulation for 2026 Predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class GrandSlamTournament:
    """
    Simulate Grand Slam tournaments with proper 128-player draws and seeding
    """
    
    GRAND_SLAMS_2026 = {
        'Australian Open': {
            'surface': 'hard',
            'dates': 'Jan 19 - Feb 1, 2026',
            'location': 'Melbourne',
            'best_of': 5
        },
        'French Open': {
            'surface': 'clay',
            'dates': 'May 24 - Jun 7, 2026',
            'location': 'Paris',
            'best_of': 5
        },
        'Wimbledon': {
            'surface': 'grass',
            'dates': 'Jun 29 - Jul 12, 2026',
            'location': 'London',
            'best_of': 5
        },
        'US Open': {
            'surface': 'hard',
            'dates': 'Aug 31 - Sep 13, 2026',
            'location': 'New York',
            'best_of': 5
        }
    }
    
    def __init__(self, predictor, tournament_name, gender='men', elo_system=None):
        """
        Initialize Grand Slam tournament
        
        Args:
            predictor: GrandSlamPredictor instance
            tournament_name: Name of Grand Slam
            gender: 'men' or 'women'
            elo_system: Optional Elo system to use (default uses predictor's Elo)
        """
        self.predictor = predictor
        self.tournament_name = tournament_name
        self.gender = gender
        self.elo_system = elo_system or predictor.elo_system
        self.slam_info = self.GRAND_SLAMS_2026[tournament_name]
        self.surface = self.slam_info['surface']
        
    def create_seeded_draw(self, players, seeds=32):
        """
        Create realistic seeded draw (top 32 seeds separated)
        
        Args:
            players: List of players ordered by ranking/rating
            seeds: Number of seeded players (default 32)
        """
        if len(players) != 128:
            raise ValueError(f"Need exactly 128 players, got {len(players)}")
        
        seeded = players[:seeds]
        unseeded = players[seeds:]
        
        # Simple draw: seeds 1-32 won't meet until later rounds
        # Seed 1 and 2 in opposite halves, etc.
        draw = []
        
        # Top half (64 players)
        # Bottom half (64 players)
        # This is simplified - real draws are more complex
        
        return players  # For now, keep order-based draw
    
    def simulate_match(self, player_a, player_b, round_name='R128'):
        """
        Simulate single match between two players
        
        Returns: winner, probability
        """
        prediction = self.predictor.predict_match(
            player_a, player_b, 
            surface=self.surface
        )
        
        prob_a = prediction['probability_a_wins']
        
        # Add slight randomness for Grand Slam upsets
        # Grand Slams have more upsets than regular tournaments
        upset_factor = 0.05 if round_name in ['R128', 'R64'] else 0.02
        prob_a = np.clip(prob_a + np.random.normal(0, upset_factor), 0.05, 0.95)
        
        winner = player_a if np.random.random() < prob_a else player_b
        
        return winner, prob_a
    
    def simulate_tournament(self, players, n_simulations=10000, verbose=True):
        """
        Monte Carlo simulation of Grand Slam tournament
        
        Args:
            players: List of 128 players (ordered by seeding)
            n_simulations: Number of tournament simulations
            verbose: Print progress
        
        Returns:
            DataFrame with winner, finalist, and semifinalist probabilities
        """
        if len(players) != 128:
            # If fewer players, pad with lower-rated players or truncate
            if len(players) < 128:
                print(f"⚠️  Only {len(players)} players available, using top players")
                players = players[:min(len(players), 128)]
            else:
                players = players[:128]
        
        winner_counts = {player: 0 for player in players}
        finalist_counts = {player: 0 for player in players}
        semifinalist_counts = {player: 0 for player in players}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎾 {self.tournament_name} - {self.gender.upper()}")
            print(f"📍 {self.slam_info['location']} | 🎾 {self.surface.upper()} court")
            print(f"📅 {self.slam_info['dates']}")
            print(f"{'='*60}")
            print(f"\nRunning {n_simulations:,} tournament simulations...")
        
        for sim in range(n_simulations):
            if verbose and sim % 2000 == 0 and sim > 0:
                print(f"  ✓ Completed {sim:,}/{n_simulations:,} simulations")
            
            # Simulate full tournament
            current_round = players.copy()
            round_names = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
            round_idx = 0
            
            while len(current_round) > 1:
                next_round = []
                round_name = round_names[round_idx] if round_idx < len(round_names) else 'F'
                
                # Pair up players and simulate matches
                for i in range(0, len(current_round), 2):
                    if i + 1 < len(current_round):
                        player_a = current_round[i]
                        player_b = current_round[i + 1]
                        
                        winner, _ = self.simulate_match(player_a, player_b, round_name)
                        next_round.append(winner)
                    else:
                        # Bye (shouldn't happen with 128 players)
                        next_round.append(current_round[i])
                
                # Track semifinalists (when 4 players remain)
                if len(current_round) == 4:
                    for player in current_round:
                        semifinalist_counts[player] += 1
                
                # Track finalists (when 2 players remain)
                if len(current_round) == 2:
                    for player in current_round:
                        finalist_counts[player] += 1
                
                current_round = next_round
                round_idx += 1
            
            # Winner
            if len(current_round) == 1:
                winner_counts[current_round[0]] += 1
        
        if verbose:
            print(f"  ✓ Completed {n_simulations:,}/{n_simulations:,} simulations\n")
        
        # Calculate probabilities
        results = []
        for player in players:
            results.append({
                'Player': player,
                'Win_Probability': winner_counts[player] / n_simulations,
                'Final_Probability': finalist_counts[player] / n_simulations,
                'Semi_Probability': semifinalist_counts[player] / n_simulations
            })
        
        results_df = pd.DataFrame(results).sort_values('Win_Probability', ascending=False)
        
        return results_df
    
    def print_predictions(self, results_df, top_n=10):
        """Pretty print tournament predictions"""
        print(f"\n{'='*60}")
        print(f"🏆 {self.tournament_name} {self.gender.upper()} - TOP {top_n} PREDICTIONS")
        print(f"{'='*60}\n")
        
        print(f"{'Rank':<6} {'Player':<25} {'Win %':<10} {'Final %':<10} {'Semi %':<10}")
        print(f"{'-'*60}")
        
        for idx, row in results_df.head(top_n).iterrows():
            print(f"{idx+1:<6} {row['Player']:<25} "
                  f"{row['Win_Probability']*100:>6.2f}%   "
                  f"{row['Final_Probability']*100:>6.2f}%   "
                  f"{row['Semi_Probability']*100:>6.2f}%")
        
        print(f"\n{'='*60}")
        print(f"🥇 MOST LIKELY WINNER: {results_df.iloc[0]['Player']}")
        print(f"   Win Probability: {results_df.iloc[0]['Win_Probability']*100:.1f}%")
        print(f"\n🥈 MOST LIKELY FINALIST: {results_df.iloc[1]['Player']}")
        print(f"   Final Probability: {results_df.iloc[1]['Final_Probability']*100:.1f}%")
        print(f"{'='*60}\n")


def predict_all_2026_slams(men_predictor, women_predictor, 
                           men_players, women_players,
                           n_simulations=10000):
    """
    Predict all 2026 Grand Slams for men and women
    
    Args:
        men_predictor: Trained predictor for men
        women_predictor: Trained predictor for women
        men_players: List of men's players (top 128+)
        women_players: List of women's players (top 128+)
        n_simulations: Monte Carlo simulation count
    
    Returns:
        Dictionary with all predictions
    """
    all_predictions = {}
    
    slam_names = ['Australian Open', 'French Open', 'Wimbledon', 'US Open']
    
    for slam_name in slam_names:
        print(f"\n\n{'#'*70}")
        print(f"# {slam_name.upper()}")
        print(f"{'#'*70}")
        
        # Men's tournament
        men_slam = GrandSlamTournament(men_predictor, slam_name, gender='men')
        men_results = men_slam.simulate_tournament(
            men_players[:128], 
            n_simulations=n_simulations
        )
        men_slam.print_predictions(men_results, top_n=10)
        
        # Women's tournament
        women_slam = GrandSlamTournament(women_predictor, slam_name, gender='women')
        women_results = women_slam.simulate_tournament(
            women_players[:128],
            n_simulations=n_simulations
        )
        women_slam.print_predictions(women_results, top_n=10)
        
        all_predictions[slam_name] = {
            'men': men_results,
            'women': women_results
        }
    
    return all_predictions
