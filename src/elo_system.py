import pandas as pd
import numpy as np

class TennisEloSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        
        # Separate Elo for each surface + overall
        self.ratings = {
            'overall': {},
            'hard': {},
            'clay': {},
            'grass': {},
            'carpet': {}
        }
    
    def get_rating(self, player, surface='overall'):
        """Get player's rating for a surface, initialize if new"""
        if player not in self.ratings[surface]:
            self.ratings[surface][player] = self.initial_rating
        return self.ratings[surface][player]
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner, loser, surface='overall', 
                       margin_multiplier=1.0, importance_multiplier=1.0):
        """
        Update ratings after a match
        
        margin_multiplier: Adjust K based on match dominance (e.g., straight sets)
        importance_multiplier: Weight GS matches higher than regular tours
        """
        # Get current ratings
        rating_winner = self.get_rating(winner, surface)
        rating_loser = self.get_rating(loser, surface)
        
        # Calculate expected scores
        exp_winner = self.expected_score(rating_winner, rating_loser)
        exp_loser = self.expected_score(rating_loser, rating_winner)
        
        # Dynamic K-factor
        k = self.k_factor * margin_multiplier * importance_multiplier
        
        # Update ratings
        new_rating_winner = rating_winner + k * (1 - exp_winner)
        new_rating_loser = rating_loser + k * (0 - exp_loser)
        
        self.ratings[surface][winner] = new_rating_winner
        self.ratings[surface][loser] = new_rating_loser
        
        return {
            'winner_old': rating_winner,
            'winner_new': new_rating_winner,
            'loser_old': rating_loser,
            'loser_new': new_rating_loser,
            'exp_winner': exp_winner
        }
    
    def process_matches(self, matches_df):
        """
        Process historical matches to build Elo ratings
        
        matches_df should have: date, winner, loser, surface, 
                                tournament_level, best_of, score
        """
        matches_df = matches_df.sort_values('date').reset_index(drop=True)
        
        results = []
        
        for idx, match in matches_df.iterrows():
            # Calculate multipliers
            margin_mult = self._calculate_margin_multiplier(
                match['score'], match['best_of']
            )
            importance_mult = self._calculate_importance_multiplier(
                match['tournament_level']
            )
            
            # Update overall Elo
            result_overall = self.update_ratings(
                match['winner'], 
                match['loser'], 
                surface='overall',
                margin_multiplier=margin_mult,
                importance_multiplier=importance_mult
            )
            
            # Update surface-specific Elo
            result_surface = self.update_ratings(
                match['winner'], 
                match['loser'], 
                surface=match['surface'].lower(),
                margin_multiplier=margin_mult,
                importance_multiplier=importance_mult
            )
            
            results.append({
                'match_id': idx,
                'date': match['date'],
                'winner': match['winner'],
                'loser': match['loser'],
                'surface': match['surface'],
                **result_surface
            })
        
        return pd.DataFrame(results)
    
    def _calculate_margin_multiplier(self, score, best_of):
        """Adjust K based on match dominance"""
        # Parse score to determine sets won
        # Example: "6-4 6-3" vs "7-6 3-6 7-6 6-7 10-8"
        # Straight sets win in best of 5 = higher multiplier
        
        if pd.isna(score):
            return 1.0
        
        try:
            sets = score.split()
            winner_sets = sum(1 for s in sets if s.split('-')[0] > s.split('-')[1])
            loser_sets = len(sets) - winner_sets
            
            if best_of == 5:
                if winner_sets == 3 and loser_sets == 0:
                    return 1.2  # Dominant straight sets
                elif winner_sets == 3 and loser_sets == 2:
                    return 0.9  # Close 5-setter
            elif best_of == 3:
                if winner_sets == 2 and loser_sets == 0:
                    return 1.1  # Straight sets
        except:
            pass
        
        return 1.0
    
    def _calculate_importance_multiplier(self, tournament_level):
        """Weight Grand Slams higher than ATP 250s"""
        weights = {
            'Grand Slam': 1.5,
            'Masters 1000': 1.2,
            'ATP 500': 1.0,
            'ATP 250': 0.8,
            'Challenger': 0.6
        }
        return weights.get(tournament_level, 1.0)
    
    def get_match_probability(self, player_a, player_b, surface='overall'):
        """Predict probability that player_a beats player_b"""
        rating_a = self.get_rating(player_a, surface)
        rating_b = self.get_rating(player_b, surface)
        return self.expected_score(rating_a, rating_b)
    
    def get_top_players(self, surface='overall', top_n=128):
        """Get top N players by rating for a surface"""
        if surface not in self.ratings:
            surface = 'overall'
        
        players_ratings = [(p, r) for p, r in self.ratings[surface].items()]
        players_ratings.sort(key=lambda x: x[1], reverse=True)
        return players_ratings[:top_n]
    
    def get_all_players(self, surface='overall'):
        """Get all players with ratings on a surface"""
        if surface not in self.ratings:
            surface = 'overall'
        return list(self.ratings[surface].keys())