def create_match_features(player_a, player_b, surface, elo_system, 
                          recent_matches_df=None, h2h_df=None):
    """
    Create feature vector for a potential matchup
    """
    features = {}
    
    # 1. ELO FEATURES (Always available)
    features['elo_diff'] = (elo_system.get_rating(player_a, surface) - 
                           elo_system.get_rating(player_b, surface))
    features['elo_overall_diff'] = (elo_system.get_rating(player_a, 'overall') - 
                                   elo_system.get_rating(player_b, 'overall'))
    features['elo_a_surface'] = elo_system.get_rating(player_a, surface)
    features['elo_b_surface'] = elo_system.get_rating(player_b, surface)
    features['elo_a_overall'] = elo_system.get_rating(player_a, 'overall')
    features['elo_b_overall'] = elo_system.get_rating(player_b, 'overall')
    
    # 2. RECENT FORM (if data available)
    if recent_matches_df is not None and len(recent_matches_df) > 0:
        player_a_recent = recent_matches_df[
            (recent_matches_df['player'] == player_a) & 
            (recent_matches_df['surface'] == surface)
        ].tail(10)
        
        player_b_recent = recent_matches_df[
            (recent_matches_df['player'] == player_b) & 
            (recent_matches_df['surface'] == surface)
        ].tail(10)
        
        features['win_pct_a_recent'] = player_a_recent['won'].mean() if len(player_a_recent) > 0 else 0.5
        features['win_pct_b_recent'] = player_b_recent['won'].mean() if len(player_b_recent) > 0 else 0.5
        features['form_diff'] = features['win_pct_a_recent'] - features['win_pct_b_recent']
    else:
        features['win_pct_a_recent'] = 0.5
        features['win_pct_b_recent'] = 0.5
        features['form_diff'] = 0.0
    
    # 3. HEAD-TO-HEAD (if data available)
    if h2h_df is not None and len(h2h_df) > 0:
        h2h = h2h_df[
            ((h2h_df['winner'] == player_a) & (h2h_df['loser'] == player_b)) |
            ((h2h_df['winner'] == player_b) & (h2h_df['loser'] == player_a))
        ]
        
        a_wins = len(h2h[h2h['winner'] == player_a])
        b_wins = len(h2h[h2h['winner'] == player_b])
        
        features['h2h_a_wins'] = a_wins
        features['h2h_b_wins'] = b_wins
        features['h2h_diff'] = a_wins - b_wins
        
        # H2H on surface
        h2h_surface = h2h[h2h['surface'] == surface]
        features['h2h_surface_diff'] = (
            len(h2h_surface[h2h_surface['winner'] == player_a]) -
            len(h2h_surface[h2h_surface['winner'] == player_b])
        )
    else:
        features['h2h_a_wins'] = 0
        features['h2h_b_wins'] = 0
        features['h2h_diff'] = 0
        features['h2h_surface_diff'] = 0
    
    return features