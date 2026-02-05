"""
Utility functions
"""

import pandas as pd
import numpy as np
import os


def create_sample_data(n_matches=5000):
    """Create synthetic tennis match data for testing"""
    np.random.seed(42)
    
    players = [f"Player_{i}" for i in range(50)]
    surfaces = ['hard', 'clay', 'grass']
    tournament_levels = ['Grand Slam', 'Masters 1000', 'ATP 500', 'ATP 250']
    
    data = []
    start_date = pd.to_datetime('2020-01-01')
    
    for i in range(n_matches):
        winner = np.random.choice(players)
        loser = np.random.choice([p for p in players if p != winner])
        
        data.append({
            'date': start_date + pd.Timedelta(days=i % 1000),
            'winner': winner,
            'loser': loser,
            'surface': np.random.choice(surfaces),
            'tournament_level': np.random.choice(tournament_levels, p=[0.1, 0.2, 0.3, 0.4]),
            'best_of': np.random.choice([3, 5], p=[0.7, 0.3]),
            'score': '6-4 6-3'
        })
    
    return pd.DataFrame(data)


def load_atp_data(filepath):
    """
    Load ATP match data from CSV
    
    Expected columns: date, winner_name, loser_name, surface, 
                     tourney_level, best_of, score
    
    Compatible with Jeff Sackmann's tennis_atp repository format:
    https://github.com/JeffSackmann/tennis_atp
    """
    try:
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'tourney_date': 'date',
            'winner_name': 'winner',
            'loser_name': 'loser',
            'tourney_level': 'tournament_level'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['date', 'winner', 'loser', 'surface']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        
        # Add tournament_level if missing
        if 'tournament_level' not in df.columns:
            df['tournament_level'] = 'ATP 250'
        
        # Map tournament levels
        level_mapping = {
            'G': 'Grand Slam',
            'M': 'Masters 1000',
            'A': 'ATP 500',
            'D': 'ATP 500',
            'F': 'ATP Finals',
            'C': 'Challenger'
        }
        df['tournament_level'] = df['tournament_level'].map(level_mapping).fillna('ATP 250')
        
        # Add best_of if missing
        if 'best_of' not in df.columns:
            df['best_of'] = 3
            # Grand Slams are best of 5
            df.loc[df['tournament_level'] == 'Grand Slam', 'best_of'] = 5
        
        # Clean surface names
        if 'surface' in df.columns:
            df['surface'] = df['surface'].str.lower()
            df['surface'] = df['surface'].fillna('hard')
        
        # Add score if missing
        if 'score' not in df.columns:
            df['score'] = '6-4 6-3'
        
        print(f"✅ Loaded {len(df)} matches from {filepath}")
        return df
        
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        print("💡 Download ATP data from: https://github.com/JeffSackmann/tennis_atp")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def load_wta_data(filepath):
    """
    Load WTA match data from CSV
    
    Compatible with Jeff Sackmann's tennis_wta repository format:
    https://github.com/JeffSackmann/tennis_wta
    """
    # WTA data has same format as ATP
    return load_atp_data(filepath)


def load_kaggle_mens_grand_slam_data(filepath):
    """
    Load ATP player statistics data from CSV
    
    Format: Player name, Rank, Matches, Wins, Losses, Win Percentage
    This generates synthetic match data based on player statistics.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Check available columns
        print(f"Loading data - columns: {df.columns.tolist()}")
        
        # Try to load from Grandslam.csv first (real tournament data)
        grandslam_path = 'data/raw/Grandslam.csv'
        if os.path.exists(grandslam_path):
            print("Found Grandslam.csv - loading real tournament data...")
            try:
                gs_df = pd.read_csv(grandslam_path)
                if len(gs_df) > 0:
                    result = _load_tournament_format(gs_df)
                    if result is not None and len(result) > 0:
                        print(f"Loaded {len(result)} real Grand Slam matches from Grandslam.csv")
                        return result
            except Exception as e:
                print(f"Could not load Grandslam.csv: {e}")
        
        # Handle player stats format (Rank, Player, Matches, Wins, Losses, PCT)
        if 'Player' in df.columns and 'Matches' in df.columns:
            print("Detected player statistics format - creating synthetic matches")
            return _create_matches_from_player_stats(df)
        
        # Handle original tournament format
        elif 'YEAR' in df.columns or 'TOURNAMENT' in df.columns:
            print("Detected tournament format")
            # Map column names to expected format
            column_mapping = {
                'YEAR': 'year',
                'TOURNAMENT': 'tournament',
                'WINNER': 'winner',
                'RUNNER-UP': 'runner_up',
                'TOURNAMENT_SURFACE': 'surface'
            }
            df = df.rename(columns=column_mapping)
        
        else:
            raise ValueError(f"Unknown data format")
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_matches_from_player_stats(df):
    """
    Create synthetic match data from player statistics.
    Uses win/loss records to generate probabilistic matches for Elo calibration.
    """
    try:
        # Ensure we have required columns
        required_cols = ['Player', 'Wins', 'Losses']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Extract player info
        players_data = []
        for _, row in df.iterrows():
            player_name = str(row['Player']).strip()
            # Remove country codes like (ESP)
            if '(' in player_name:
                player_name = player_name.split('(')[0].strip()
            
            wins = int(row['Wins']) if pd.notna(row['Wins']) else 0
            losses = int(row['Losses']) if pd.notna(row['Losses']) else 0
            total_matches = wins + losses
            
            if total_matches > 0:
                players_data.append({
                    'player': player_name,
                    'wins': wins,
                    'losses': losses,
                    'total': total_matches,
                    'win_pct': wins / total_matches
                })
        
        # Create synthetic matches based on player quality
        matches = []
        num_players = len(players_data)
        np.random.seed(42)
        
        # Generate matches between players based on their win percentages
        for i in range(num_players):
            for j in range(i + 1, min(i + 5, num_players)):
                player_a = players_data[i]
                player_b = players_data[j]
                
                # Create multiple matches between pairs
                num_meetings = 3
                for meeting in range(num_meetings):
                    # Determine winner based on win percentage
                    prob_a_wins = player_a['win_pct']
                    
                    if np.random.random() < prob_a_wins:
                        winner = player_a['player']
                        loser = player_b['player']
                    else:
                        winner = player_b['player']
                        loser = player_a['player']
                    
                    surface = np.random.choice(['hard', 'clay', 'grass'])
                    
                    # Create date spread
                    year = 2020 + (i + j * num_players + meeting) % 6
                    month = 1 + (i + j) % 12
                    day = 1 + (i * j + meeting) % 28
                    
                    try:
                        match_date = pd.Timestamp(year=year, month=month, day=day)
                    except:
                        match_date = pd.Timestamp(f'{year}-01-01')
                    
                    matches.append({
                        'date': match_date,
                        'winner': winner,
                        'loser': loser,
                        'surface': surface,
                        'tournament_level': np.random.choice(['Masters 1000', 'ATP 500', 'ATP 250']),
                        'best_of': 3,
                        'score': 'Match'
                    })
        
        result_df = pd.DataFrame(matches)
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        print(f"Created {len(result_df)} synthetic matches from {len(players_data)} players")
        print(f"Date range: {result_df['date'].min().date()} to {result_df['date'].max().date()}")
        
        return result_df
        
    except Exception as e:
        print(f"Error creating matches: {e}")
        return None


def _load_tournament_format(df):
    """
    Load tournament-based Grand Slam data (original format).
    Format: YEAR, TOURNAMENT, WINNER, RUNNER-UP
    """
    try:
        # Map tournament names and surfaces
        tournament_surface_mapping = {
            'Australian Open': 'hard',
            'Austalian Open': 'hard',  # Handle typo in data
            'French Open': 'clay',
            'Wimbledon': 'grass',
            'U.S. Open': 'hard',
            'US Open': 'hard',
        }
        
        def get_surface(tournament_name):
            for key, surface in tournament_surface_mapping.items():
                if key.lower() in tournament_name.lower():
                    return surface
            return 'hard'  # Default
        
        def normalize_tournament_name(name):
            for key in tournament_surface_mapping.keys():
                if key.lower() in name.lower():
                    return key
            return name
        
        matches = []
        for _, row in df.iterrows():
            try:
                year = int(row.get('Year', 2023)) if pd.notna(row.get('Year')) else 2023
                tournament = str(row.get('Tournament', '')).strip()
                winner = str(row.get('Winner', '')).strip()
                loser = str(row.get('Runner-Up', '')).strip()
                
                if not winner or not loser or not tournament:
                    continue
                
                # Normalize tournament name
                tournament_clean = normalize_tournament_name(tournament)
                
                # Get surface
                surface = get_surface(tournament)
                
                # Create date based on tournament
                if 'Australian' in tournament:
                    date = pd.Timestamp(f'{year}-01-28')
                elif 'French' in tournament:
                    date = pd.Timestamp(f'{year}-06-09')
                elif 'Wimbledon' in tournament:
                    date = pd.Timestamp(f'{year}-07-14')
                elif 'U.S.' in tournament or 'US' in tournament:
                    date = pd.Timestamp(f'{year}-09-08')
                else:
                    date = pd.Timestamp(f'{year}-06-15')
                
                matches.append({
                    'date': date,
                    'winner': winner,
                    'loser': loser,
                    'surface': surface,
                    'tournament_level': 'Grand Slam',
                    'best_of': 5,
                    'tournament': tournament_clean,
                    'score': 'Final'
                })
            except Exception as e:
                print(f"Skipping row: {e}")
                continue
        
        result_df = pd.DataFrame(matches)
        if len(result_df) > 0:
            result_df = result_df.sort_values('date').reset_index(drop=True)
            return result_df
        else:
            print("No valid matches found in tournament data")
            return None
        
    except Exception as e:
        print(f"Error loading tournament format: {e}")
        return None


def save_model(model, filepath):
    """Save model to disk"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from disk"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)