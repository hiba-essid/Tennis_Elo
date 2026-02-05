#!/usr/bin/env python
"""Flask web interface for Tennis Elo Predictions"""

from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os
from pathlib import Path

app = Flask(__name__)

# Load predictions from CSV files
def load_predictions():
    tournaments = {
        'australian_open': 'Australian Open (Hard - Melbourne)',
        'french_open': 'French Open (Clay - Paris)',
        'wimbledon': 'Wimbledon (Grass - London)',
        'us_open': 'US Open (Hard - New York)'
    }
    
    predictions = {}
    for key, name in tournaments.items():
        file = f'data/processed/predictions_{key}_2026.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            predictions[key] = {
                'name': name,
                'data': df.to_dict('records')
            }
    
    return predictions

def load_training_history():
    """Load training history from JSON"""
    history_file = 'models/training_history/training_history.json'
    
    if Path(history_file).exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    """Homepage with predictions summary"""
    predictions = load_predictions()
    
    summary = {}
    for key, data in predictions.items():
        if 'data' in data and len(data['data']) > 0:
            top_player = data['data'][0]
            summary[key] = {
                'tournament': data['name'],
                'winner': top_player.get('Player', 'Unknown'),
                'probability': f"{top_player.get('Win_Probability', 0)*100:.0f}%",
                'finalist': data['data'][1].get('Player', 'Unknown') if len(data['data']) > 1 else 'TBD'
            }
    
    return render_template('index.html', summary=summary)

@app.route('/tournament/<tournament>')
def tournament_detail(tournament):
    """Detailed predictions for a specific tournament"""
    predictions = load_predictions()
    
    if tournament not in predictions:
        return jsonify({'error': 'Tournament not found'}), 404
    
    data = predictions[tournament]
    return render_template('tournament.html', 
                         tournament_name=data['name'],
                         players=data['data'])

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions data"""
    predictions = load_predictions()
    return jsonify(predictions)

@app.route('/model-performance')
def model_performance():
    """Model performance metrics and training curves"""
    training_history = load_training_history()
    
    if not training_history:
        return jsonify({'error': 'Training history not found'}), 404
    
    return render_template('model_performance.html', 
                         training_data=json.dumps(training_history))

@app.route('/api/training-history')
def api_training_history():
    """API endpoint for training history"""
    training_history = load_training_history()
    
    if not training_history:
        return jsonify({'error': 'Training history not found'}), 404
    
    return jsonify(training_history)

if __name__ == '__main__':
    print("=" * 70)
    print("🎾 TENNIS ELO PREDICTIONS - WEB INTERFACE")
    print("=" * 70)
    print("\n📊 Predictions loaded successfully!")
    print("\n🌐 Open your browser to: http://localhost:5000/")
    print("\nRoutes:")
    print("  • http://localhost:5000/")
    print("  • http://localhost:5000/tournament/australian_open")
    print("  • http://localhost:5000/tournament/french_open")
    print("  • http://localhost:5000/tournament/wimbledon")
    print("  • http://localhost:5000/tournament/us_open")
    print("  • http://localhost:5000/model-performance")
    print("  • http://localhost:5000/api/predictions")
    print("  • http://localhost:5000/api/training-history")
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)
