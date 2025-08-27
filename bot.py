import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedUK49sPredictor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = self.load_data()
        
    def load_data(self):
        """Load and preprocess data"""
        try:
            data = pd.read_csv(self.csv_file)
            data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_draw_data(self, draw_type):
        """Get data for specific draw type"""
        return self.data[self.data['draw'] == draw_type].copy()
    
    def frequency_analysis(self, draw_type, lookback_days=60):
        """Advanced frequency analysis with recency weighting"""
        draw_data = self.get_draw_data(draw_type)
        if draw_data.empty:
            return []
        
        # Weight recent draws more heavily
        recent_data = draw_data.tail(lookback_days)
        weights = np.linspace(0.1, 1.0, len(recent_data))  # Linear weights
        
        number_scores = {}
        for idx, (_, row) in enumerate(recent_data.iterrows()):
            weight = weights[idx]
            for i in range(1, 7):
                num = row[f'n{i}']
                if num in number_scores:
                    number_scores[num] += weight
                else:
                    number_scores[num] = weight
        
        # Get top 12 numbers for combination
        top_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)[:12]
        return [num for num, score in top_numbers]
    
    def pattern_analysis(self, draw_type):
        """Analyze number patterns and sequences"""
        draw_data = self.get_draw_data(draw_type)
        if len(draw_data) < 10:
            return []
        
        # Analyze common number pairs and triplets
        all_pairs = []
        all_triplets = []
        
        for _, row in draw_data.iterrows():
            numbers = sorted([row[f'n{i}'] for i in range(1, 7)])
            
            # Get all pairs
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    all_pairs.append((numbers[i], numbers[j]))
            
            # Get all triplets
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    for k in range(j+1, len(numbers)):
                        all_triplets.append((numbers[i], numbers[j], numbers[k]))
        
        # Find most common patterns
        pair_freq = Counter(all_pairs).most_common(10)
        triplet_freq = Counter(all_triplets).most_common(8)
        
        # Extract unique numbers from top patterns
        pattern_numbers = set()
        for pair, count in pair_freq:
            pattern_numbers.update(pair)
        for triplet, count in triplet_freq:
            pattern_numbers.update(triplet)
        
        return sorted(pattern_numbers)
    
    def hot_cold_analysis(self, draw_type, window_size=20):
        """Identify hot and cold numbers"""
        draw_data = self.get_draw_data(draw_type)
        if len(draw_data) < window_size:
            return [], []
        
        recent_draws = draw_data.tail(window_size)
        all_numbers = set(range(1, 50))
        
        # Hot numbers (appeared recently)
        hot_numbers = set()
        for _, row in recent_draws.iterrows():
            for i in range(1, 7):
                hot_numbers.add(row[f'n{i}'])
        
        # Cold numbers (not appeared recently)
        cold_numbers = all_numbers - hot_numbers
        
        return sorted(hot_numbers), sorted(cold_numbers)
    
    def ml_prediction(self, draw_type):
        """Machine learning prediction using linear regression"""
        draw_data = self.get_draw_data(draw_type)
        if len(draw_data) < 30:
            return []
        
        # Prepare features (last 10 draws)
        features = []
        targets = []
        
        for i in range(10, len(draw_data)):
            # Use previous 10 draws as features
            prev_draws = []
            for j in range(1, 11):
                if i - j >= 0:
                    row = draw_data.iloc[i - j]
                    prev_draws.extend([row[f'n{k}'] for k in range(1, 7)])
                else:
                    prev_draws.extend([0] * 6)  # Padding
            
            # Current draw as target
            current_row = draw_data.iloc[i]
            target = [current_row[f'n{k}'] for k in range(1, 7)]
            
            features.append(prev_draws)
            targets.append(target)
        
        if not features:
            return []
        
        # Train model for each number position
        predictions = []
        for pos in range(6):
            try:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                y_pos = [target[pos] for target in targets]
                model.fit(features, y_pos)
                
                # Predict using most recent data
                latest_features = features[-1]
                pred = model.predict([latest_features])[0]
                predictions.append(max(1, min(49, round(pred))))
            except:
                predictions.append(np.random.randint(1, 50))
        
        return sorted(predictions)
    
    def generate_combinations(self, numbers, count=6):
        """Generate smart combinations from candidate numbers"""
        if len(numbers) < count:
            numbers.extend(list(range(1, 50)))
            numbers = list(set(numbers))[:20]
        
        # Ensure good number distribution
        combinations = []
        base_numbers = numbers[:count]
        
        # Create variations
        for i in range(5):
            combo = base_numbers.copy()
            # Replace 1-2 numbers with alternatives
            replace_count = min(2, len(numbers) - count)
            for j in range(replace_count):
                if j < len(numbers) - count:
                    combo[np.random.randint(0, count)] = numbers[count + j]
            combinations.append(sorted(combo))
        
        return combinations[:3]  # Return top 3 combinations
    
    def predict_draw(self, draw_type):
        """Main prediction function"""
        print(f"\nAnalyzing {draw_type}...")
        
        # Multiple prediction strategies
        freq_numbers = self.frequency_analysis(draw_type, 90)
        pattern_numbers = self.pattern_analysis(draw_type)
        hot_numbers, cold_numbers = self.hot_cold_analysis(draw_type)
        ml_pred = self.ml_prediction(draw_type)
        
        # Combine all candidate numbers
        all_candidates = list(set(freq_numbers + pattern_numbers + hot_numbers + cold_numbers + ml_pred))
        
        if len(all_candidates) < 6:
            all_candidates.extend(list(range(1, 50)))
            all_candidates = list(set(all_candidates))
        
        # Generate smart combinations
        predictions = self.generate_combinations(all_candidates, 6)
        
        return predictions
    
    def analyze_trends(self, draw_type):
        """Comprehensive trend analysis"""
        draw_data = self.get_draw_data(draw_type)
        if draw_data.empty:
            return {}
        
        analysis = {
            'total_draws': len(draw_data),
            'date_range': f"{draw_data['date'].min().date()} to {draw_data['date'].max().date()}",
            'most_common': [],
            'recent_trends': [],
            'number_stats': {}
        }
        
        # Most common numbers
        all_numbers = []
        for _, row in draw_data.iterrows():
            all_numbers.extend([row[f'n{i}'] for i in range(1, 7)])
        
        analysis['most_common'] = Counter(all_numbers).most_common(15)
        
        # Recent trends (last 10 draws)
        recent = draw_data.tail(10)
        analysis['recent_trends'] = [
            [row[f'n{i}'] for i in range(1, 7)] for _, row in recent.iterrows()
        ]
        
        return analysis

def main():
    print("🤖 Advanced UK49s Prediction Bot")
    print("=================================")
    
    predictor = AdvancedUK49sPredictor('uk49s_results.csv')
    
    if predictor.data.empty:
        print("❌ Could not load data. Check your CSV file.")
        return
    
    results = {}
    
    for draw_type in ['Lunchtime', 'Teatime']:
        print(f"\n🔍 Analyzing {draw_type}...")
        
        # Get predictions
        predictions = predictor.predict_draw(draw_type)
        results[draw_type] = {
            'predictions': predictions,
            'primary_prediction': predictions[0] if predictions else [],
            'analysis': predictor.analyze_trends(draw_type)
        }
        
        print(f"   Top prediction: {predictions[0] if predictions else 'N/A'}")
        print(f"   Alternative: {predictions[1] if len(predictions) > 1 else 'N/A'}")
    
    # Save results to file
    save_results(results)
    print(f"\n✅ Predictions saved to PREDICTIONS.txt")

def save_results(results):
    """Save comprehensive results"""
    with open('PREDICTIONS.txt', 'w') as f:
        f.write("🎯 ADVANCED UK49s PREDICTIONS\n")
        f.write("=============================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for draw_type, data in results.items():
            f.write(f"⭐ {draw_type.upper()} PREDICTION\n")
            f.write("=" * 40 + "\n")
            
            if data['predictions']:
                f.write(f"🎯 Primary Prediction: {data['predictions'][0]}\n")
                f.write(f"   Bet Format: {'-'.join(map(str, data['predictions'][0]))}\n\n")
                
                f.write("🔄 Alternative Predictions:\n")
                for i, pred in enumerate(data['predictions'][1:], 2):
                    f.write(f"   Option {i}: {pred}\n")
                f.write("\n")
            
            # Add analysis
            if data['analysis']:
                f.write("📊 TREND ANALYSIS:\n")
                f.write(f"   Total draws analyzed: {data['analysis']['total_draws']}\n")
                f.write(f"   Date range: {data['analysis']['date_range']}\n\n")
                
                f.write("🔥 Most common numbers:\n")
                for num, count in data['analysis']['most_common'][:10]:
                    f.write(f"   {num:2}: {count:3} times\n")
            
            f.write("\n" + "=" * 40 + "\n\n")
        
        f.write("🤖 Prediction Strategies Used:\n")
        f.write("- Frequency Analysis with Recency Weighting\n")
        f.write("- Pattern Recognition (Pairs & Triplets)\n")
        f.write("- Hot/Cold Number Identification\n")
        f.write("- Machine Learning (Random Forest)\n")
        f.write("- Smart Combination Generation\n")

if __name__ == "__main__":
    main()
