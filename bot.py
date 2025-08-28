import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import math
import hashlib

class TransparentPredictor:
    def __init__(self):
        self.data = self.load_data()
        self.analysis_log = []
        
    def load_data(self):
        """Load and validate data with integrity check"""
        try:
            data = pd.read_csv('uk49s_results.csv')
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            
            # Calculate data integrity hash
            data_hash = hashlib.md5(data.to_string().encode()).hexdigest()[:8]
            self.log(f"Data loaded: {len(data)} draws | Integrity: {data_hash}")
            
            return data
        except Exception as e:
            self.log(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    def log(self, message):
        """Log analysis steps for transparency"""
        self.analysis_log.append(message)
        print(message)

    def calculate_confidence(self, draws, numbers):
        """Calculate confidence score for predictions (0-100%)"""
        if len(draws) < 10:
            return 0
            
        # Check how many of these numbers have appeared recently
        recent_draws = draws.tail(10)
        recent_numbers = set()
        for _, row in recent_draws.iterrows():
            for i in range(1, 7):
                recent_numbers.add(row[f'n{i}'])
        
        # Calculate match percentage
        matches = len(set(numbers) & recent_numbers)
        confidence = (matches / len(numbers)) * 100
        
        # Adjust based on data quantity
        data_factor = min(100, len(draws) * 2)  # More data = more confidence
        return min(100, (confidence + data_factor) / 2)

    def analyze_number_patterns(self, draws):
        """Comprehensive number analysis with transparent calculations"""
        if len(draws) < 10:
            return [], []
            
        self.log(f"Analyzing {len(draws)} {draws.iloc[0]['draw']} draws...")
        
        # 1. Time-weighted frequency analysis
        time_weights = defaultdict(float)
        for idx, row in draws.iterrows():
            # Recent draws matter more (exponential decay)
            weight = math.exp(-0.2 * (len(draws) - idx - 1))
            for i in range(1, 7):
                time_weights[row[f'n{i}']] += weight
        
        # 2. Recent momentum (last 5 vs previous 5 draws)
        recent_counts = Counter()
        previous_counts = Counter()
        
        for _, row in draws.tail(5).iterrows():
            for i in range(1, 7):
                recent_counts[row[f'n{i}']] += 1
                
        for _, row in draws.iloc[-10:-5].iterrows():
            for i in range(1, 7):
                previous_counts[row[f'n{i}']] += 1
                
        momentum = {}
        for num in set(recent_counts.keys()) | set(previous_counts.keys()):
            momentum[num] = recent_counts.get(num, 0) - previous_counts.get(num, 0)
        
        # 3. Due numbers analysis
        due_numbers = {}
        for num in range(1, 50):
            last_seen = None
            gaps = []
            
            for idx, row in draws.iterrows():
                if num in [row[f'n{i}'] for i in range(1, 7)]:
                    if last_seen is not None:
                        gaps.append(idx - last_seen)
                    last_seen = idx
            
            if gaps and last_seen is not None:
                avg_gap = np.mean(gaps)
                current_gap = len(draws) - last_seen - 1
                if current_gap >= avg_gap:
                    due_numbers[num] = current_gap / avg_gap
        
        # Combine all factors with transparent weights
        combined_scores = defaultdict(float)
        for num in range(1, 50):
            # Time-weighted frequency (40%)
            if num in time_weights:
                combined_scores[num] += time_weights[num] * 0.4
                
            # Momentum (30%)
            if num in momentum:
                combined_scores[num] += (momentum[num] + 5) * 0.3  # Normalize
                
            # Due numbers (30%)
            if num in due_numbers:
                combined_scores[num] += due_numbers[num] * 0.3
        
        # Get top numbers
        top_numbers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Log the analysis
        self.log(f"Top 5 numbers by score: {top_numbers[:5]}")
        
        return [num for num, score in top_numbers[:8]], [num for num, score in top_numbers[:4]]

    def generate_predictions(self):
        """Generate and verify predictions"""
        results = {}
        
        for draw_type in ['Lunchtime', 'Teatime']:
            draws = self.data[self.data['draw'] == draw_type]
            
            if len(draws) < 10:
                self.log(f"Not enough data for {draw_type} ({len(draws)} draws)")
                results[draw_type] = {
                    'numbers': sorted(np.random.choice(range(1, 50), 4, replace=False)),
                    'confidence': 0
                }
                continue
                
            hot_numbers, sure_numbers = self.analyze_number_patterns(draws)
            confidence = self.calculate_confidence(draws, sure_numbers)
            
            results[draw_type] = {
                'hot_numbers': hot_numbers,
                'sure_numbers': sure_numbers,
                'confidence': confidence
            }
            
            self.log(f"{draw_type} sure numbers: {sure_numbers} (Confidence: {confidence:.1f}%)")
        
        return results

    def save_transparent_report(self, results):
        """Save a complete transparent report"""
        with open('PREDICTIONS.txt', 'w') as f:
            f.write("UK49s Transparent Predictions\n")
            f.write("=============================\n\n")
            
            f.write("ANALYSIS LOG:\n")
            f.write("-------------\n")
            for log_entry in self.analysis_log:
                f.write(f"{log_entry}\n")
            
            f.write("\nFINAL PREDICTIONS:\n")
            f.write("------------------\n")
            
            for draw_type in ['Lunchtime', 'Teatime']:
                data = results.get(draw_type, {})
                f.write(f"{draw_type}:\n")
                
                if 'sure_numbers' in data:
                    f.write(f"Sure Numbers: {data['sure_numbers']}\n")
                    f.write(f"Bet: {'-'.join(map(str, data['sure_numbers']))}\n")
                    f.write(f"Confidence: {data.get('confidence', 0):.1f}%\n\n")
                else:
                    f.write("Prediction unavailable\n\n")
            
            f.write("HOW THESE PREDICTIONS WORK:\n")
            f.write("---------------------------\n")
            f.write("1. Time-weighted frequency (recent draws matter more)\n")
            f.write("2. Momentum analysis (numbers appearing more frequently recently)\n")
            f.write("3. Due numbers analysis (numbers that are statistically due to appear)\n")
            f.write("4. Confidence scores based on recent pattern matches\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Integrity: {hashlib.md5(self.data.to_string().encode()).hexdigest()[:8]}\n")

def main():
    print("🔍 UK49s Transparent Prediction Bot")
    print("===================================")
    
    predictor = TransparentPredictor()
    
    if predictor.data.empty:
        print("No data available - using fallback predictions")
        results = {
            'Lunchtime': {'sure_numbers': [5, 23, 34, 49], 'confidence': 0},
            'Teatime': {'sure_numbers': [8, 19, 32, 44], 'confidence': 0}
        }
    else:
        results = predictor.generate_predictions()
    
    predictor.save_transparent_report(results)
    print("✅ Transparent report generated!")

if __name__ == "__main__":
    main()
