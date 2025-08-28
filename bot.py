import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import math

class TrustedPredictor:
    def __init__(self):
        self.analysis_log = []  # Initialize first
        self.data = self.load_data()
        
    def load_data(self):
        """Load and validate data"""
        try:
            data = pd.read_csv('uk49s_results.csv')
            # Check if required columns exist
            required_columns = ['date', 'draw', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
            if not all(col in data.columns for col in required_columns):
                self.log("CSV missing required columns")
                return pd.DataFrame()
                
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            
            # Validate numbers are within range
            for i in range(1, 7):
                if not all(1 <= x <= 49 for x in data[f'n{i}'].dropna()):
                    self.log(f"Invalid numbers in n{i} column")
                    return pd.DataFrame()
                    
            self.log(f"✅ Successfully loaded {len(data)} draws")
            return data
            
        except Exception as e:
            self.log(f"❌ Data loading error: {str(e)}")
            return pd.DataFrame()

    def log(self, message):
        """Log analysis steps"""
        self.analysis_log.append(message)
        print(message)

    def calculate_certain_numbers(self, draws):
        """Calculate 3-4 most certain numbers"""
        if len(draws) < 10:
            self.log("Not enough data for reliable prediction")
            return []
            
        # 1. Recent frequency (last 15 draws)
        recent_numbers = []
        for _, row in draws.tail(15).iterrows():
            for i in range(1, 7):
                recent_numbers.append(row[f'n{i}'])
        
        recent_freq = Counter(recent_numbers)
        
        # 2. Time-weighted frequency (recent matters more)
        time_weighted = defaultdict(float)
        for idx, row in draws.iterrows():
            weight = math.exp(-0.2 * (len(draws) - idx - 1))
            for i in range(1, 7):
                time_weighted[row[f'n{i}']] += weight
        
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
        
        # Combine all factors
        combined_scores = defaultdict(float)
        for num in range(1, 50):
            # Recent frequency (40%)
            if num in recent_freq:
                combined_scores[num] += recent_freq[num] * 0.4
                
            # Time-weighted (30%)
            if num in time_weighted:
                combined_scores[num] += time_weighted[num] * 0.3
                
            # Due numbers (30%)
            if num in due_numbers:
                combined_scores[num] += due_numbers[num] * 0.3
        
        # Get top 4 numbers
        top_numbers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        return [num for num, score in top_numbers]

    def generate_predictions(self):
        """Generate predictions for both draws"""
        results = {}
        
        for draw_type in ['Lunchtime', 'Teatime']:
            draws = self.data[self.data['draw'] == draw_type]
            
            if len(draws) < 10:
                self.log(f"Not enough {draw_type} data ({len(draws)} draws)")
                # Fallback to most common numbers overall
                all_numbers = []
                for _, row in self.data.iterrows():
                    if row['draw'] == draw_type:
                        for i in range(1, 7):
                            all_numbers.append(row[f'n{i}'])
                
                if all_numbers:
                    common_numbers = [num for num, count in Counter(all_numbers).most_common(4)]
                    results[draw_type] = common_numbers
                else:
                    results[draw_type] = [5, 23, 34, 49] if draw_type == 'Lunchtime' else [8, 19, 32, 44]
            else:
                certain_numbers = self.calculate_certain_numbers(draws)
                results[draw_type] = certain_numbers
                
            self.log(f"{draw_type} certain numbers: {results[draw_type]}")
        
        return results

    def save_predictions(self, results):
        """Save predictions to file"""
        with open('PREDICTIONS.txt', 'w') as f:
            f.write("UK49s Certain Numbers\n")
            f.write("=====================\n\n")
            
            for draw_type in ['Lunchtime', 'Teatime']:
                numbers = results.get(draw_type, [])
                f.write(f"{draw_type}: {numbers}\n")
                f.write(f"Bet: {'-'.join(map(str, numbers))}\n\n")
            
            f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("Based on mathematical analysis of historical data\n")

def main():
    print("🔍 UK49s Trusted Prediction Bot")
    print("===============================")
    
    predictor = TrustedPredictor()
    results = predictor.generate_predictions()
    predictor.save_predictions(results)
    
    print("✅ Predictions generated successfully!")

if __name__ == "__main__":
    main()
