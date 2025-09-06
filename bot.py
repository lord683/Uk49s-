import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictor:
    def __init__(self):
        self.analysis_log = []
        self.data = self.load_data()
        self.all_numbers = list(range(1, 50))
        
    def load_data(self):  
        """Load and validate data with better error handling"""  
        try:  
            data = pd.read_csv('uk49s_results.csv')  
            
            # Check if required columns exist  
            required_columns = ['date', 'draw', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'bonus']  
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
    
    def calculate_number_patterns(self, draws):
        """Advanced pattern recognition for numbers"""
        patterns = {}
        
        for num in self.all_numbers:
            # Calculate appearance frequency
            freq = sum(1 for _, row in draws.iterrows() 
                      if num in [row[f'n{i}'] for i in range(1, 7)])
            
            # Calculate last appearance gap
            last_seen = None
            for idx, row in draws.iterrows():
                if num in [row[f'n{i}'] for i in range(1, 7)]:
                    last_seen = idx
            
            gap = len(draws) - last_seen if last_seen is not None else len(draws)
            
            # Calculate average gap between appearances
            appearances = []
            for idx, row in draws.iterrows():
                if num in [row[f'n{i}'] for i in range(1, 7)]:
                    appearances.append(idx)
            
            gaps = []
            if len(appearances) > 1:
                gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
            else:
                avg_gap = 0
            
            # Calculate probability based on gap analysis
            if avg_gap > 0:
                probability = min(1.0, gap / avg_gap)
            else:
                probability = 0.5
                
            patterns[num] = {
                'frequency': freq,
                'last_seen': gap,
                'avg_gap': avg_gap,
                'probability': probability,
                'gaps': gaps
            }
            
        return patterns
    
    def find_hot_cold_numbers(self, patterns, draws):
        """Identify hot and cold numbers with advanced metrics"""
        # Calculate z-scores for frequency
        frequencies = [patterns[num]['frequency'] for num in self.all_numbers]
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        hot_numbers = []
        cold_numbers = []
        
        for num in self.all_numbers:
            z_score = (patterns[num]['frequency'] - mean_freq) / std_freq if std_freq > 0 else 0
            
            if z_score > 0.5:  # More frequent than average
                hot_numbers.append((num, patterns[num]['probability']))
            elif z_score < -0.5:  # Less frequent than average
                cold_numbers.append((num, patterns[num]['probability']))
                
        # Sort by probability (descending for hot, ascending for cold)
        hot_numbers.sort(key=lambda x: x[1], reverse=True)
        cold_numbers.sort(key=lambda x: x[1])
        
        return hot_numbers, cold_numbers
    
    def analyze_number_pairs(self, draws):
        """Analyze frequently occurring number pairs"""
        pairs_count = defaultdict(int)
        
        for _, row in draws.iterrows():
            numbers = [row[f'n{i}'] for i in range(1, 7)]
            # Count all pairs in this draw
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    pairs_count[pair] += 1
        
        # Return top pairs
        return sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)[:20]
    
    def predict_with_ensemble(self, draws):
        """Use ensemble approach combining multiple strategies"""
        patterns = self.calculate_number_patterns(draws)
        hot_numbers, cold_numbers = self.find_hot_cold_numbers(patterns, draws)
        top_pairs = self.analyze_number_pairs(draws)
        
        # Strategy 1: High probability hot numbers
        hot_candidates = [num for num, prob in hot_numbers[:10]]
        
        # Strategy 2: Due cold numbers with high probability
        cold_candidates = [num for num, prob in cold_numbers[:10] if prob > 0.7]
        
        # Strategy 3: Numbers from frequent pairs
        pair_candidates = []
        for pair, count in top_pairs:
            if count >= 2:  # At least appeared together twice
                pair_candidates.extend(pair)
        
        # Combine all strategies
        all_candidates = list(set(hot_candidates + cold_candidates + pair_candidates))
        
        # Score candidates based on multiple factors
        candidate_scores = {}
        for num in all_candidates:
            score = 0
            
            # Base score from probability
            score += patterns[num]['probability'] * 40
            
            # Bonus for being in hot numbers
            if num in [n for n, p in hot_numbers[:15]]:
                score += 20
                
            # Bonus for being in frequent pairs
            pair_bonus = sum(count for pair, count in top_pairs if num in pair)
            score += min(pair_bonus * 5, 20)
            
            # Bonus for recent appearance pattern
            if patterns[num]['last_seen'] <= patterns[num].get('avg_gap', 10) / 2:
                score += 15
                
            candidate_scores[num] = score
        
        # Select top candidates
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [num for num, score in top_candidates[:6]], patterns
    
    def generate_advanced_predictions(self):
        """Generate predictions using advanced algorithms"""
        results = {}
        confidence_scores = {}
        
        for draw_type in ['Lunchtime', 'Teatime']:
            draws = self.data[self.data['draw'] == draw_type]
            
            if len(draws) < 20:
                self.log(f"⚠️ Limited {draw_type} data ({len(draws)} draws), using fallback")
                # Fallback to statistical analysis
                all_numbers = []
                for _, row in draws.iterrows():
                    for i in range(1, 7):
                        all_numbers.append(row[f'n{i}'])
                
                # Use frequency analysis with probability weighting
                number_counts = Counter(all_numbers)
                total_draws = len(draws)
                predicted_numbers = [num for num, count in number_counts.most_common(6)]
                results[draw_type] = predicted_numbers
                confidence_scores[draw_type] = 0.5
            else:
                predicted_numbers, patterns = self.predict_with_ensemble(draws)
                results[draw_type] = predicted_numbers
                
                # Calculate confidence score
                avg_prob = np.mean([patterns[num]['probability'] for num in predicted_numbers])
                confidence_scores[draw_type] = min(0.9, avg_prob * 1.2)
                
            self.log(f"{draw_type} predicted numbers: {results[draw_type]}")
            self.log(f"{draw_type} confidence: {confidence_scores[draw_type]:.2%}")
        
        return results, confidence_scores
    
    def save_predictions(self, results, confidence_scores):
        """Save predictions with detailed analysis"""
        with open('ADVANCED_PREDICTIONS.txt', 'w') as f:
            f.write("UK49s Advanced Predictions\n")
            f.write("==========================\n\n")
            
            for draw_type in ['Lunchtime', 'Teatime']:
                numbers = results.get(draw_type, [])
                f.write(f"{draw_type} Prediction:\n")
                f.write(f"Numbers: {sorted(numbers)}\n")
                f.write(f"Confidence: {confidence_scores.get(draw_type, 0):.2%}\n")
                f.write(f"Bet: {'-'.join(map(str, sorted(numbers)))}\n\n")
            
            f.write("Analysis Summary:\n")
            for log_entry in self.analysis_log[-10:]:  # Last 10 log entries
                f.write(f"- {log_entry}\n")
                
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Based on advanced pattern recognition and statistical analysis\n")

def main():
    print("🔍 UK49s Advanced Prediction Bot")
    print("================================")
    
    predictor = AdvancedPredictor()
    
    if len(predictor.data) == 0:
        print("❌ Failed to load data. Please check your CSV file.")
        return
    
    results, confidence_scores = predictor.generate_advanced_predictions()
    predictor.save_predictions(results, confidence_scores)
    
    print("✅ Advanced predictions generated successfully!")
    print("📊 Confidence scores:")
    for draw_type in ['Lunchtime', 'Teatime']:
        print(f"   {draw_type}: {confidence_scores.get(draw_type, 0):.2%}")

if __name__ == "__main__":
    main()
