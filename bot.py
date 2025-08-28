import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import math

def load_data():
    """Load and validate the data"""
    try:
        data = pd.read_csv('uk49s_results.csv')
        data['date'] = pd.to_datetime(data['date'])
        return data.sort_values('date')
    except:
        return None

def calculate_deep_patterns(draws, draw_type):
    """DEEP pattern analysis for real predictions"""
    specific_draws = draws[draws['draw'] == draw_type]
    if len(specific_draws) < 10:
        return None
    
    patterns = {}
    
    # DEEP PATTERN 1: Time-weighted frequency with exponential decay
    numbers_weighted = []
    max_weight = 0
    for idx, row in specific_draws.iterrows():
        # Recent draws get exponentially more weight
        weight = math.exp(-0.15 * (len(specific_draws) - idx - 1))
        max_weight = max(max_weight, weight)
        for i in range(1, 7):
            numbers_weighted.extend([row[f'n{i}']] * int(weight * 1000))
    
    patterns['weighted_freq'] = Counter(numbers_weighted)
    
    # DEEP PATTERN 2: Consecutive appearance patterns
    appearance_pattern = {}
    for num in range(1, 50):
        appearances = []
        current_streak = 0
        for _, row in specific_draws.iterrows():
            numbers = [row[f'n{i}'] for i in range(1, 7)]
            if num in numbers:
                current_streak += 1
            else:
                if current_streak > 0:
                    appearances.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            appearances.append(current_streak)
        appearance_pattern[num] = appearances
    
    patterns['appearance_pattern'] = appearance_pattern
    
    # DEEP PATTERN 3: Gap prediction (when numbers will reappear)
    last_appearance = {}
    gap_prediction = {}
    for num in range(1, 50):
        last_seen = None
        gaps = []
        for idx, row in specific_draws.iterrows():
            numbers = [row[f'n{i}'] for i in range(1, 7)]
            if num in numbers:
                if last_seen is not None:
                    gaps.append(idx - last_seen)
                last_seen = idx
        
        if gaps:
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            last_appearance[num] = len(specific_draws) - last_seen - 1 if last_seen else None
            gap_prediction[num] = (avg_gap, std_gap, last_appearance[num])
    
    patterns['gap_analysis'] = gap_prediction
    patterns['last_appearance'] = last_appearance
    
    # DEEP PATTERN 4: Number pair momentum
    pair_momentum = Counter()
    for i in range(10, len(specific_draws)):
        current_row = specific_draws.iloc[i]
        current_numbers = set([current_row[f'n{j}'] for j in range(1, 7)])
        
        prev_row = specific_draws.iloc[i-1]
        prev_numbers = set([prev_row[f'n{j}'] for j in range(1, 7)])
        
        # Numbers that appeared together recently
        for num1 in current_numbers:
            for num2 in current_numbers:
                if num1 < num2:
                    pair_momentum[(num1, num2)] += 1
    
    patterns['pair_momentum'] = pair_momentum
    
    # DEEP PATTERN 5: Positional hot streaks
    positional_streaks = {i: {} for i in range(1, 7)}
    for pos in range(1, 7):
        current_streak = {}
        for idx, row in specific_draws.iterrows():
            num = row[f'n{pos}']
            if num in current_streak:
                current_streak[num] += 1
            else:
                current_streak[num] = 1
        positional_streaks[pos] = current_streak
    
    patterns['positional_streaks'] = positional_streaks
    
    return patterns

def predict_upcoming_numbers(patterns, draw_type):
    """PREDICT actual upcoming numbers based on deep patterns"""
    if not patterns:
        return None
    
    # PREDICTION STRATEGY 1: Numbers due to appear based on gap analysis
    due_numbers = []
    for num, (avg_gap, std_gap, last_seen) in patterns['gap_analysis'].items():
        if last_seen is not None and last_seen >= avg_gap - std_gap:
            # Number is due to appear based on historical gaps
            due_score = (last_seen - (avg_gap - std_gap)) / std_gap if std_gap > 0 else 1
            due_numbers.append((num, due_score))
    
    # Sort by most due numbers
    due_numbers.sort(key=lambda x: x[1], reverse=True)
    
    # PREDICTION STRATEGY 2: Numbers with positive momentum
    top_weighted = [num for num, count in patterns['weighted_freq'].most_common(20)]
    
    # PREDICTION STRATEGY 3: Numbers from hot pairs
    hot_pairs = [pair for pair, count in patterns['pair_momentum'].most_common(15)]
    pair_numbers = set()
    for pair in hot_pairs:
        pair_numbers.update(pair)
    
    # PREDICTION STRATEGY 4: Numbers ending streaks
    streak_candidates = []
    for pos in range(1, 7):
        for num, streak in patterns['positional_streaks'][pos].items():
            if streak >= 2:  # Numbers on positional streaks
                streak_candidates.append(num)
    
    # COMBINE ALL PREDICTIONS with weights
    prediction_scores = {}
    
    # Due numbers (40% weight)
    for num, score in due_numbers[:15]:
        prediction_scores[num] = prediction_scores.get(num, 0) + 0.4 * min(score, 3)
    
    # Weighted frequency (25% weight)
    for idx, num in enumerate(top_weighted):
        prediction_scores[num] = prediction_scores.get(num, 0) + 0.25 * (1 - idx/20)
    
    # Pair momentum (20% weight)
    for num in pair_numbers:
        prediction_scores[num] = prediction_scores.get(num, 0) + 0.20
    
    # Streak candidates (15% weight)
    for num in streak_candidates:
        prediction_scores[num] = prediction_scores.get(num, 0) + 0.15
    
    # Get top candidates
    top_candidates = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # CREATE FINAL PREDICTION with mathematical optimization
    final_prediction = []
    candidates = [num for num, score in top_candidates]
    
    # Ensure good number distribution
    low_numbers = [n for n in candidates if n <= 25]
    high_numbers = [n for n in candidates if n > 25]
    
    # Select 3 low and 3 high numbers from top candidates
    final_prediction.extend(low_numbers[:3])
    final_prediction.extend(high_numbers[:3])
    
    # If not enough numbers, fill with next best
    if len(final_prediction) < 6:
        remaining = [n for n in candidates if n not in final_prediction]
        final_prediction.extend(remaining[:6 - len(final_prediction)])
    
    return sorted(final_prediction)

def main():
    print("🔮 Calculating DEEP UK49s Predictions...")
    print("🎯 Predicting ACTUAL upcoming numbers")
    
    data = load_data()
    if data is None:
        print("❌ Failed to load data")
        return
    
    results = {}
    
    for draw_type in ['Lunchtime', 'Teatime']:
        print(f"\n📊 Analyzing {draw_type} for upcoming prediction...")
        
        # Calculate DEEP patterns
        patterns = calculate_deep_patterns(data, draw_type)
        
        if patterns:
            # PREDICT upcoming numbers
            prediction = predict_upcoming_numbers(patterns, draw_type)
            results[draw_type] = prediction
            
            if prediction:
                print(f"   🎯 Predicted upcoming: {prediction}")
            else:
                print(f"   ❌ Could not predict {draw_type}")
        else:
            print(f"   ❌ Not enough data for {draw_type}")
            results[draw_type] = None
    
    # Write PREDICTIONS only
    with open('PREDICTIONS.txt', 'w') as f:
        f.write("✅ UK49s Predictions\n")
        f.write("===================\n\n")
        
        for draw_type in ['Lunchtime', 'Teatime']:
            prediction = results.get(draw_type)
            if prediction:
                f.write(f"{draw_type.upper():12} {prediction}\n")
                f.write(f"Bet: {'-'.join(map(str, prediction))}\n\n")
            else:
                f.write(f"{draw_type.upper():12} [Insufficient data]\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("DEEP pattern analysis - Predicting upcoming numbers\n")
    
    print(f"\n✅ DEEP predictions completed!")

if __name__ == "__main__":
    main()
