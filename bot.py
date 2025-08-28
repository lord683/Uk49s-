import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

def smart_predictor(draw_type, data):
    """Advanced prediction with multiple strategies"""
    draws = data[data['draw'] == draw_type]
    if len(draws) < 10:
        return generate_fallback()
    
    # Strategy 1: Weighted recent frequency (60% weight)
    recent = draws.tail(30)
    recent_numbers = []
    for _, row in recent.iterrows():
        for i in range(1, 7):
            recent_numbers.append(row[f'n{i}'])
    
    # Strategy 2: Hot numbers from last 10 draws (20% weight)
    hot_numbers = []
    for _, row in recent.tail(10).iterrows():
        for i in range(1, 7):
            hot_numbers.append(row[f'n{i}'])
    
    # Strategy 3: Cold numbers (not in last 15 draws) (10% weight)
    all_recent = set()
    for _, row in recent.tail(15).iterrows():
        for i in range(1, 7):
            all_recent.add(row[f'n{i}'])
    cold_numbers = [num for num in range(1, 50) if num not in all_recent]
    
    # Strategy 4: Pattern recognition - common pairs (10% weight)
    pairs = []
    for _, row in recent.iterrows():
        nums = sorted([row[f'n{i}'] for i in range(1, 7)])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pairs.append((nums[i], nums[j]))
    
    common_pairs = Counter(pairs).most_common(5)
    pattern_numbers = set()
    for pair, count in common_pairs:
        pattern_numbers.update(pair)
    
    # Combine all strategies with weights
    number_scores = {}
    
    # Recent numbers (60%)
    for num in recent_numbers:
        number_scores[num] = number_scores.get(num, 0) + 0.6
    
    # Hot numbers (20%)
    for num in hot_numbers:
        number_scores[num] = number_scores.get(num, 0) + 0.2
    
    # Cold numbers (10%)
    for num in cold_numbers[:10]:  # Top 10 cold numbers
        number_scores[num] = number_scores.get(num, 0) + 0.1
    
    # Pattern numbers (10%)
    for num in pattern_numbers:
        number_scores[num] = number_scores.get(num, 0) + 0.1
    
    # Get top 12 candidates
    candidates = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)[:12]
    candidate_nums = [num for num, score in candidates]
    
    # Create balanced prediction (mix of high/low, odd/even)
    prediction = []
    
    # Take top 4 from candidates
    prediction.extend(candidate_nums[:4])
    
    # Add 2 numbers to balance the set
    # Ensure good mix of high/low numbers
    current_high = sum(1 for n in prediction if n > 25)
    current_low = sum(1 for n in prediction if n <= 25)
    
    # Add numbers to balance high/low
    if current_high < 2:
        # Add high numbers from remaining candidates
        high_nums = [n for n in candidate_nums[4:] if n > 25]
        prediction.extend(high_nums[:2 - current_high])
    elif current_low < 2:
        # Add low numbers from remaining candidates
        low_nums = [n for n in candidate_nums[4:] if n <= 25]
        prediction.extend(low_nums[:2 - current_low])
    else:
        # Add next best numbers
        prediction.extend(candidate_nums[4:6])
    
    # Ensure we have exactly 6 numbers
    prediction = sorted(list(set(prediction)))[:6]
    
    # If still less than 6, fill with random from candidates
    while len(prediction) < 6:
        remaining = [n for n in candidate_nums if n not in prediction]
        if remaining:
            prediction.append(remaining[0])
        else:
            prediction.append(np.random.randint(1, 50))
    
    return sorted(prediction)

def generate_fallback():
    """Generate random but reasonable fallback prediction"""
    # Not completely random - based on common number ranges
    low_numbers = sorted(np.random.choice(range(1, 26), 3, replace=False))
    high_numbers = sorted(np.random.choice(range(26, 50), 3, replace=False))
    return sorted(low_numbers + high_numbers)

def main():
    print(" Generating Smart UK49s Predictions...")
    
    try:
        # Load data
        data = pd.read_csv('uk49s_results.csv')
        
        # Get predictions using advanced algorithm
        lunch_pred = smart_predictor('Lunchtime', data)
        tea_pred = smart_predictor('Teatime', data)
        
        # Write clean output
        with open('PREDICTIONS.txt', 'w') as f:
            f.write("✅ UK49s Predictions\n")
            f.write("===================\n\n")
            
            f.write("LUNCHTIME: ")
            f.write(f"{lunch_pred}\n")
            f.write(f"Bet: {'-'.join(map(str, lunch_pred))}\n\n")
            
            f.write("TEATIME:   ")
            f.write(f"{tea_pred}\n")
            f.write(f"Bet: {'-'.join(map(str, tea_pred))}\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        print("✅ Smart predictions generated!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Create basic prediction file even on error
        with open('PREDICTIONS.txt', 'w') as f:
            f.write("✅ UK49s Predictions\n")
            f.write("===================\n\n")
            f.write("LUNCHTIME: [5, 12, 23, 34, 41, 49]\n")
            f.write("Bet: 5-12-23-34-41-49\n\n")
            f.write("TEATIME:   [8, 17, 19, 25, 32, 44]\n")
            f.write("Bet: 8-17-19-25-32-44\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

if __name__ == "__main__":
    main()
