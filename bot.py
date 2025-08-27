import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import json

def load_lottery_data(csv_file, draw_type=None):
    """Load lottery data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        data = []
        
        for _, row in df.iterrows():
            if draw_type and row['draw'].lower() != draw_type.lower():
                continue
                
            # Extract main numbers
            numbers = [int(row[f'n{i}']) for i in range(1, 7)]
            if numbers and len(numbers) == 6:
                data.append(numbers)
        
        return data
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def get_recent_draws(csv_file, draw_type, num_draws=10):
    """Get most recent draws"""
    try:
        df = pd.read_csv(csv_file)
        filtered = df[df['draw'].str.lower() == draw_type.lower()]
        recent = filtered.tail(num_draws)
        
        data = []
        for _, row in recent.iterrows():
            numbers = [int(row[f'n{i}']) for i in range(1, 7)]
            data.append(numbers)
        
        return data
    except Exception as e:
        print(f"Error getting recent draws: {e}")
        return []

def predict_hot_numbers(csv_file, draw_type, draw_size=6, recent_draws=None):
    """Predict based on most frequent numbers"""
    if recent_draws:
        data = get_recent_draws(csv_file, draw_type, recent_draws)
    else:
        data = load_lottery_data(csv_file, draw_type)
    
    if not data:
        return generate_random_prediction(draw_size)
    
    all_numbers = [num for draw in data for num in draw]
    frequency = Counter(all_numbers)
    most_common = [num for num, count in frequency.most_common(draw_size)]
    return sorted(most_common)

def predict_recent_trend(csv_file, draw_type, draw_size=6, lookback=20):
    """Predict based on recent trends"""
    data = get_recent_draws(csv_file, draw_type, lookback)
    
    if not data:
        return generate_random_prediction(draw_size)
    
    number_weights = {}
    
    for i, draw in enumerate(data):
        weight = (i + 1) / lookback
        for num in draw:
            if num in number_weights:
                number_weights[num] += weight
            else:
                number_weights[num] = weight
    
    sorted_numbers = sorted(number_weights.items(), key=lambda x: x[1], reverse=True)
    prediction = [num for num, weight in sorted_numbers[:draw_size]]
    return sorted(prediction)

def predict_avoid_repeats(csv_file, draw_type, draw_size=6):
    """Predict numbers that haven't appeared recently"""
    data = get_recent_draws(csv_file, draw_type, 15)
    
    if not data:
        return generate_random_prediction(draw_size)
    
    recent_numbers = set()
    for draw in data:
        recent_numbers.update(draw)
    
    all_possible = set(range(1, 50))
    cold_numbers = list(all_possible - recent_numbers)
    
    if len(cold_numbers) >= draw_size:
        return sorted(cold_numbers[:draw_size])
    else:
        data = load_lottery_data(csv_file, draw_type)
        all_numbers = [num for draw in data for num in draw]
        frequency = Counter(all_numbers)
        least_common = [num for num, count in frequency.most_common()[-draw_size:]]
        return sorted(least_common)

def predict_combined(csv_file, draw_type, draw_size=6):
    """Combined strategy"""
    hot = predict_hot_numbers(csv_file, draw_type, draw_size)
    recent = predict_recent_trend(csv_file, draw_type, draw_size, 20)
    avoid = predict_avoid_repeats(csv_file, draw_type, draw_size)
    
    all_predictions = hot + recent + avoid
    frequency = Counter(all_predictions)
    combined = [num for num, count in frequency.most_common(draw_size * 2)]
    return sorted(combined[:draw_size])

def generate_random_prediction(draw_size=6):
    """Generate random prediction as fallback"""
    return sorted(np.random.choice(range(1, 50), size=draw_size, replace=False))

def analyze_trends(csv_file, draw_type):
    """Analyze trends and patterns"""
    data = load_lottery_data(csv_file, draw_type)
    
    if not data:
        return {"total_draws": 0}
    
    all_numbers = [num for draw in data for num in draw]
    frequency = Counter(all_numbers)
    
    return {
        "total_draws": len(data),
        "most_common": frequency.most_common(10),
        "least_common": frequency.most_common()[-10:],
        "average_sum": np.mean([sum(draw) for draw in data]),
        "recent_trend": get_recent_draws(csv_file, draw_type, 5)
    }

def main():
    print("🎯 UK49s Prediction Bot - Lunchtime & Teatime")
    print(f"📅 Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    csv_file = "uk49s_results.csv"
    results = {}
    trends = {}
    
    for lottery_type in ["lunchtime", "teatime"]:
        print(f"\n📊 Processing {lottery_type.upper()}...")
        
        data = load_lottery_data(csv_file, lottery_type)
        trends[lottery_type] = analyze_trends(csv_file, lottery_type)
        
        print(f"   Loaded {len(data)} historical draws")
        
        strategies = {
            "Hot Numbers": predict_hot_numbers(csv_file, lottery_type, 6),
            "Recent Trend": predict_recent_trend(csv_file, lottery_type, 6, 25),
            "Avoid Repeats": predict_avoid_repeats(csv_file, lottery_type, 6),
            "Combined Strategy": predict_combined(csv_file, lottery_type, 6)
        }
        
        results[lottery_type] = strategies
        
        print(f"   Predictions for {lottery_type.upper()}:")
        for strategy, numbers in strategies.items():
            print(f"     {strategy:15}: {numbers}")
    
    # Save results
    save_detailed_results(results, trends, csv_file)
    save_next_predictions(results)
    
    print(f"\n✅ Prediction completed successfully!")

def save_detailed_results(results, trends, csv_file):
    """Save detailed prediction results"""
    with open('predictions_output.txt', 'w') as f:
        f.write("UK49s PREDICTIONS - LUNCHTIME & TEATIME\n")
        f.write("=" * 55 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for lottery_type in results:
            f.write(f"{lottery_type.upper()} ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total draws analyzed: {trends[lottery_type]['total_draws']}\n")
            
            if trends[lottery_type]['total_draws'] > 0:
                f.write("\nMost common numbers:\n")
                for num, count in trends[lottery_type]['most_common']:
                    f.write(f"  {num:2}: {count:3} times\n")
                
                f.write("\nLeast common numbers:\n")
                for num, count in trends[lottery_type]['least_common']:
                    f.write(f"  {num:2}: {count:3} times\n")
            
            f.write(f"\n{lottery_type.upper()} PREDICTIONS:\n")
            f.write("-" * 40 + "\n")
            for strategy, numbers in results[lottery_type].items():
                f.write(f"{strategy:20}: {numbers}\n")
            f.write("\n")
        
        f.write("🎯 RECOMMENDED STRAIGHT NUMBERS:\n")
        f.write("-" * 40 + "\n")
        for lottery_type in results:
            combined = results[lottery_type]["Combined Strategy"]
            f.write(f"{lottery_type.upper():15}: {combined}\n")
            f.write(f"                Bet: {'-'.join(map(str, combined))}\n\n")

def save_next_predictions(results):
    """Save next predictions for quick reference"""
    with open('next_predictions.txt', 'w') as f:
        f.write("🎯 NEXT DRAW PREDICTIONS - STRAIGHT NUMBERS\n")
        f.write("=" * 55 + "\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for lottery_type in results:
            combined = results[lottery_type]["Combined Strategy"]
            f.write(f"⭐ {lottery_type.upper()}:\n")
            f.write(f"   Straight Numbers: {combined}\n")
            f.write(f"   Recommended Bet:  {'-'.join(map(str, combined))}\n\n")

if __name__ == "__main__":
    main()
