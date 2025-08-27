import csv
import random
from collections import Counter

# === CONFIG ===
DATA_FILE = 'uk49s_results.csv'  # place CSV in repo root
OUTPUT_FILE = 'predictions.txt'
NUMBERS_RANGE = list(range(1, 50))
PREDICT_COUNT = 6

# === LOAD DATA ===
history = []
with open(DATA_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        numbers = [int(row[f'n{i}']) for i in range(1, 7)]
        history.append({
            'date': row['date'],
            'draw': row['draw'],
            'numbers': numbers,
            'bonus': int(row['bonus'])
        })

# === STRATEGIES ===
def strategy_most_frequent(history):
    counter = Counter()
    for h in history:
        counter.update(h['numbers'])
    return [num for num, _ in counter.most_common(PREDICT_COUNT)]

def strategy_least_frequent(history):
    counter = Counter()
    for h in history:
        counter.update(h['numbers'])
    least_common = [num for num, _ in counter.most_common()][-PREDICT_COUNT:]
    return least_common

def strategy_consecutive(history):
    consecutive = []
    for h in history:
        nums = sorted(h['numbers'])
        for i in range(len(nums)-1):
            if nums[i]+1 == nums[i+1]:
                consecutive.append(nums[i])
                consecutive.append(nums[i+1])
    return list(set(consecutive))[:PREDICT_COUNT]

def strategy_last_draw(history):
    return history[-1]['numbers'][:PREDICT_COUNT]

def strategy_random_mix(history):
    high = [n for n in NUMBERS_RANGE if n > 25]
    low = [n for n in NUMBERS_RANGE if n <= 25]
    return random.sample(low, 3) + random.sample(high, 3)

# === GENERATE PREDICTIONS ===
predictions = {
    'most_frequent': strategy_most_frequent(history),
    'least_frequent': strategy_least_frequent(history),
    'consecutive': strategy_consecutive(history),
    'last_draw': strategy_last_draw(history),
    'random_mix': strategy_random_mix(history)
}

# === SAVE AND SHOW RESULTS ===
with open(OUTPUT_FILE, 'w') as f:
    f.write("UK49s Prediction Results\n")
    f.write("=======================\n\n")
    for strategy, nums in predictions.items():
        f.write(f"{strategy}: {sorted(nums)}\n")

print("Predictions generated successfully!\n")
for strategy, nums in predictions.items():
    print(f"{strategy}: {sorted(nums)}")
