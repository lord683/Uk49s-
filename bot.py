import csv
import itertools
from collections import Counter

# === CONFIG ===
DATA_FILE = 'data/uk49s_results.csv'
OUTPUT_FILE = 'predictions.txt'
NUMBERS_RANGE = list(range(1, 50))  # UK49s numbers
PREDICT_COUNT = 6  # numbers per draw

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

# === STRATEGY 1: Most frequent numbers ===
def strategy_most_frequent(history):
    counter = Counter()
    for h in history:
        counter.update(h['numbers'])
    most_common = [num for num, _ in counter.most_common(PREDICT_COUNT)]
    return most_common

# === STRATEGY 2: Least frequent numbers ===
def strategy_least_frequent(history):
    counter = Counter()
    for h in history:
        counter.update(h['numbers'])
    least_common = [num for num, _ in counter.most_common()][-PREDICT_COUNT:]
    return least_common

# === STRATEGY 3: Consecutive pattern ===
def strategy_consecutive(history):
    consecutive = []
    for h in history:
        nums = sorted(h['numbers'])
        for i in range(len(nums)-1):
            if nums[i]+1 == nums[i+1]:
                consecutive.append(nums[i])
                consecutive.append(nums[i+1])
    return list(set(consecutive))[:PREDICT_COUNT]

# === STRATEGY 4: Last draw repetition ===
def strategy_last_draw(history):
    last = history[-1]['numbers']
    return last[:PREDICT_COUNT]

# === STRATEGY 5: Random combination of high/low numbers ===
import random
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

# === SAVE RESULTS ===
with open(OUTPUT_FILE, 'w') as f:
    f.write("UK49s Prediction Results\n")
    f.write("=======================\n\n")
    for strategy, nums in predictions.items():
        f.write(f"{strategy}: {sorted(nums)}\n")

print("Predictions generated successfully!")
for strategy, nums in predictions.items():
    print(f"{strategy}: {sorted(nums)}")
