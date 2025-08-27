import csv
import random
from collections import Counter

# === CONFIG ===
DATA_FILE = 'uk49s_results.csv'  # your CSV file in root or data folder
OUTPUT_FILE = 'predictions.txt'
NUMBERS_RANGE = list(range(1, 50))  # UK49s numbers
PREDICT_COUNT = 6  # numbers per draw

# === LOAD DATA ===
history_lunchtime = []
history_teatime = []

with open(DATA_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        numbers = [int(row[f'n{i}']) for i in range(1, 7)]
        if row['draw'].lower() == 'lunchtime':
            history_lunchtime.append(numbers)
        elif row['draw'].lower() == 'teatime':
            history_teatime.append(numbers)

# === STRATEGY FUNCTIONS ===
def strategy_most_frequent(history):
    counter = Counter()
    for nums in history:
        counter.update(nums)
    most_common = [num for num, _ in counter.most_common(PREDICT_COUNT)]
    return most_common

def strategy_least_frequent(history):
    counter = Counter()
    for nums in history:
        counter.update(nums)
    least_common = [num for num, _ in counter.most_common()][-PREDICT_COUNT:]
    return least_common

def strategy_consecutive(history):
    consecutive = []
    for nums in history:
        nums_sorted = sorted(nums)
        for i in range(len(nums_sorted)-1):
            if nums_sorted[i]+1 == nums_sorted[i+1]:
                consecutive.append(nums_sorted[i])
                consecutive.append(nums_sorted[i+1])
    return list(set(consecutive))[:PREDICT_COUNT]

def strategy_last_draw(history):
    return history[-1][:PREDICT_COUNT]

def strategy_random_mix(history):
    high = [n for n in NUMBERS_RANGE if n > 25]
    low = [n for n in NUMBERS_RANGE if n <= 25]
    return random.sample(low, 3) + random.sample(high, 3)

# === GENERATE PREDICTIONS FUNCTION ===
def generate_predictions(history):
    return {
        'most_frequent': strategy_most_frequent(history),
        'least_frequent': strategy_least_frequent(history),
        'consecutive': strategy_consecutive(history),
        'last_draw': strategy_last_draw(history),
        'random_mix': strategy_random_mix(history)
    }

# === PREDICTIONS FOR LUNCHTIME AND TEATIME ===
predictions_lunchtime = generate_predictions(history_lunchtime)
predictions_teatime = generate_predictions(history_teatime)

# === SAVE RESULTS ===
with open(OUTPUT_FILE, 'w') as f:
    f.write("UK49s Prediction Results\n")
    f.write("=======================\n\n")

    f.write("=== Lunchtime Predictions ===\n")
    for strategy, nums in predictions_lunchtime.items():
        f.write(f"{strategy}: {sorted(nums)}\n")
    f.write("\n=== Teatime Predictions ===\n")
    for strategy, nums in predictions_teatime.items():
        f.write(f"{strategy}: {sorted(nums)}\n")

print("Predictions generated successfully!\n")
print("=== Lunchtime Predictions ===")
for strategy, nums in predictions_lunchtime.items():
    print(f"{strategy}: {sorted(nums)}")
print("\n=== Teatime Predictions ===")
for strategy, nums in predictions_teatime.items():
    print(f"{strategy}: {sorted(nums)}")
