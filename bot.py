from collections import Counter, defaultdict
import numpy as np, math

class LottoBot:
    def __init__(self):
        self.last_prediction = None  # memory of last top3

    def calculate_certain_numbers(self, draws):
        if len(draws) < 30:
            self.log("Not enough data for reliable prediction")
            return []

        # 1. Recent frequency (last 25 draws)
        recent_numbers = []
        for _, row in draws.tail(25).iterrows():
            recent_numbers.extend([row[f'n{i}'] for i in range(1, 7)])
        recent_freq = Counter(recent_numbers)

        # 2. Time-weighted frequency
        time_weighted = defaultdict(float)
        for idx, row in draws.iterrows():
            weight = math.exp(-0.08 * (len(draws) - idx - 1))  # smoother decay
            for i in range(1, 7):
                time_weighted[row[f'n{i}']] += weight

        # 3. Due numbers
        due_numbers = {}
        for num in range(1, 50):
            last_seen, gaps = None, []
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

        # Helper: normalize
        def normalize(d):
            if not d: return defaultdict(float)
            mx = max(d.values())
            return {k: v/mx for k, v in d.items()}

        recent_norm = normalize(recent_freq)
        time_norm   = normalize(time_weighted)
        due_norm    = normalize(due_numbers)

        # 4. Combine with weights
        combined = defaultdict(float)
        for num in range(1, 50):
            combined[num] += recent_norm.get(num, 0) * 0.3
            combined[num] += time_norm.get(num, 0) * 0.5
            combined[num] += due_norm.get(num, 0) * 0.2

        # 5. Pair boost (last 50 draws)
        pairs = Counter()
        for _, row in draws.tail(50).iterrows():
            nums = [row[f'n{i}'] for i in range(1, 7)]
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pairs[(nums[i], nums[j])] += 1
        for (a, b), cnt in pairs.items():
            if cnt >= 5:  # strong pair
                combined[a] += 0.05
                combined[b] += 0.05

        # 6. Sort
        sorted_nums = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top3 = [n for n, _ in sorted_nums[:3]]
        bonus = sorted_nums[3][0] if len(sorted_nums) > 3 else None

        # 7. Avoid repeating last prediction
        if self.last_prediction and set(top3) == set(self.last_prediction):
            top3 = [n for n, _ in sorted_nums[3:6]]  # shift to next best

        # Store prediction in memory
        self.last_prediction = top3

        return top3, bonus

    def log(self, msg):
        print(msg)
