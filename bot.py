def calculate_certain_numbers(self, draws):  
    """Calculate 3-4 most certain numbers using all available history"""  
    if len(draws) < 5:  # allow smaller datasets
        self.log("Not enough data for reliable prediction")
        return []

    # 1. Recent frequency (use all history)
    recent_numbers = []
    for _, row in draws.iterrows():  # changed from .tail(15)
        for i in range(1, 7):
            recent_numbers.append(row[f'n{i}'])
    recent_freq = Counter(recent_numbers)

    # 2. Time-weighted frequency
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

    # 4. Combine all factors
    combined_scores = defaultdict(float)
    for num in range(1, 50):
        combined_scores[num] += recent_freq.get(num, 0) * 0.4
        combined_scores[num] += time_weighted.get(num, 0) * 0.3
        combined_scores[num] += due_numbers.get(num, 0) * 0.3

    # Get top 4 numbers
    top_numbers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    return [num for num, score in top_numbers]
