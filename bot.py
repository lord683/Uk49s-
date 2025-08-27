import pandas as pd
from collections import Counter
from datetime import datetime

def main():
    print(" Generating UK49s predictions...")
    
    try:
        # Read your CSV data
        data = pd.read_csv('uk49s_results.csv')
        
        # Create predictions file
        with open('PREDICTIONS.txt', 'w') as f:
            f.write("🎯 UK49s DAILY PREDICTIONS\n")
            f.write("==========================\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Lunchtime predictions
            lunch = data[data['draw'] == 'Lunchtime']
            if not lunch.empty:
                nums = []
                for i in range(1, 7):
                    nums.extend(lunch[f'n{i}'].tolist())
                common = Counter(nums).most_common(6)
                pred = sorted([num for num, count in common])
                
                f.write("⭐ LUNCHTIME:\n")
                f.write(f"Numbers: {pred}\n")
                f.write(f"Bet: {'-'.join(map(str, pred))}\n\n")
            
            # Teatime predictions
            tea = data[data['draw'] == 'Teatime']
            if not tea.empty:
                nums = []
                for i in range(1, 7):
                    nums.extend(tea[f'n{i}'].tolist())
                common = Counter(nums).most_common(6)
                pred = sorted([num for num, count in common])
                
                f.write("⭐ TEATIME:\n")
                f.write(f"Numbers: {pred}\n")
                f.write(f"Bet: {'-'.join(map(str, pred))}\n")
        
        print("✅ Predictions generated in PREDICTIONS.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
        
