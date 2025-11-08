import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_dataset():
    """Create a comprehensive sample dataset for testing"""
    np.random.seed(42)
    
    athletes = ['A001', 'A002', 'A003']
    all_data = []
    
    for athlete in athletes:
        base_date = datetime.now() - timedelta(days=30)
        
        for session in range(15):
            session_date = base_date + timedelta(days=session*2)
            
            # Simulate different performance levels
            if athlete == 'A001':
                base_score = 625 + (session * 0.8)  # Steady improvement
            elif athlete == 'A002':
                base_score = 622 + (session * 1.5)  # Rapid improvement
            else:
                base_score = 618 + (session * 0.5)  # Slow improvement
            
            for shot in range(60):
                score_variation = np.random.normal(0, 0.3)
                score = 10.0 + (base_score - 600) / 25 + score_variation
                score = max(9.0, min(10.9, score))
                
                all_data.append({
                    'session_id': f"S{session:03d}",
                    'athlete_id': athlete,
                    'date': session_date.strftime('%Y-%m-%d'),
                    'shot_number': shot + 1,
                    'score': round(score, 1),
                    'hold_time': round(np.random.normal(4.2, 0.6), 2),
                    'trigger_time': round(np.random.normal(0.35, 0.08), 3),
                    'heart_rate': np.random.randint(65, 75),
                    'fatigue_level': np.random.randint(1, 6),
                    'series': (shot // 10) + 1
                })
    
    df = pd.DataFrame(all_data)
    df.to_csv('sample_shooting_data.csv', index=False)
    print("Sample dataset created: sample_shooting_data.csv")
    print(f"Total records: {len(df)}")
    print(f"Athletes: {df['athlete_id'].unique()}")
    print(f"Sessions per athlete: {df['session_id'].nunique() // len(athletes)}")
    return df

if __name__ == "__main__":
    create_sample_dataset()