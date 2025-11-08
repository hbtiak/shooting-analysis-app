# 🎯 Shooting Training AI Analyst

A Streamlit web application that provides AI-powered analysis and recommendations for 10m rifle shooting training.

## Features

- **Performance Tracking**: Monitor scores, consistency, and progression over time
- **AI-Powered Insights**: Get personalized training recommendations using OpenAI GPT
- **Predictive Analysis**: Forecast future performance based on historical data
- **Visual Analytics**: Interactive charts and graphs for performance trends
- **Drill Recommendations**: Specific training exercises based on performance gaps

## Quick Start

1. **Generate Sample Data**: Click "Generate Sample Data" to start with demo data
2. **Upload Your Data**: Use the CSV uploader for your own shooting data
3. **Get AI Insights**: Click "Generate AI Analysis" for personalized recommendations

## Data Format

For custom data upload, use this CSV format:

```csv
session_id,athlete_id,date,shot_number,score,hold_time,trigger_time,heart_rate,fatigue_level,series
S001,A001,2024-01-15,1,10.5,4.2,0.3,68,3,1
S001,A001,2024-01-15,2,10.3,4.1,0.32,67,3,1