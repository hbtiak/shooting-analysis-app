import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openai
import os

# Configure the app for Streamlit Cloud
st.set_page_config(
    page_title="Shooting Training AI Analyst",
    page_icon="🎯",
    layout="wide"
)

# Enhanced OpenAI initialization for Streamlit Cloud
@st.cache_resource
def init_openai():
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
        st.sidebar.success("✅ OpenAI API key loaded")
    # Try environment variable (for local development)
    elif 'OPENAI_API_KEY' in os.environ:
        api_key = os.environ['OPENAI_API_KEY']
        st.sidebar.success("✅ OpenAI API key loaded from environment")
    else:
        st.sidebar.warning("⚠️ OpenAI API key not found. Using fallback analysis.")
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Test the connection
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Connected'"}],
            max_tokens=5
        )
        return client
    except Exception as e:
        st.sidebar.error(f"❌ OpenAI connection failed")
        return None

client = init_openai()

class ShootingAnalyzer:
    def __init__(self):
        self.metrics_history = []
    
    def generate_sample_data(self, athlete_id="A001", sessions=10):
        """Generate realistic sample shooting data"""
        np.random.seed(42)
        data = []
        
        base_date = datetime.now() - timedelta(days=sessions*2)
        
        for session in range(sessions):
            session_date = base_date + timedelta(days=session*2)
            session_score = 620 + (session * 1.2) + np.random.normal(0, 1.5)
            
            for shot in range(60):
                hold_time = np.random.normal(4.5, 0.8)
                trigger_time = np.random.normal(0.4, 0.1)
                heart_rate = np.random.normal(68, 5)
                
                shot_score = 10.0 + (session * 0.02) + np.random.normal(0, 0.3)
                shot_score = max(9.0, min(10.9, shot_score))
                
                data.append({
                    'session_id': f"S{session:03d}",
                    'athlete_id': athlete_id,
                    'date': session_date.strftime('%Y-%m-%d'),
                    'shot_number': shot + 1,
                    'score': round(shot_score, 1),
                    'hold_time': round(hold_time, 2),
                    'trigger_time': round(trigger_time, 2),
                    'heart_rate': int(heart_rate),
                    'fatigue_level': np.random.randint(1, 6),
                    'series': (shot // 10) + 1
                })
        
        return pd.DataFrame(data)
    
    def calculate_session_metrics(self, df):
        """Calculate key performance metrics for each session"""
        session_metrics = []
        
        for session_id in df['session_id'].unique():
            session_data = df[df['session_id'] == session_id]
            
            metrics = {
                'session_id': session_id,
                'date': session_data['date'].iloc[0],
                'total_score': session_data['score'].sum(),
                'average_score': session_data['score'].mean(),
                'score_std': session_data['score'].std(),
                'hold_stability': 1 / session_data['hold_time'].std() if session_data['hold_time'].std() > 0 else 1,
                'trigger_consistency': 1 / session_data['trigger_time'].std() if session_data['trigger_time'].std() > 0 else 1,
                'heart_rate_variability': session_data['heart_rate'].std(),
                'endurance_drop': self._calculate_endurance_drop(session_data)
            }
            session_metrics.append(metrics)
        
        return pd.DataFrame(session_metrics)
    
    def _calculate_endurance_drop(self, session_data):
        """Calculate performance drop between first and last series"""
        if len(session_data[session_data['series'] == 1]) == 0 or len(session_data[session_data['series'] == 6]) == 0:
            return 0
        first_series_avg = session_data[session_data['series'] == 1]['score'].mean()
        last_series_avg = session_data[session_data['series'] == 6]['score'].mean()
        return first_series_avg - last_series_avg
    
    def predict_next_session(self, metrics_df):
        """Predict expected performance for next session"""
        if len(metrics_df) < 2:
            return None
            
        X = np.arange(len(metrics_df)).reshape(-1, 1)
        y = metrics_df['total_score'].values
        
        if len(metrics_df) >= 3:
            recent_X = X[-3:]
            recent_y = y[-3:]
            trend = np.polyfit(recent_X.flatten(), recent_y, 1)[0]
        else:
            trend = np.polyfit(X.flatten(), y, 1)[0]
        
        next_score = y[-1] + trend
        confidence = max(0.5, min(0.95, 1 - (metrics_df['score_std'].iloc[-1] / 2)))
        
        return {
            'expected_score': round(next_score, 1),
            'confidence': round(confidence, 2),
            'trend': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
        }
    
    def generate_ai_insights(self, metrics_df, session_data):
        """Generate AI-powered insights using OpenAI"""
        if not client:
            return self._generate_fallback_insights(metrics_df)
        
        try:
            latest_session = metrics_df.iloc[-1]
            trend_data = metrics_df.tail(min(3, len(metrics_df)))
            
            prompt = f"""
            Analyze this shooting training data and provide specific, actionable insights:
            
            LATEST SESSION:
            - Total Score: {latest_session['total_score']}
            - Average Score: {latest_session['average_score']:.2f}
            - Consistency (Std Dev): {latest_session['score_std']:.2f}
            - Hold Stability: {latest_session['hold_stability']:.2f}
            - Trigger Consistency: {latest_session['trigger_consistency']:.2f}
            - Endurance Drop: {latest_session['endurance_drop']:.2f}
            
            RECENT TREND (last {len(trend_data)} sessions):
            {trend_data[['session_id', 'total_score', 'average_score', 'score_std']].to_string()}
            
            Please provide:
            1. Key strengths to maintain
            2. Top 2 areas for improvement
            3. Specific drills recommendation for next session
            4. Expected timeline for noticeable improvement
            
            Format the response in a structured way without markdown.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert shooting sports analyst. Provide concise, technical, and actionable feedback for athletes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self._generate_fallback_insights(metrics_df)
    
    def _generate_fallback_insights(self, metrics_df):
        """Generate insights without AI as fallback"""
        latest = metrics_df.iloc[-1]
        
        insights = []
        insights.append("KEY STRENGTHS:")
        if latest['hold_stability'] > 2.0:
            insights.append("• Excellent hold stability - maintain this foundation")
        if latest['trigger_consistency'] > 8.0:
            insights.append("• Good trigger control consistency")
        
        insights.append("\nAREAS FOR IMPROVEMENT:")
        if latest['endurance_drop'] > 0.3:
            insights.append(f"• Endurance: {latest['endurance_drop']:.2f} point drop in later series")
        if latest['score_std'] > 0.4:
            insights.append(f"• Consistency: Shot-to-shot variation needs work")
        
        insights.append("\nNEXT SESSION FOCUS:")
        insights.append("• 20-shot hold stability series")
        insights.append("• 15-shot endurance maintenance drills")
        insights.append("• 10-shot rhythm consistency exercise")
        
        return "\n".join(insights)

def main():
    st.title("🎯 Shooting Training AI Analyst")
    st.markdown("Analyze training performance and get AI-powered improvement recommendations")
    
    # Initialize analyzer
    analyzer = ShootingAnalyzer()
    
    # Sidebar for data input
    st.sidebar.header("Data Configuration")
    athlete_id = st.sidebar.text_input("Athlete ID", value="A001")
    sessions_to_generate = st.sidebar.slider("Training Sessions to Generate", 3, 20, 10)
    
    # Generate or upload data
    if st.sidebar.button("Generate Sample Data") or 'data' not in st.session_state:
        with st.spinner("Generating realistic training data..."):
            st.session_state.data = analyzer.generate_sample_data(athlete_id, sessions_to_generate)
            st.session_state.metrics = analyzer.calculate_session_metrics(st.session_state.data)
            st.success("✅ Sample data generated successfully!")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Or upload your data (CSV)", type=['csv'])
    if uploaded_file:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.metrics = analyzer.calculate_session_metrics(st.session_state.data)
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
    
    if 'data' not in st.session_state:
        st.info("👆 Click 'Generate Sample Data' to start or upload your CSV file")
        return
    
    # Display data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_score = st.session_state.metrics['total_score'].iloc[-1]
        st.metric("Latest Session Score", f"{latest_score:.1f}")
    
    with col2:
        avg_score = st.session_state.metrics['average_score'].iloc[-1]
        st.metric("Average Shot Score", f"{avg_score:.2f}")
    
    with col3:
        consistency = st.session_state.metrics['score_std'].iloc[-1]
        st.metric("Consistency (Std Dev)", f"{consistency:.2f}")
    
    with col4:
        prediction = analyzer.predict_next_session(st.session_state.metrics)
        if prediction:
            st.metric("Next Session Prediction", 
                     f"{prediction['expected_score']:.1f}",
                     delta=f"{prediction['trend']}")
    
    # Performance Charts
    st.subheader("Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_score = px.line(st.session_state.metrics, x='date', y='total_score',
                           title='Total Score Progression', markers=True)
        fig_score.update_layout(height=300)
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        fig_consistency = go.Figure()
        fig_consistency.add_trace(go.Scatter(x=st.session_state.metrics['date'], 
                                           y=st.session_state.metrics['score_std'],
                                           mode='lines+markers',
                                           name='Score Variation (Lower = Better)',
                                           line=dict(color='red')))
        fig_consistency.update_layout(title='Shot Consistency Trend', height=300)
        st.plotly_chart(fig_consistency, use_container_width=True)
    
    # Detailed analysis
    st.subheader("Session Analysis")
    
    selected_session = st.selectbox("Select Session for Detailed Analysis", 
                                   st.session_state.metrics['session_id'].unique())
    
    if selected_session:
        session_data = st.session_state.data[st.session_state.data['session_id'] == selected_session]
        
        col1, col2 = st.columns(2)
        
        with col1:
            series_avg = session_data.groupby('series')['score'].mean().reset_index()
            fig_series = px.bar(series_avg, x='series', y='score',
                               title='Average Score by Series')
            st.plotly_chart(fig_series, use_container_width=True)
        
        with col2:
            fig_dist = px.histogram(session_data, x='score', 
                                   title='Shot Score Distribution',
                                   nbins=20)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # AI Insights Section
    st.subheader("AI-Powered Training Insights")
    
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("🤖 Analyzing performance and generating recommendations..."):
            insights = analyzer.generate_ai_insights(st.session_state.metrics, st.session_state.data)
            
            st.info("💡 Training Recommendations")
            st.text_area("Analysis Results", insights, height=300, key="ai_insights")
    
    # Drill Recommendations
    st.subheader("Recommended Training Drills")
    
    latest_metrics = st.session_state.metrics.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Focus Areas:**")
        if latest_metrics['endurance_drop'] > 0.3:
            st.error("🎯 Endurance Training Needed")
            st.write("• 40-shot continuous series")
            st.write("• Progressive hold time exercises")
        else:
            st.success("✅ Endurance: Good")
    
    with col2:
        st.write("**Technical Skills:**")
        if latest_metrics['score_std'] > 0.4:
            st.error("🎯 Consistency Drills")
            st.write("• 10-shot repeatability sets")
            st.write("• Rhythm timing exercises")
        else:
            st.success("✅ Consistency: Good")
    
    with col3:
        st.write("**Mental Training:**")
        st.write("• Pressure simulation drills")
        st.write("• Breathing control exercises")
        st.write("• Visualization techniques")
    
    # Data export
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("Export Session Metrics"):
        csv = st.session_state.metrics.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name=f"shooting_metrics_{athlete_id}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()