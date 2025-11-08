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
        return client
    except Exception as e:
        st.sidebar.error(f"❌ OpenAI connection failed")
        return None

client = init_openai()

class ShootingAnalyzer:
    def __init__(self):
        self.metrics_history = []
    
    def generate_sample_data(self, athlete_id="SHOOTER_001", sessions=3):
        """Generate realistic sample shooting data"""
        np.random.seed(42)
        data = []
        
        base_date = datetime.now() - timedelta(days=sessions*3)
        
        for session in range(sessions):
            session_date = base_date + timedelta(days=session*3)
            session_num = session + 1
            
            for shot in range(60):
                series_num = (shot // 10) + 1
                
                # Base performance that improves each session
                base_score = 10.3 + (session * 0.1)
                
                # Simulate fatigue across series
                fatigue_penalty = (series_num - 1) * 0.03
                
                # Add some randomness
                score_variation = np.random.normal(0, 0.15)
                
                final_score = base_score - fatigue_penalty + score_variation
                final_score = max(9.0, min(10.9, final_score))
                
                # Realistic hold and trigger times
                hold_time = 4.2 - (session * 0.1) + (series_num * 0.08) + np.random.normal(0, 0.2)
                trigger_time = 0.3 - (session * 0.01) + (series_num * 0.005) + np.random.normal(0, 0.02)
                
                data.append({
                    'session_id': f"S{session_num:03d}",
                    'athlete_id': athlete_id,
                    'date': session_date.strftime('%Y-%m-%d'),
                    'shot_number': shot + 1,
                    'score': round(final_score, 1),
                    'hold_time': round(hold_time, 2),
                    'trigger_time': round(trigger_time, 3),
                    'heart_rate': 65 + (series_num * 2) + np.random.randint(0, 3),
                    'fatigue_level': min(5, 2 + series_num),
                    'series': series_num
                })
        
        return pd.DataFrame(data)
    
    def validate_data(self, df):
        """Validate uploaded data has required columns"""
        required_columns = ['session_id', 'athlete_id', 'shot_number', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add missing columns with default values if needed
        if 'date' not in df.columns:
            df['date'] = datetime.now().strftime('%Y-%m-%d')
        if 'hold_time' not in df.columns:
            df['hold_time'] = 4.0
        if 'trigger_time' not in df.columns:
            df['trigger_time'] = 0.3
        if 'heart_rate' not in df.columns:
            df['heart_rate'] = 68
        if 'fatigue_level' not in df.columns:
            df['fatigue_level'] = 3
        if 'series' not in df.columns:
            df['series'] = (df['shot_number'] - 1) // 10 + 1
        
        return df
    
    def calculate_session_metrics(self, df):
        """Calculate key performance metrics for each session"""
        session_metrics = []
        
        for session_id in sorted(df['session_id'].unique()):
            session_data = df[df['session_id'] == session_id]
            
            try:
                # Basic metrics
                total_score = session_data['score'].sum()
                average_score = session_data['score'].mean()
                score_std = session_data['score'].std() if len(session_data) > 1 else 0.1
                
                # Stability metrics
                hold_std = session_data['hold_time'].std() if len(session_data) > 1 else 0.5
                trigger_std = session_data['trigger_time'].std() if len(session_data) > 1 else 0.05
                hold_stability = 1 / hold_std if hold_std > 0 else 1
                trigger_consistency = 1 / trigger_std if trigger_std > 0 else 1
                
                # Endurance calculation
                endurance_drop = self._calculate_endurance_drop(session_data)
                
                metrics = {
                    'session_id': session_id,
                    'date': session_data['date'].iloc[0] if 'date' in session_data.columns and len(session_data) > 0 else f"Session {session_id}",
                    'total_score': total_score,
                    'average_score': average_score,
                    'score_std': score_std,
                    'hold_stability': hold_stability,
                    'trigger_consistency': trigger_consistency,
                    'heart_rate_variability': session_data['heart_rate'].std() if len(session_data) > 1 else 0,
                    'endurance_drop': endurance_drop,
                    'shots_count': len(session_data)
                }
                session_metrics.append(metrics)
                
            except Exception as e:
                st.error(f"Error calculating metrics for session {session_id}: {str(e)}")
                continue
        
        if not session_metrics:
            st.error("No valid sessions found in the data")
            return pd.DataFrame()
        
        return pd.DataFrame(session_metrics)
    
    def _calculate_endurance_drop(self, session_data):
        """Calculate performance drop between first and last series"""
        try:
            if 'series' not in session_data.columns:
                return 0
                
            series_groups = session_data.groupby('series')['score'].mean()
            if len(series_groups) < 2:
                return 0
                
            first_series_avg = series_groups.iloc[0]
            last_series_avg = series_groups.iloc[-1]
            return first_series_avg - last_series_avg
            
        except Exception as e:
            return 0
    
    def create_performance_charts(self, metrics_df, session_data):
        """Create all performance visualization charts"""
        charts = {}
        
        try:
            # Chart 1: Score progression
            if len(metrics_df) > 0:
                fig_score = px.line(
                    metrics_df, 
                    x='date', 
                    y='total_score',
                    title='📈 Total Score Progression',
                    markers=True,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_score.update_layout(
                    height=400,
                    xaxis_title="Session Date",
                    yaxis_title="Total Score",
                    showlegend=False
                )
                charts['score_progression'] = fig_score
            
            # Chart 2: Consistency trend
            if len(metrics_df) > 0:
                fig_consistency = px.line(
                    metrics_df,
                    x='date',
                    y='score_std',
                    title='🎯 Shot Consistency (Lower = Better)',
                    markers=True,
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_consistency.update_layout(
                    height=400,
                    xaxis_title="Session Date",
                    yaxis_title="Standard Deviation",
                    showlegend=False
                )
                charts['consistency'] = fig_consistency
            
            # Chart 3: Series performance for latest session
            if len(session_data) > 0:
                latest_session = session_data['session_id'].iloc[-1]
                latest_data = session_data[session_data['session_id'] == latest_session]
                
                if len(latest_data) > 0:
                    series_avg = latest_data.groupby('series')['score'].mean().reset_index()
                    fig_series = px.bar(
                        series_avg,
                        x='series',
                        y='score',
                        title=f'📊 Average Score by Series (Session {latest_session})',
                        color='score',
                        color_continuous_scale='Viridis'
                    )
                    fig_series.update_layout(
                        height=400,
                        xaxis_title="Series Number",
                        yaxis_title="Average Score"
                    )
                    charts['series_performance'] = fig_series
            
            # Chart 4: Shot distribution
            if len(session_data) > 0:
                fig_distribution = px.histogram(
                    session_data,
                    x='score',
                    title='🎯 Shot Score Distribution',
                    nbins=20,
                    color_discrete_sequence=['#2ca02c']
                )
                fig_distribution.update_layout(
                    height=400,
                    xaxis_title="Score",
                    yaxis_title="Number of Shots"
                )
                charts['distribution'] = fig_distribution
                
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
        
        return charts
    
    def predict_next_session(self, metrics_df):
        """Predict expected performance for next session"""
        if len(metrics_df) < 2:
            return None
            
        try:
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
        except:
            return None
    
    def generate_ai_insights(self, metrics_df, session_data):
        """Generate AI-powered insights using OpenAI"""
        if not client or len(metrics_df) == 0:
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
            - Shots Count: {latest_session['shots_count']}
            
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
        if len(metrics_df) == 0:
            return "No data available for analysis."
        
        latest = metrics_df.iloc[-1]
        
        insights = []
        insights.append("KEY STRENGTHS:")
        if latest['hold_stability'] > 2.0:
            insights.append("• Excellent hold stability - maintain this foundation")
        if latest['trigger_consistency'] > 8.0:
            insights.append("• Good trigger control consistency")
        else:
            insights.append("• Solid baseline performance")
        
        insights.append("\nAREAS FOR IMPROVEMENT:")
        if latest['endurance_drop'] > 0.3:
            insights.append(f"• Endurance: {latest['endurance_drop']:.2f} point drop in later series")
        if latest['score_std'] > 0.4:
            insights.append(f"• Consistency: Shot-to-shot variation needs work (std: {latest['score_std']:.2f})")
        
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
    athlete_id = st.sidebar.text_input("Athlete ID", value="SHOOTER_001")
    sessions_to_generate = st.sidebar.slider("Training Sessions to Generate", 2, 10, 3)
    
    # Generate sample data
    if st.sidebar.button("Generate Sample Data") or 'data' not in st.session_state:
        with st.spinner("Generating realistic training data..."):
            try:
                st.session_state.data = analyzer.generate_sample_data(athlete_id, sessions_to_generate)
                st.session_state.metrics = analyzer.calculate_session_metrics(st.session_state.data)
                st.success("✅ Sample data generated successfully!")
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Or upload your data (CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            st.session_state.data = pd.read_csv(uploaded_file)
            
            # Validate and process the data
            st.session_state.data = analyzer.validate_data(st.session_state.data)
            st.session_state.metrics = analyzer.calculate_session_metrics(st.session_state.data)
            
            st.sidebar.success(f"✅ Data uploaded successfully! {len(st.session_state.data)} shots loaded.")
            
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            st.info("💡 Make sure your CSV has columns: session_id, athlete_id, date, shot_number, score")
    
    if 'data' not in st.session_state:
        st.info("👆 Click 'Generate Sample Data' to start or upload your CSV file")
        return
    
    # Display data overview
    if len(st.session_state.metrics) > 0:
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
            else:
                st.metric("Next Session", "Need more data", delta="")
        
        # Performance Charts Section
        st.subheader("📊 Performance Analysis Charts")
        
        # Generate all charts
        charts = analyzer.create_performance_charts(st.session_state.metrics, st.session_state.data)
        
        # Display charts in a grid layout
        if charts:
            # First row: Score progression and Consistency
            col1, col2 = st.columns(2)
            
            with col1:
                if 'score_progression' in charts:
                    st.plotly_chart(charts['score_progression'], use_container_width=True)
                else:
                    st.info("Score progression chart not available")
            
            with col2:
                if 'consistency' in charts:
                    st.plotly_chart(charts['consistency'], use_container_width=True)
                else:
                    st.info("Consistency chart not available")
            
            # Second row: Series performance and Distribution
            col3, col4 = st.columns(2)
            
            with col3:
                if 'series_performance' in charts:
                    st.plotly_chart(charts['series_performance'], use_container_width=True)
                else:
                    st.info("Series performance chart not available")
            
            with col4:
                if 'distribution' in charts:
                    st.plotly_chart(charts['distribution'], use_container_width=True)
                else:
                    st.info("Score distribution chart not available")
        else:
            st.warning("No charts could be generated. Please check your data.")
        
        # AI Insights Section
        st.subheader("🤖 AI-Powered Training Insights")
        
        if st.button("Generate AI Analysis", type="primary"):
            with st.spinner("Analyzing performance and generating recommendations..."):
                insights = analyzer.generate_ai_insights(st.session_state.metrics, st.session_state.data)
                
                st.info("💡 Training Recommendations")
                st.text_area("Analysis Results", insights, height=300, key="ai_insights")
        
        # Drill Recommendations
        st.subheader("🎯 Recommended Training Drills")
        
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
        
        # Data summary
        st.sidebar.subheader("Data Summary")
        st.sidebar.write(f"Sessions: {len(st.session_state.metrics)}")
        st.sidebar.write(f"Total Shots: {len(st.session_state.data)}")
        st.sidebar.write(f"Athlete: {athlete_id}")
        
    else:
        st.error("No valid data to display. Please check your data format.")

if __name__ == "__main__":
    main()