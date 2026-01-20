import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="RoadSafe Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_data(df):
    # Cleaning up missing values and creating time-based features
    df = df.dropna(axis=1, thresh=len(df)*0.5)
    df = df.fillna(df.select_dtypes(include=['number']).median())
    df = df.fillna(df.select_dtypes(include=['object']).mode().iloc[0])
    df = df.drop_duplicates()
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')
    
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    
    def road_surface(weather):
        weather = str(weather).lower()
        if 'rain' in weather:
            return 'Wet'
        elif 'snow' in weather or 'ice' in weather:
            return 'Snow/Ice'
        elif 'fog' in weather:
            return 'Foggy'
        else:
            return 'Dry'
    
    df['Road_Surface'] = df['Weather_Condition'].astype(str).apply(road_surface)
    
    def simplify_weather(weather):
        weather = str(weather).lower()
        if 'clear' in weather or 'fair' in weather:
            return 'Clear'
        elif 'cloud' in weather or 'overcast' in weather:
            return 'Cloudy'
        elif 'rain' in weather or 'drizzle' in weather:
            return 'Rain'
        elif 'snow' in weather:
            return 'Snow'
        elif 'fog' in weather or 'mist' in weather:
            return 'Fog'
        elif 'storm' in weather or 'thunder' in weather:
            return 'Storm'
        else:
            return 'Other'
    
    df['Weather_Simple'] = df['Weather_Condition'].astype(str).apply(simplify_weather)
    
    df["Traffic_Congestion_Score"] = (
        df["Traffic_Signal"].astype(int) +
        df["Junction"].astype(int) +
        df["Crossing"].astype(int) +
        df["Stop"].astype(int)
    )
    
    return df

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🚗 RoadSafe Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    data_path = "/Users/chetangavali/Downloads/US_Accidents_March23.csv"
    
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading data..."):
            df = load_and_preprocess_data(data_path)
            st.session_state.data_loaded = True
            st.session_state.df = df
    else:
        df = st.session_state.df
    
    if df is None or len(df) == 0:
        st.error("Unable to load data. Please check the file path.")
        return
    
    # State filter
    st.sidebar.title("Filters")
    if 'State' in df.columns:
        states = sorted(df['State'].unique().tolist())
        states.insert(0, "All States")
        
        selected_state = st.sidebar.selectbox("Select State", options=states, index=0)
        
        if selected_state != "All States":
            filtered_df = df[df['State'] == selected_state].copy()
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()
    
    st.sidebar.markdown(f"**Records:** {len(filtered_df):,}")
    
    df = filtered_df
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "⏰ Time Analysis", 
        "🌦️ Weather Analysis",
        "🛣️ Road Features",
        "📍 Geographic Analysis"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accidents", f"{len(df):,}")
        
        with col2:
            if 'Severity' in df.columns:
                avg_severity = df['Severity'].mean()
                st.metric("Average Severity", f"{avg_severity:.2f}")
        
        with col3:
            if 'State' in df.columns:
                unique_states = df['State'].nunique()
                st.metric("States", unique_states)
        
        with col4:
            if 'City' in df.columns:
                unique_cities = df['City'].nunique()
                st.metric("Cities", unique_cities)
        
        st.markdown("---")
        
        st.subheader("Distribution of Accident Severity Levels")
        if 'Severity' in df.columns:
            severity_counts = df['Severity'].value_counts().sort_index()
            fig = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                labels={'x': 'Severity Level', 'y': 'Number of Accidents'},
                title='Distribution of Accident Severity Levels',
                color=severity_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            severity_pct = df['Severity'].value_counts(normalize=True) * 100
            st.dataframe(severity_pct.round(2).to_frame('Percentage (%)'))
    
    with tab2:
        st.header("Time-based Accident Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accident Frequency by Hour of Day")
            if 'Hour' in df.columns:
                hour_counts = df['Hour'].value_counts().sort_index()
                fig = px.bar(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    labels={'x': 'Hour of Day', 'y': 'Number of Accidents'},
                    title='Accident Frequency by Hour of Day',
                    color=hour_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Accident Frequency by Day of Week")
            if 'Weekday' in df.columns:
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_counts = df['Weekday'].value_counts()
                weekday_counts = weekday_counts.reindex([d for d in weekday_order if d in weekday_counts.index])
                fig = px.bar(
                    x=weekday_counts.index,
                    y=weekday_counts.values,
                    labels={'x': 'Day of Week', 'y': 'Number of Accidents'},
                    title='Accident Frequency by Day of Week',
                    color=weekday_counts.values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Accident Frequency by Month")
        if 'Month' in df.columns:
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_counts = df['Month'].value_counts()
            month_counts = month_counts.reindex([m for m in month_order if m in month_counts.index])
            fig = px.bar(
                x=month_counts.index,
                y=month_counts.values,
                labels={'x': 'Month', 'y': 'Number of Accidents'},
                title='Accident Frequency by Month',
                color=month_counts.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Weather-based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Weather Conditions During Accidents")
            if 'Weather_Condition' in df.columns:
                weather_counts = df['Weather_Condition'].value_counts().head(10)
                fig = px.bar(
                    x=weather_counts.values,
                    y=weather_counts.index,
                    orientation='h',
                    labels={'x': 'Number of Accidents', 'y': 'Weather Condition'},
                    title='Top 10 Weather Conditions During Accidents',
                    color=weather_counts.values,
                    color_continuous_scale='Purples'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Severity vs Visibility")
            if 'Severity' in df.columns and 'Visibility(mi)' in df.columns:
                fig = px.box(
                    df,
                    x='Severity',
                    y='Visibility(mi)',
                    title='Severity vs Visibility',
                    color='Severity',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        if all(col in df.columns for col in ['Severity', 'Visibility(mi)', 'Distance(mi)', 'Wind_Speed(mph)', 'Humidity(%)']):
            corr_cols = ['Severity', 'Visibility(mi)', 'Distance(mi)', 'Wind_Speed(mph)', 'Humidity(%)']
            corr = df[corr_cols].corr()
            fig = px.imshow(
                corr,
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_cols,
                y=corr_cols,
                color_continuous_scale='RdBu',
                aspect="auto",
                title='Correlation Heatmap'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Accident Severity vs Weather Condition")
        if 'Weather_Simple' in df.columns and 'Severity' in df.columns:
            severity_weather_ct = pd.crosstab(df["Weather_Simple"], df["Severity"])
            fig = px.imshow(
                severity_weather_ct.values,
                labels=dict(x="Severity Level", y="Weather Condition", color="Count"),
                x=severity_weather_ct.columns.tolist(),
                y=severity_weather_ct.index.tolist(),
                color_continuous_scale='YlOrRd',
                aspect="auto",
                title='Heatmap of Accident Severity vs Weather Condition'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Average Accident Severity by Road Surface Condition")
        if 'Road_Surface' in df.columns and 'Severity' in df.columns:
            road_surface_severity = df.groupby('Road_Surface')['Severity'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=road_surface_severity.index,
                y=road_surface_severity.values,
                labels={'x': 'Road Surface Condition', 'y': 'Average Severity'},
                title='Average Accident Severity by Road Surface Condition',
                color=road_surface_severity.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Road Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accidents by Road Features")
            road_features = [
                'Junction', 'Crossing', 'Stop',
                'Traffic_Signal', 'Roundabout',
                'Railway', 'No_Exit'
            ]
            
            road_counts = {}
            for feature in road_features:
                if feature in df.columns:
                    road_counts[feature] = df[feature].sum()
            
            if road_counts:
                road_counts_df = pd.DataFrame(list(road_counts.items()), columns=['Road Feature', 'Count'])
                fig = px.bar(
                    road_counts_df,
                    x='Road Feature',
                    y='Count',
                    labels={'x': 'Road Feature', 'y': 'Number of Accidents'},
                    title='Accidents by Road Features',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Accident Severity vs Traffic Congestion")
            if 'Traffic_Congestion_Score' in df.columns and 'Severity' in df.columns:
                fig = px.box(
                    df,
                    x='Severity',
                    y='Traffic_Congestion_Score',
                    title='Accident Severity vs Traffic Congestion',
                    color='Severity',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Multivariate Relationships")
        if all(col in df.columns for col in ['Severity', 'Visibility(mi)', 'Distance(mi)']):
            pair_cols = ['Severity', 'Visibility(mi)', 'Distance(mi)']
            sample_size = min(3000, len(df))
            if len(df) > sample_size:
                sample_df = df[pair_cols].sample(n=sample_size, random_state=42)
            else:
                sample_df = df[pair_cols]
            fig = px.scatter_matrix(
                sample_df,
                dimensions=pair_cols,
                title='Pair Plot: Multivariate Relationships',
                color='Severity' if 'Severity' in pair_cols else None,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 5 Accident-Prone States")
            if 'State' in df.columns:
                top_states = df['State'].value_counts().head(5)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    labels={'x': 'State', 'y': 'Number of Accidents'},
                    title='Top 5 Accident-Prone States',
                    color=top_states.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 5 Accident-Prone Cities")
            if 'City' in df.columns:
                top_cities = df['City'].value_counts().head(5)
                fig = px.bar(
                    x=top_cities.index,
                    y=top_cities.values,
                    labels={'x': 'City', 'y': 'Number of Accidents'},
                    title='Top 5 Accident-Prone Cities',
                    color=top_cities.values,
                    color_continuous_scale='Blues'
                )
                fig.update_xaxes(tickangle=-30)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Accident Hotspots (Geographic Distribution)")
        if 'Start_Lng' in df.columns and 'Start_Lat' in df.columns:
            sample_size = min(5000, len(df))
            if len(df) > sample_size:
                sample_df = df.sample(n=sample_size, random_state=42)
            else:
                sample_df = df
            
            fig = px.scatter(
                sample_df,
                x='Start_Lng',
                y='Start_Lat',
                title='Accident Hotspots (Latitude vs Longitude)',
                opacity=0.3,
                labels={'Start_Lng': 'Longitude', 'Start_Lat': 'Latitude'},
                color='Severity' if 'Severity' in sample_df.columns else None,
                color_continuous_scale='Reds',
                size_max=3
            )
            fig.update_traces(marker=dict(size=3))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Accident Density Map")
        if 'Start_Lng' in df.columns and 'Start_Lat' in df.columns:
            sample_size = min(20000, len(df))
            if len(df) > sample_size:
                sample_df = df.sample(n=sample_size, random_state=42)
            else:
                sample_df = df
            
            fig = px.density_heatmap(
                sample_df,
                x='Start_Lng',
                y='Start_Lat',
                title='Accident Density Map',
                color_continuous_scale='Reds',
                nbinsx=40,
                nbinsy=40
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
