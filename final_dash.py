
# %%
# %%
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv("summer.csv")

"""
Define functions
"""
# Add additional analysis columns
host_countries = {1896: 'GRE', 1900: 'FRA', 1904: 'USA', 1908: 'GBR', 1912: 'SWE'}
df['IsHostCountry'] = df.apply(lambda row: row['Country'] == host_countries.get(row['Year'], ''), axis=1)

team_sports = ['Football', 'Hockey', 'Water polo', 'Rugby', 'Lacrosse', 'Polo']
df['IsTeamSport'] = df['Sport'].isin(team_sports)

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

# Add Gini to metrics
def compute_gini(sport):
    counts = sport["MedalCount"].values
    gini_index = gini(counts)
    return pd.Series({
        "Gini": gini_index,
        "Countries": len(sport)
    })

#%%
"""
EDA
"""
print(df["Sport"].nunique())
print(df["Country"].nunique())

#%%
"""
Which sports are dominated by certain country vs which sports have the most competitive medalists
"""
# Step 1: Count medals by Sport + Country
medal_counts = df.groupby(["Sport", "Country"])["Medal"].count().reset_index()
medal_counts.rename(columns={"Medal": "MedalCount"}, inplace=True)

# Step 2: Compute Gini index on all Sports
sport_metrics = medal_counts.groupby("Sport").apply(compute_gini).reset_index()

# Step 3: Only retain sports where they're played for more than 10 or equal to years to give enough sample size
sports_dropped = sport_metrics[sport_metrics["Countries"] < 10]["Sport"]
sport_metrics = sport_metrics[sport_metrics["Countries"] >= 10].sort_values(by="Gini")

#%%
"""
Find countries that dominated the sport more than X% of the time
"""
# Medal counts per sport-country
sport_country = df.groupby(["Sport", "Country"])["Medal"].count().reset_index()
sport_country.rename(columns={"Medal": "MedalCount"}, inplace=True)

# Total medals per sport
sport_totals = sport_country.groupby("Sport")["MedalCount"].sum().reset_index()
sport_totals.rename(columns={"MedalCount": "TotalMedals"}, inplace=True)

# Merge back to compute share
sport_country = sport_country.merge(sport_totals, on="Sport")
sport_country["Share"] = sport_country["MedalCount"] / sport_country["TotalMedals"]

threshold = 0.25
dominant_sports = sport_country[sport_country["Share"] > threshold].sort_values("Share", ascending=False)

# Sports that's significantly dominated by one country
significant = dominant_sports[~dominant_sports["Sport"].isin(sports_dropped)]
sport_country_dict = dict(zip(significant["Sport"], significant["Country"]))
sport_share_dict = dict(zip(significant["Sport"], significant["Share"]))
significant_sport = significant["Sport"]

# add indicator for dominated sport
sport_metrics["Dominant"] = sport_metrics["Sport"].isin(significant_sport)
sport_metrics["Dominant Country"] = sport_metrics["Sport"].map(sport_country_dict)
sport_metrics["Dominant Share"] = sport_metrics["Sport"].map(sport_share_dict)

sport_metrics.to_csv("sport_metrics.csv")

#%%
"""
Prepare data for the new chart - Yearly Medal Trend by Top Countries
"""
# Prepare data for yearly medal trend
yearly_medals = df.groupby(['Year', 'Country'])['Medal'].count().reset_index()
yearly_medals.rename(columns={'Medal': 'MedalCount'}, inplace=True)

# Get top 10 countries by total medals
top_countries = yearly_medals.groupby('Country')['MedalCount'].sum().nlargest(10).index.tolist()

# Filter data for top countries
yearly_top_countries = yearly_medals[yearly_medals['Country'].isin(top_countries)]

# Data preprocessing
# Medal counts by year and country
medals_by_year_country = df.groupby(["Year", "Country"]).size().reset_index(name="Medals")

# Top 10 countries by total medals
top10_countries = (medals_by_year_country.groupby("Country")["Medals"]
                   .sum()
                   .nlargest(10)
                   .index)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Host country mapping dictionary
host_countries = {
    1896: "GRE",  # Athens (Greece)
    1900: "FRA",  # Paris (France)
    1904: "USA",  # St. Louis (United States)
    1908: "GBR",  # London (United Kingdom)
    1912: "SWE",  # Stockholm (Sweden)
    1920: "BEL",  # Antwerp (Belgium)
    1924: "FRA",  # Paris (France)
    1928: "NED",  # Amsterdam (Netherlands)
    1932: "USA",  # Los Angeles (United States)
    1936: "GER",  # Berlin (Germany)
    1948: "GBR",  # London (United Kingdom)
    1952: "FIN",  # Helsinki (Finland)
    1956: "AUS",  # Melbourne (Australia)
    1960: "ITA",  # Rome (Italy)
    1964: "JPN",  # Tokyo (Japan)
    1968: "MEX",  # Mexico City (Mexico)
    1972: "GER",  # Munich (Germany)
    1976: "CAN",  # Montreal (Canada)
    1980: "RUS",  # Moscow (Russia)
    1984: "USA",  # Los Angeles (United States)
    1988: "KOR",  # Seoul (South Korea)
    1992: "ESP",  # Barcelona (Spain)
    1996: "USA",  # Atlanta (United States)
    2000: "AUS",  # Sydney (Australia)
    2004: "GRE",  # Athens (Greece)
    2008: "CHN",  # Beijing (China)
    2012: "GBR"   # London (United Kingdom)
}

def calculate_host_medal_stats(df, host_countries):
    """
    Calculate medal statistics for host countries in each Olympic Games
    
    Parameters:
    df: DataFrame containing Olympic medal data
    host_countries: Host country mapping dictionary
    
    Returns:
    DataFrame: Table with year, host country, host year medals, previous games medals, 
               next games medals, and mean of adjacent games
    """
    # Get all unique years and sort them
    all_years = sorted(df['Year'].unique())
    
    # Calculate total medals for each country each year
    medal_counts = df.groupby(['Year', 'Country']).size().reset_index(name='Medal_Count')
    
    results = []
    
    for year, host_country in host_countries.items():
        if year not in all_years:
            continue
            
        # Get host year medal count
        host_medals = medal_counts[(medal_counts['Year'] == year) & 
                                  (medal_counts['Country'] == host_country)]
        host_medal_count = host_medals['Medal_Count'].values[0] if not host_medals.empty else 0
        
        # Find adjacent game years
        year_idx = all_years.index(year)
        prev_year = all_years[year_idx - 1] if year_idx > 0 else None
        next_year = all_years[year_idx + 1] if year_idx < len(all_years) - 1 else None
        
        # Get previous games medal count
        prev_medal_count = None
        if prev_year:
            prev_medals = medal_counts[(medal_counts['Year'] == prev_year) & 
                                      (medal_counts['Country'] == host_country)]
            prev_medal_count = prev_medals['Medal_Count'].values[0] if not prev_medals.empty else 0
        
        # Get next games medal count
        next_medal_count = None
        if next_year:
            next_medals = medal_counts[(medal_counts['Year'] == next_year) & 
                                      (medal_counts['Country'] == host_country)]
            next_medal_count = next_medals['Medal_Count'].values[0] if not next_medals.empty else 0
        
        # Calculate mean of adjacent games (handle edge cases)
        if prev_medal_count is not None and next_medal_count is not None:
            mean_medals = (prev_medal_count + next_medal_count) / 2
        elif prev_medal_count is not None:
            mean_medals = prev_medal_count
        elif next_medal_count is not None:
            mean_medals = next_medal_count
        else:
            mean_medals = 0
        
        results.append({
            'Year': year,
            'Host_Country': host_country,
            'Host_Year_Medals': host_medal_count,
            'Previous_Games_Medals': prev_medal_count if prev_medal_count is not None else 'N/A',
            'Next_Games_Medals': next_medal_count if next_medal_count is not None else 'N/A',
            'Mean_Adjacent_Games': mean_medals
        })
    
    return pd.DataFrame(results)

# Calculate host country medal statistics
result_df = calculate_host_medal_stats(df, host_countries)

# Display results
print("Host Country Medal Statistics:")
print("=" * 80)
print(result_df.to_string(index=False))

# Data preprocessing for gender analysis
df['Decade'] = (df['Year'] // 10) * 10
gender_by_year = df.groupby(['Year', 'Gender']).size().unstack(fill_value=0)
gender_by_year['Total'] = gender_by_year.sum(axis=1)
gender_by_year['Women_Ratio'] = gender_by_year['Women'] / gender_by_year['Total']
gender_by_year['Men_Ratio'] = gender_by_year['Men'] / gender_by_year['Total']
gender_by_year['Gender_Ratio'] = gender_by_year['Women'] / gender_by_year['Men']

# First year women won medals by sport
first_women_year = df[df['Gender'] == 'Women'].groupby('Sport')['Year'].min().sort_values()

# Women's participation by decade
decade_stats = df.groupby('Decade').apply(
    lambda x: (x['Gender'] == 'Women').sum() / len(x)
).reset_index()
decade_stats.columns = ['Decade', 'Women_Ratio']

# Gender distribution by discipline
gender_discipline_pivot = pd.pivot_table(df, values='Medal', index='Discipline',
                                        columns='Gender', aggfunc='count', fill_value=0)
gender_discipline_pivot['Total'] = gender_discipline_pivot.sum(axis=1)
gender_discipline_pivot['Women_Percentage'] = (gender_discipline_pivot['Women'] / gender_discipline_pivot['Total'] * 100).round(1)

# Heatmap data by decade and sport
gender_sport_decade = df[df['Gender'] == 'Women'].pivot_table(
    values='Medal', index='Sport', columns='Decade', aggfunc='count', fill_value=0
)/df.pivot_table(values='Medal', index='Sport', columns='Decade', aggfunc='count', fill_value=0)

# Recent gender ratio data
recent_years = df[df['Year'] >= 2000] if df['Year'].max() >= 2000 else df
sport_gender_ratio = recent_years.groupby('Sport').apply(
    lambda x: (x['Gender'] == 'Women').sum() / len(x)
).sort_values(ascending=False)

"""
Dash App with Callback
"""
# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Olympics Deep Analysis Dashboard"
valid_countries = sorted([c for c in df['Country'].unique() if pd.notna(c)])
valid_sports = sorted([s for s in df['Sport'].unique() if pd.notna(s)])

host_medal_data = []
for year, host_country in host_countries.items():
    if year in df['Year'].unique():
        host_medals_count = len(df[(df['Year'] == year) & (df['Country'] == host_country)])
        
        # 获取相邻年份（前后一届）
        all_years = sorted(df['Year'].unique())
        year_index = all_years.index(year)
        
        prev_year = all_years[year_index - 1] if year_index > 0 else None
        next_year = all_years[year_index + 1] if year_index < len(all_years) - 1 else None
        
        neighbor_medals = []
        if prev_year:
            prev_medals = len(df[(df['Year'] == prev_year) & (df['Country'] == host_country)])
            neighbor_medals.append(prev_medals)
        if next_year:
            next_medals = len(df[(df['Year'] == next_year) & (df['Country'] == host_country)])
            neighbor_medals.append(next_medals)
        
        neighbor_avg = np.mean(neighbor_medals) if neighbor_medals else np.nan
        
        host_medal_data.append({
            'Year': year,
            'Host_Country': host_country,
            'Medals': host_medals_count,
            'Neighbor_Avg': neighbor_avg
        })

host_medals = pd.DataFrame(host_medal_data)


fig = go.Figure()

# Host medals (blue bars)
fig.add_trace(go.Bar(
    x=host_medals["Year"],
    y=host_medals["Medals"],
    name="Host Medals",
    marker_color="steelblue"
))

# Neighbor averages (orange if available, gray if missing)
colors = ["orange" if not pd.isna(v) else "gray" for v in host_medals["Neighbor_Avg"]]
texts = [f"{v:.0f}" if not pd.isna(v) else "No Data" for v in host_medals["Neighbor_Avg"]]

fig.add_trace(go.Bar(
    x=host_medals["Year"],
    y=host_medals["Neighbor_Avg"].fillna(0),
    name="Neighbor Avg",
    marker_color=colors,
    text=texts,
    textposition="outside"
))

fig.update_layout(
    barmode="group",
    title="Host Country Medals vs Neighbor Olympics Average",
    xaxis_title="Year",
    yaxis_title="Medal Count",
    bargap=0.2,
    template="plotly_white"
)
app.layout = html.Div([
    html.H1("🏅 Olympics Deep Data Analysis", style={'textAlign': 'center', 'color': '#1f77b4'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Year Range:"),
            dcc.RangeSlider(
                id='year-slider',
                min=int(df['Year'].min()),
                max=int(df['Year'].max()),
                step=1,
                marks={str(year): str(year) for year in range(int(df['Year'].min()), int(df['Year'].max())+1, 20)},
                value=[int(df['Year'].min()), int(df['Year'].max())]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Sport:"),
            dcc.Dropdown(
                id='sport-dropdown',
                options=[{'label': 'All Sports', 'value': 'all'}] + 
                        [{'label': sport, 'value': sport} for sport in sorted(df['Sport'].unique())],
                value='all',
                clearable=False
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    dcc.Tabs([
        # Tab 1: Overview
        dcc.Tab(label='Overview', children=[
            html.Div([
                dcc.RangeSlider(
                    id='overview-year-slider',
                    min=df['Year'].min(),
                    max=df['Year'].max(),
                    step=1,
                    marks={year: str(year) for year in range(df['Year'].min(), df['Year'].max()+1, 20)},
                    value=[1950, df['Year'].max()]
                ),
                
                html.Div([
                    dcc.Graph(id='overview-medal-pie'),
                    dcc.Graph(id='overview-gender-bar')
                ], style={'display': 'flex'}),
                
                html.Div([
                    dcc.Graph(id='overview-year-trend'),
                    dcc.Graph(id='overview-top-countries')
                ], style={'display': 'flex'})
            ])
        ]),
        
        # Tab 2: Country Comparison
        dcc.Tab(label='Country Comparison', children=[
            html.Div([
                dcc.Dropdown(
                    id='country-comparison-dropdown',
                    options=[{'label': c, 'value': c} for c in valid_countries],
                    value=['USA', 'GBR', 'GER', 'CHN', 'RUS'],
                    multi=True
                ),
                
                dcc.Graph(id='country-medal-trend'),
                dcc.Graph(id='country-medal-composition')
            ])
        ]),
        
        
        
        
        # Tab 5: Time Trend Analysis
        dcc.Tab(label='Time Trends', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='women-ratio-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='gender-ratio-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
            ]),
            
            html.Div([
                dcc.Graph(id='stacked-gender-chart')
            ], style={'margin-top': '20px'})
        ]),
        
        # Tab 6: Sport Gender Analysis
        dcc.Tab(label='Sport Gender Analysis', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='first-women-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='sport-participation-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
            ]),
            
            html.Div([
                dcc.Graph(id='heatmap-chart')
            ], style={'margin-top': '20px'})
        ]),
        
        # Tab 7: Detailed Statistics
        dcc.Tab(label='Detailed Statistics', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='decade-progress-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='discipline-gender-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
            ]),
            
            html.Div([
                html.H4("Key Statistics"),
                html.Div(id='key-statistics')
            ], style={'margin-top': '20px', 'padding': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '10px'})
        ]),
        # Tab 3: Medal concentration
        dcc.Tab(label='Medal concentration', children=[
            html.Div([
                dcc.Dropdown(
                    id='concentration-country-dropdown',
                    options=[{'label': country, 'value': country} for country in top_countries],
                    value=top_countries[:5],
                    multi=True
                ),
                html.Div([
                    dcc.Graph(id='gini-chart')
                ], style={'width': '96%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='yearly-trend-chart')
                ], style={'width': '96%', 'display': 'inline-block', 'margin-left': '4%'})
            ]),
        ]),   
        
        # Tab 4: Host advantage
        dcc.Tab(label='Host advantage', children=[
            html.Div([
                html.H1("Olympic Host Performance"),
    dcc.Graph(figure=fig)
            ])
        ])
    ])
])

# Callback functions
@app.callback(
    [Output('overview-medal-pie', 'figure'),
     Output('overview-gender-bar', 'figure'),
     Output('overview-year-trend', 'figure'),
     Output('overview-top-countries', 'figure')],
    [Input('overview-year-slider', 'value')]
)
def update_overview(selected_years):
    filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    
    # Medal distribution pie chart
    medal_fig = px.pie(filtered_df, names='Medal', title='Medal Distribution')
    
    # Gender distribution
    gender_counts = filtered_df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    gender_fig = px.bar(gender_counts, 
                       x='Gender', y='Count', title='Gender Distribution',
                       labels={'Gender': 'Gender', 'Count': 'Medal Count'})
    
    # Yearly trend
    year_trend = filtered_df.groupby('Year').size().reset_index(name='Count')
    trend_fig = px.line(year_trend, x='Year', y='Count', title='Yearly Medal Trend')
    
    # Top countries ranking
    top_countries = filtered_df['Country'].value_counts().head(15).reset_index()
    top_countries.columns = ['Country', 'Count']
    country_fig = px.bar(top_countries, x='Country', y='Count', title='Top 15 Countries',
                        labels={'Country': 'Country', 'Count': 'Medal Count'})
    
    return medal_fig, gender_fig, trend_fig, country_fig

@app.callback(
    [Output('country-medal-trend', 'figure'),
     Output('country-medal-composition', 'figure')],
    [Input('country-comparison-dropdown', 'value')]
)
def update_country_comparison(selected_countries):
    if not selected_countries:
        return px.scatter(title='No data'), px.scatter(title='No data')
    
    filtered_df = df[df['Country'].isin(selected_countries)]
    
    # Country medal trend
    country_trend = filtered_df.groupby(['Year', 'Country']).size().reset_index(name='Count')
    trend_fig = px.line(country_trend, x='Year', y='Count', color='Country', 
                       title='Country Medal Trends Over Time')
    
    # Country medal composition
    medal_composition = filtered_df.groupby(['Country', 'Medal']).size().reset_index(name='Count')
    composition_fig = px.bar(medal_composition, x='Country', y='Count', color='Medal',
                            title='Medal Composition by Country', barmode='group')
    
    return trend_fig, composition_fig

@app.callback(
    [Output('gini-chart', 'figure'),
     Output('yearly-trend-chart', 'figure')],
    [Input('concentration-country-dropdown', 'value')]
)
def update_concentration_charts(selected_countries):
    # Update Gini chart
    sport_metrics_sorted = sport_metrics.sort_values("Gini").reset_index(drop=True)
    
    fig_gini = px.scatter(
        sport_metrics_sorted, x="Sport", y="Gini",
        color="Dominant", 
        symbol="Dominant",
        size="Countries",
        size_max=20,
        labels={"Sport": "Olympic Sports", "Gini": "Gini Index"},
        title="Olympic Sports Dominance by Gini Index"
    )
    
    # Add annotations for dominant sports
    annotations = []
    for idx, row in sport_metrics_sorted.iterrows():
        if row["Dominant"] and pd.notnull(row["Dominant Country"]):
            share = row["Dominant Share"]
            share_text = f"{share*100:.0f}%" if pd.notnull(share) else "N/A"
            country = sport_country_dict.get(row["Sport"], "Unknown")
            
            annotations.append(
                dict(
                    x=row["Sport"],
                    y=row["Gini"],
                    xref="x",
                    yref="y",
                    text=f"{country}: {share_text}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    font=dict(size=12, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            )
    
    fig_gini.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig_gini.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=sport_metrics_sorted["Sport"]),
        legend_title_text="Dominated by One Country (>25% Medals)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        annotations=annotations,
        height=500
    )
    
    # Update Yearly Trend chart
    if selected_countries:
        filtered_data = yearly_top_countries[yearly_top_countries['Country'].isin(selected_countries)]
    else:
        filtered_data = yearly_top_countries
    
    fig_trend = px.line(
        filtered_data,
        x='Year',
        y='MedalCount',
        color='Country',
        title='Yearly Medal Trends by Country',
        labels={'MedalCount': 'Number of Medals', 'Year': 'Year'},
        height=500
    )
    
    fig_trend.update_layout(
        xaxis=dict(tickmode='linear', dtick=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig_gini, fig_trend


@app.callback(
    [Output('women-ratio-chart', 'figure'),
     Output('gender-ratio-chart', 'figure'),
     Output('stacked-gender-chart', 'figure'),
     Output('first-women-chart', 'figure'),
     Output('sport-participation-chart', 'figure'),
     Output('heatmap-chart', 'figure'),
     Output('decade-progress-chart', 'figure'),
     Output('discipline-gender-chart', 'figure'),
     Output('key-statistics', 'children')],
    [Input('year-slider', 'value'),
     Input('sport-dropdown', 'value')]
)
def update_all_charts(selected_years, selected_sport):
    # Filter data
    filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    if selected_sport != 'all':
        filtered_df = filtered_df[filtered_df['Sport'] == selected_sport]
    
    # Recalculate filtered data for gender analysis
    filtered_gender_by_year = filtered_df.groupby(['Year', 'Gender']).size().unstack(fill_value=0)
    if 'Women' not in filtered_gender_by_year.columns:
        filtered_gender_by_year['Women'] = 0
    if 'Men' not in filtered_gender_by_year.columns:
        filtered_gender_by_year['Men'] = 0
        
    filtered_gender_by_year['Total'] = filtered_gender_by_year.sum(axis=1)
    filtered_gender_by_year['Women_Ratio'] = filtered_gender_by_year['Women'] / filtered_gender_by_year['Total']
    filtered_gender_by_year['Gender_Ratio'] = filtered_gender_by_year['Women'] / filtered_gender_by_year['Men']
    
    # 1. Women's Ratio Trend Chart
    fig1 = px.line(
        filtered_gender_by_year.reset_index(),
        x='Year', y='Women_Ratio',
        title='Trend of Female Medal Winners Ratio',
        labels={'Women_Ratio': 'Women Ratio', 'Year': 'Year'}
    )
    fig1.update_traces(line=dict(width=3), marker=dict(size=6))
    fig1.update_layout(yaxis_tickformat='.0%')
    
    # 2. Gender Ratio Chart
    fig2_data = filtered_gender_by_year.reset_index()
    fig2 = px.line(
        fig2_data,
        x='Year', y='Gender_Ratio',
        title='Gender Ratio Trend (Women:Men)',
        labels={'Gender_Ratio': 'Gender Ratio', 'Year': 'Year'}
    )
    fig2.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Equality Line")
    fig2.update_traces(line=dict(width=3, color='green'), marker=dict(size=6, color='green'))
    
    # 3. Stacked Area Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=filtered_gender_by_year.index, y=filtered_gender_by_year['Men'],
        mode='lines', name='Men', stackgroup='one',
        line=dict(width=0.5, color='blue'), fill='tonexty'
    ))
    fig3.add_trace(go.Scatter(
        x=filtered_gender_by_year.index, y=filtered_gender_by_year['Women'],
        mode='lines', name='Women', stackgroup='one',
        line=dict(width=0.5, color='pink'), fill='tonexty'
    ))
    fig3.update_layout(title='Medal Winners Distribution by Gender', 
                      xaxis_title='Year', yaxis_title='Number of Medals')
    
    # 4. First Women Winners Chart - handle empty data
    filtered_first_women = first_women_year[first_women_year.index.isin(filtered_df['Sport'].unique())]
    if len(filtered_first_women) > 0:
        fig4 = px.bar(
            x=filtered_first_women.index, y=filtered_first_women.values,
            title='First Year Women Won Medals by Sport',
            labels={'x': 'Sport', 'y': 'Year'}
        )
    else:
        fig4 = px.bar(title='No data available for selected filters')
    fig4.update_layout(xaxis_tickangle=45)
    
    # 5. Sport Participation Chart - handle empty data
    available_sports = [sport for sport in filtered_df['Sport'].unique() if sport in sport_gender_ratio.index]
    if available_sports:
        filtered_sport_ratio = sport_gender_ratio[available_sports]
        fig5 = px.bar(
            x=filtered_sport_ratio.index, y=filtered_sport_ratio.values,
            title='Sports with Highest Female Participation (Recent Years)',
            labels={'x': 'Sport', 'y': 'Women Ratio'}
        )
    else:
        fig5 = px.bar(title='No data available for selected filters')
    fig5.update_layout(xaxis_tickangle=45, yaxis_tickformat='.0%')
    
    # 6. Heatmap Chart - handle empty data
    available_sports_heatmap = [sport for sport in filtered_df['Sport'].unique() if sport in gender_sport_decade.index]
    if available_sports_heatmap:
        fig6 = px.imshow(
            gender_sport_decade.loc[available_sports_heatmap],
            aspect='auto',
            title='Female Medal Winners Heatmap by Sport and Decade',
            labels={'x': 'Decade', 'y': 'Sport', 'color': 'Medal Count'}
        )
    else:
        fig6 = px.imshow(np.zeros((1, 1)), title='No data available for selected filters')
    
    # 7. Decade Progress Chart - use filtered data
    filtered_decade_stats = filtered_df.groupby('Decade').apply(
        lambda x: (x['Gender'] == 'Women').sum() / len(x) if len(x) > 0 else 0
    ).reset_index()
    filtered_decade_stats.columns = ['Decade', 'Women_Ratio']
    
    fig7 = px.line(
        filtered_decade_stats, x='Decade', y='Women_Ratio',
        title='Progress in Gender Equality (by Decade)',
        labels={'Women_Ratio': 'Women Ratio', 'Decade': 'Decade'}
    )
    fig7.update_traces(line=dict(width=3), marker=dict(size=8))
    fig7.update_layout(yaxis_tickformat='.0%')
    
    # 8. Discipline Gender Distribution Chart - use filtered data
    filtered_gender_discipline = filtered_df.pivot_table(
        values='Medal', index='Discipline', columns='Gender', 
        aggfunc='count', fill_value=0
    )
    if 'Women' not in filtered_gender_discipline.columns:
        filtered_gender_discipline['Women'] = 0
    if 'Men' not in filtered_gender_discipline.columns:
        filtered_gender_discipline['Men'] = 0
        
    filtered_gender_discipline['Total'] = filtered_gender_discipline.sum(axis=1)
    filtered_gender_discipline['Women_Percentage'] = (filtered_gender_discipline['Women'] / filtered_gender_discipline['Total'] * 100).round(1)
    
    fig8_data = filtered_gender_discipline.reset_index().sort_values('Women_Percentage', ascending=False).head(20)
    fig8 = px.bar(
        fig8_data,
        x='Discipline', y='Women_Percentage',
        title='Female Participation Percentage by Discipline',
        labels={'Women_Percentage': 'Women Percentage (%)', 'Discipline': 'Discipline'}
    )
    fig8.update_layout(xaxis_tickangle=45)
    
    # 9. Key Statistics
    total_medals = len(filtered_df)
    women_medals = len(filtered_df[filtered_df['Gender'] == 'Women']) if 'Women' in filtered_df['Gender'].values else 0
    women_ratio = women_medals / total_medals if total_medals > 0 else 0
    
    women_df = filtered_df[filtered_df['Gender'] == 'Women']
    first_women_year_val = women_df['Year'].min() if not women_df.empty else 'No data'
    
    stats_text = [
        html.P(f"Total Medals: {total_medals:,}"),
        html.P(f"Women Medals: {women_medals:,} ({women_ratio:.1%})"),
        html.P(f"Men Medals: {len(filtered_df[filtered_df['Gender'] == 'Men']) if 'Men' in filtered_df['Gender'].values else 0:,}"),
        html.P(f"First Year Women Won: {first_women_year_val}"),
        html.P(f"Sports Included: {filtered_df['Sport'].nunique()}"),
        html.P(f"Countries Included: {filtered_df['Country'].nunique()}")
    ]
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, stats_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
    print("Dash app running at http://127.0.0.1:8050/")

df['Is_Host'] = df.apply(lambda row: row['Country'] == host_countries.get(row['Year'], None), axis=1)

medal_counts = df[df['Medal'].notnull()].groupby(['Year', 'Country', 'Is_Host']).size().reset_index(name='Medal_Count')

medal_counts.head()

from scipy.stats import ttest_ind
df = medal_counts.copy()
host_medals = df[df['Is_Host'] == True]['Medal_Count']
non_host_medals = df[df['Is_Host'] == False]['Medal_Count']

# Perform independent t-test
t_stat, p_val = ttest_ind(host_medals, non_host_medals, equal_var=False)  # Welch's t-test is safer

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Optional: interpret
if p_val < 0.05:
    print("Statistically significant difference in medal counts between host and non-host countries.")
else:
    print("No statistically significant difference.")


# %%
# %%
