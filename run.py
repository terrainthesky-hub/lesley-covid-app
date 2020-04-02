import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

df = pd.read_csv('https://raw.githubusercontent.com/terrainthesky-hub/Covid_App/master/COVID-19%20Cases%20(1).csv')

df['Date'] = pd.to_datetime(df['Date']) 
df['Date_year'] = df['Date'].dt.year
df['Date_month'] = df['Date'].dt.month
df['Date_day'] = df['Date'].dt.day
df['Latest_Date'] = pd.to_datetime(df['Date'])
df['Latest_Date_year'] = df['Latest_Date'].dt.year
df['Latest_Date_month'] = df['Latest_Date'].dt.month
df['Latest_Date_day'] = df['Latest_Date'].dt.day
df['Province_State'] = df['Province_State'].replace({np.NaN: 'Missing'})
df = df.drop(columns=['Latest_Date', 'Admin2', 'Combined_Key', 'FIPS'])
df = df.dropna(axis=0)


train = df[df['Date'] < '2020-03-20']
test = df[df['Date'] >= '2020-03-20']

target = 'Cases'
#Dropping data leakage of sales from other countries



X_train = train.drop(columns=['Cases', 'Date'])
y_train = train[target]
X_test = test.drop(columns=['Cases', 'Date'])
y_test = test[target]

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor


pipeline = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    AdaBoostRegressor(n_estimators=200, random_state=42)
    
)


pipeline.fit(X_train, y_train)

y_test = y_test.fillna(y_test.mean())


y_pred = pipeline.predict(X_test)
r2_score(y_test, y_pred)

test['predicted_cases'] = pd.DataFrame(y_pred)
test_california = test[test['Province_State'].str.contains('California') & (test['Case_Type'].str.contains('Confirmed'))]

import plotly.graph_objects as go

fig1 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_california['Date'], y=test_california['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_california['Date'], y=test_california['Cases'])
          ])
# Change the bar mode
fig1.update_layout(barmode='group')

test_new_york = test[test['Province_State'].str.contains('New York') & (test['Case_Type'].str.contains('Confirmed'))]


fig2 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_new_york['Date'], y=test_new_york['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_new_york['Date'], y=test_new_york['Cases'])
          ])
# Change the bar mode
fig2.update_layout(barmode='group')

test_florida = test[test['Province_State'].str.contains('Florida') & (test['Case_Type'].str.contains('Confirmed'))]
fig3 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_florida['Date'], y=test_florida['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_florida['Date'], y=test_florida['Cases'])
          ])
# Change the bar mode
fig3.update_layout(barmode='group')

test_tenessee = test[test['Province_State'].str.contains('Tennessee') & (test['Case_Type'].str.contains('Confirmed'))]
fig4 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_tenessee['Date'], y=test_tenessee['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_tenessee['Date'], y=test_tenessee['Cases'])
          ])
fig4.update_layout(barmode='group')

test_colorado = test[test['Province_State'].str.contains('Colorado') & (test['Case_Type'].str.contains('Confirmed'))]
fig5 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_colorado['Date'], y=test_colorado['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_colorado['Date'], y=test_colorado['Cases'])
          ])
fig5.update_layout(barmode='group')

test_italy = test[test['Country_Region'].str.contains('Italy') & (test['Case_Type'].str.contains('Confirmed'))]
fig6 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_italy['Date'], y=test_italy['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_italy['Date'], y=test_italy['Cases'])
          ])
fig6.update_layout(barmode='group')

test_china = test[test['Country_Region'].str.contains('China') & (test['Case_Type'].str.contains('Confirmed'))]
fig7 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_china['Date'], y=test_china['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_china['Date'], y=test_china['Cases'])
          ])
fig7.update_layout(barmode='group')

test_us = test[test['Country_Region'].str.contains('US') & (test['Case_Type'].str.contains('Confirmed'))]
fig8 = go.Figure(data=[
    go.Bar(name='Predicted Confirmed Cases', x=test_us['Date'], y=test_us['predicted_cases']),
    go.Bar(name='Actual Confirmed Cases', x=test_us['Date'], y=test_us['Cases'])
          ])
fig8.update_layout(barmode='group')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    html.H1(children='Predicted vs Actual Cases of Corona Virus'),
    dcc.Dropdown(id='demo-dropdown', 
        options=[{'label': 'California', 'value': 'Cali'}, 
                 {'label': 'New York', 'value': 'NY'},
                 {'label': 'Florida', 'value': 'FL'},
                 {'label': 'Tenessee', 'value': 'TN'},
                 {'label': 'Colorado', 'value': 'CO'},
                 {'label': 'China', 'value': 'CH'},
                 {'label': 'Italy', 'value': 'IT'},
                 {'label': 'United States', 'value': 'US'}
                ]
    ),
    html.Div(id='my-div', children='''
        Charts will go here
    '''),
])
@app.callback(
    Output(component_id='my-div', component_property='children'), 
    [Input(component_id='demo-dropdown', component_property='value')]
)
def update_chart(dropdown_option):
    if dropdown_option == 'Cali':
        return dcc.Graph(figure=fig1)
    if dropdown_option == 'NY':
        return dcc.Graph(figure=fig2)
    if dropdown_option == 'FL':
        return dcc.Graph(figure=fig3)
    if dropdown_option == 'TN':
        return dcc.Graph(figure=fig4)
    if dropdown_option == 'CO':
        return dcc.Graph(figure=fig5)
    if dropdown_option == 'CH':
        return dcc.Graph(figure=fig6)
    if dropdown_option == 'IT':
        return dcc.Graph(figure=fig7)
    if dropdown_option == 'US':
        return dcc.Graph(figure=fig8)

server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)
