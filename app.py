import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px

from sklearn.linear_model import LinearRegression

df = pd.read_csv('correlation_data.csv', index_col=[0])[1:]

x_list = ['population', 'pop_2010', 'pop_change',
        'under_5', 'under_18', 'over_65', 'women', 'white',
       'black', 'native_american', 'asian', 'pacific_islander',
       'two_or_more_races', 'hispanic', 'white_not_hispanic', 'veterans',
       'foreign_born', 'housing_units', 'owner_occupied_housing_units',
       'med_value', 'med_owner_costs_w_mortgage',
       'med_owner_costs_no_mortgage', 'median_rent', 'building permits',
       'households', 'ppl_per_household', 'same_house_1_year',
       'language_non_english', 'households_w_computer',
       'households_w_broadband', 'high_school_grad', 'college_grad',
       'disability', 'ppl_without_health_insurance', 'labor_force',
       'women_labor_force', 'accom_food_sales', 'healthcare_revenue',
       'manufacturer_shipments', 'retail_sales',
       'retail_sales_per_capita', 'commute_time', 'median_household_income',
       'per_capita_income', 'poverty', 'hospital_beds']

title_dict = {'population': 'Population', 'pop_2010': 'Population in 2010',
        'pop_change': 'Population change, percentage, 2010-2018',
        'under_5': 'Under 5 years old, percent of persons',
        'under_18': 'Under 18 years old, percent of persons',
        'over_65': 'Over 65 years old, percent of persons', 'women': 'Women, percentage',
        'white': 'White, percentage',
        'black': 'Black or African American, percentage', 'native_american': 'Native American, percentage',
        'asian': 'Asian, percentage', 'pacific_islander': 'Native Hawaiian and Other Pacific Islander',
       'two_or_more_races': 'Two or more races, percentage', 'hispanic': 'Hispanic or Latino, percentage',
              'white_not_hispanic': 'White, not Hispanic, percentage', 'veterans': 'Veterans, total number, 2014-2018',
       'foreign_born': 'Foreign born persons, percentage, 2014-2018', 'housing_units': 'Housing units, July 2018',
              'owner_occupied_housing_units': 'Owner occupied housing unit rate, 2014-2018',
       'med_value': 'Median value of owner occupied housing unites, 2014-2018',
       'med_owner_costs_w_mortgage': 'Median selected monthly owner costs, with a mortgage, 2014-2018',
       'med_owner_costs_no_mortgage':  'Median selected monthly owner costs, with a mortgage, 2014-2018',
              'median_rent': 'Median gross rent, 2014-2018', 'building permits': 'Building permits, 2018',
       'households': 'Households, 2014-2018', 'ppl_per_household': 'Persons per household, 2014-2018',
              'same_house_1_year': 'Living in same house 1 year ago, percent of persons',
       'language_non_english': 'Language other than English spoken at home, percent of persons',
              'households_w_computer': 'Households with a computer, percentage, 2014-2018',
       'households_w_broadband': 'Households with broadband internet, 2014-2018',
       'high_school_grad': 'High school graduate or higher, percent of persons 25+ years old',
       'college_grad': "Bachelor's degree or higher, percent of persons 25+ years old",
       'disability': 'With a disability, under age 65, percent',
       'ppl_without_health_insurance': 'Persons withouth health insurance, under age 65, percent',
              'labor_force': 'In civilian labor force, total, percent of population 16+ years old',
       'women_labor_force': 'In civilian labor force, women, percent of population 16+ years old',
              'accom_food_sales': 'Total accommodation and food service sales, 2012 (thousands)',
              'healthcare_revenue': 'Total healthcare and social assitance revenue, 2012 (thousands)',
       'manufacturer_shipments': 'Total manufacturer shipments, 2012 (thousands)',
       'retail_sales': 'Total retail sales, 2012 (thousands)',
       'retail_sales_per_capita': 'Total retail sales per capita, 2012', 'commute_time':
                  'Mean travel time to work (minutes)', 'median_household_income': 'Median household income (in 2018 dollars)',
       'per_capita_income': 'Per capita income in last 12 months (in 2018 dollars)',
              'poverty': 'Persons in poverty, percentage', 'hospital_beds':'Total hospital beds at facilities in county'}

y_list = ['cases', 'deaths', 'cases_per_1000', 'deaths_per_1000', 'deaths_per_cases']

#dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

fig = go.Figure()

app.layout = html.Div([
    html.H2("Correlation plots"),
    html.H4("How do specific demographics correlate to death rates in South Carolina counties?"),
    dcc.Graph(id='graph-with-dropdown'),
    html.Div(id='display-value'),
    html.Br(),
    dcc.Dropdown(
        id='feature_select',
        options=[{'label': title_dict[i], 'value': i} for i in x_list],
        value='population'

    ),
    html.Br(),
    html.P('Select a feature from the dropdown box to plot it against deaths per 1000. '
           'Hover over each point to see the county name and other data.'),
    html.Br(),
    html.P('Notes: After selecting a feature, the Pearson correlation coefficient is also calculated. '
           'Values closer to 1 and -1 imply stronger relationships, while values closer to 0 suggest the opposite. '
           'The trend line provided is calculated using least squares, and it is not an indication of a relationship, '
           'it only illustrates the general trend. Again, correlation does not imply causation.'),
    html.Br(),
    html.Img(src='https://www.data4sc.com/wp-content/uploads/2021/08/d4sc_logo4-sm.jpg'),
    html.Br(),
    html.A("Data 4 SC", href="https://www.data4sc.com")
], style={"margin": 100})

@app.callback(
    [Output('graph-with-dropdown', 'figure'),
     Output('display-value', 'children')],
    [Input('feature_select', 'value')])

# def update_figure(selected_feature):
def multi_update(selected_feature):

    model = LinearRegression()
    model.fit(np.array(df[selected_feature]).reshape((-1,1)), df['deaths_per_1000'])

    x_range = np.linspace(df[selected_feature].min(), df[selected_feature].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    figure = px.scatter(
        df, x=selected_feature, y='deaths_per_1000',
        hover_data=['county'],
        labels={
            "deaths_per_1000": "Deaths per 1000 persons",
            selected_feature: title_dict[selected_feature]},
        template='ggplot2'
    )
    figure.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression trend line'))

    figure.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    corr = df.corr()['deaths_per_1000'].loc[selected_feature]

    value = f"Pearson correlation coefficient: {corr:0.02}"

    return figure, value


if __name__ == '__main__':
    app.run_server(debug=True)