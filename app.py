from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

app = Flask(__name__)

data = pd.read_excel("C:/Users/VladislavManolo/project/data/Fund_Managers_and_Monte_Carlo/vc_funds_fund_manager_groups.xlsx", index_col = 0)
post_2000_data = data[data['Vintage'] >= 2000].copy()
emerging_funds = post_2000_data[post_2000_data['Emerging FM'] == 'Yes']
established_funds = post_2000_data[post_2000_data['Established FM'] == 'Yes']

# Function for running 10,000 Monte Carlo simulations and building portfolios of VC funds based on user criteria
def simulate_portfolio_median_returns(data, n_portfolios = None, minimum_return = None, minimum_risk = None, maximum_risk = None): # n_emerging_funds, n_established_funds,
    
    portfolios_returns = []
    portfolios_risks = []
    portfolios_assets = []
    portfolios_ratios = []

    optimal_portfolio_return = None
    optimal_portfolio_risk = None
    optimal_portfolio_assets = None

    for _ in tqdm(range(n_portfolios), desc = "Simulating Portfolios...", unit = "portfolio"):
        """
        selected_emerging = emerging_funds.sample(n = 1, replace = False)

        emerging_vintage_min = max(2000, selected_emerging['Vintage'].min() - 2)
        emerging_vintage_max = selected_emerging['Vintage'].max() + 2

        additional_emerging = emerging_funds[
            (emerging_funds['Vintage'] >= emerging_vintage_min) &
            (emerging_funds['Vintage'] <= emerging_vintage_max)
        ].sample(n = n_emerging_funds, replace = False)

        selected_established = established_funds[
            (established_funds['Vintage'] >= emerging_vintage_min) &
            (established_funds['Vintage'] <= emerging_vintage_max)
        ].sample(n = n_established_funds, replace = False)
        """
        funds = np.random.randint(6,20)
        portfolio_composition = post_2000_data.sample(funds, replace=False)
        #portfolio_composition = pd.concat([selected_emerging, additional_emerging, selected_established])

        returns = portfolio_composition.loc[:, 'MOIC']
        selected_assets = portfolio_composition.loc[:, 'FUND ID'].sample(n = len(portfolio_composition), replace = False).values
        
        try:
            weights = np.ones(len(selected_assets)) / len(selected_assets)
            portfolio_return = np.dot(weights.T, returns)
            portfolio_risk = np.std(returns)
            portfolio_ratio = portfolio_return / portfolio_risk

            portfolios_returns.append(portfolio_return)
            portfolios_risks.append(portfolio_risk)
            portfolios_assets.append(selected_assets)
            portfolios_ratios.append(portfolio_ratio)

            if optimal_portfolio_return is None or (portfolio_ratio) > (optimal_portfolio_ratio):
                optimal_portfolio_return = portfolio_return
                optimal_portfolio_risk = portfolio_risk
                optimal_portfolio_ratio = portfolio_ratio
                optimal_portfolio_assets = selected_assets #.copy()
        except ValueError as e:
            continue

    if optimal_portfolio_assets is not None:
        print(f"Assets in the Optimal Portfolio: {optimal_portfolio_assets}")

    optimal_portfolio_index = portfolios_risks.index(optimal_portfolio_risk)
    print(f"Portfolio Assets for Optimal Portfolio: {portfolios_assets[optimal_portfolio_index]}")

    filtered_portfolios_assets = []
    filtered_portfolios_returns = []
    filtered_portfolios_risks = []
    filtered_portfolios_ratios = []

    for assets, return_val, risk_val, ratio_val in zip(portfolios_assets, portfolios_returns, portfolios_risks, portfolios_ratios):
        if (return_val >= minimum_return) and (risk_val >= minimum_risk and risk_val <= maximum_risk):
            filtered_portfolios = [
                {
                    'Assets': ', '.join(data[data['FUND ID'].isin(assets)]['NAME']),
                    'Return': return_val.round(2),
                    'Risk': risk_val.round(2),
                    'Return-to-Risk Ratio': ratio_val
                }
            ]
            filtered_portfolios_assets.append(assets)
            filtered_portfolios_returns.append(return_val.round(2))
            filtered_portfolios_risks.append(risk_val.round(2))
            filtered_portfolios_ratios.append(ratio_val.round(2))

    fig = go.Figure()

    portfolio_text = [
        f"Assets: {', '.join(map(str, assets))}<br>\
        Portfolio Return: {return_val.round(2)}<br>\
        Portfolio Risk: {risk_val.round(2)}<br>\
        Portfolio Return-to-Risk Ratio: {ratio_val:.2f}"
        for assets, return_val, risk_val, ratio_val in zip(filtered_portfolios_assets, filtered_portfolios_returns, filtered_portfolios_risks, filtered_portfolios_ratios)
    ]
    
    fig.add_trace(
        go.Scatter(
            x = filtered_portfolios_risks,
            y = filtered_portfolios_returns,
            mode = 'markers',
            marker = dict(
                color = filtered_portfolios_ratios,
                colorscale = 'RdBu',
                showscale = True,
                colorbar = dict(title = 'Risk-Return Ratio')
            ),
            hoverinfo = 'text',
            text = portfolio_text
        )
    )

    optimal_portfolio_text = f"Assets: {', '.join(map(str, optimal_portfolio_assets))}<br>\
        Portfolio Return: {optimal_portfolio_return.round(2)}<br>\
        Portfolio Risk: {optimal_portfolio_risk.round(2)}<br>\
        Portfolio Return-to-Risk Ratio: {optimal_portfolio_ratio:.2f}"

    if optimal_portfolio_return is not None and optimal_portfolio_risk is not None:
        fig.add_trace(
            go.Scatter(
                x = [optimal_portfolio_risk],
                y = [optimal_portfolio_return],
                mode = 'markers',
                marker = dict(color = 'red'),
                name = 'Optimal Portfolio',
                text = optimal_portfolio_text,
                hoverinfo = 'text'
            )
        )

    fig.update_layout(
        title = f'Risk-Return Graph', #: {manager_type}
        xaxis = dict(title='Portfolio Risk (Volatility)'),
        yaxis = dict(title='Portfolio Return'),
        showlegend = True,
        legend = dict(x = 1.02, y = 1.3, traceorder = 'normal', orientation = 'v')
    )
    
    fig.show()
    
    filtered_portfolios = [
        {
            'Assets': ', '.join(data[data['FUND ID'].isin(assets)]['NAME']),
            'Return': return_val.round(2),
            'Risk': risk_val.round(2),
            'Return-to-Risk Ratio': ratio_val
        }
        for assets, return_val, risk_val, ratio_val in zip(portfolios_assets, portfolios_returns, portfolios_risks, portfolios_ratios)
        if (return_val >= minimum_return) and (risk_val >= minimum_risk and risk_val <= maximum_risk)
    ]
    
    return {'portfolio_composition': portfolio_composition, 'filtered_portfolios': filtered_portfolios}
    
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':         # Retrieving user inputs from the form
        #num_funds = int(request.form['num_funds'])
        #n_emerging = int(request.form['n_emerging'])
        #n_established = int(request.form['n_established'])
        num_simulations = int(request.form['num_simulations'])
        min_return = float(request.form['min_return'])
        min_risk = float(request.form['min_risk'])
        max_risk = float(request.form['max_risk'])
        # Calling the simulate_portfolio_median_returns function with user-entered parameters
        result = simulate_portfolio_median_returns(post_2000_data,  n_portfolios = num_simulations, minimum_return = min_return, minimum_risk = min_risk, maximum_risk = max_risk) # plot_title, num_funds, n_emerging, n_established,
        filtered_portfolios = result['filtered_portfolios']

        optimal_portfolio_index = np.argmax([portfolio['Return-to-Risk Ratio'] for portfolio in filtered_portfolios])
        optimal_portfolio = filtered_portfolios[optimal_portfolio_index]
        print(optimal_portfolio)
        return render_template('result.html', result = result, optimal_portfolio = optimal_portfolio)

    return render_template('index.html')

@app.route('/select_portfolios', methods=['POST'])
def select_portfolios():
    selected_portfolios = request.form.getlist('selected_portfolios')
    print("Selected Portfolios:", selected_portfolios)
    return redirect('/')  # Redirect to the home page or another page as needed

if __name__ == '__main__':
    app.run(debug = True)