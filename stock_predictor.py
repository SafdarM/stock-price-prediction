# Working

from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import plotly.express as px
import sys

app = Flask(__name__)
# nifty50_symbols = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
#                    'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS',
#                    'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFC.NS',
#                    'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
#                    'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',
#                    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
#                    'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS',
#                    'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS']
nifty50_symbols = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS']

# Define the start and end dates for data retrieval
end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
start_date_training = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 years ago
start_date_actual = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days ago
end_date_actual = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
start_date_predict = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date_predict = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')


# Define the SARIMA model
def fit_sarima_model(data):
    model = SARIMAX(data, order=(2, 1, 2))
    model_fit = model.fit(disp=False)
    return model_fit

@app.route('/')
def index():
    return render_template('index.html', nifty50_symbols=nifty50_symbols)

@app.route('/', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        selected_stock = request.form['stock_selection']
        stock_data = yf.download(selected_stock, start=start_date_training, end=end_date_predict)

        # Fit SARIMA model
        model_fit = fit_sarima_model(stock_data['Close'])

        # Get the predicted values and the last month's actual values
        predicted_values = model_fit.get_prediction(start=-30, dynamic=False)
        actual_last_month = stock_data['Close'][-30:]

        # Calculate delta values and percentages
        deltas = [abs(predicted - actual) for predicted, actual in
                  zip(predicted_values.predicted_mean, actual_last_month)]
        delta_percentages = [(delta_value / actual_value) * 100 for delta_value, actual_value in
                             zip(deltas, actual_last_month)]

        # Create a Plotly figure
        fig = go.Figure()

        # Add predicted line
        fig.add_trace(go.Scatter(x=predicted_values.predicted_mean.index, y=predicted_values.predicted_mean,
                                 customdata=list(zip(predicted_values.predicted_mean, actual_last_month, deltas,
                                                     delta_percentages)),
                                 mode='lines+markers', name='Predicted Last Month',
                                 hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: %{y:$}<br>Actual: %{customdata[1]:$}<br>Delta: %{customdata[2]:.2f}<br>Delta (%): %{customdata[3]:.2f}'))

        # Add actual line
        fig.add_trace(go.Scatter(x=actual_last_month.index, y=actual_last_month,
                                 customdata=list(zip(predicted_values.predicted_mean, actual_last_month, deltas,
                                                     delta_percentages)),
                                 mode='lines+markers', name='Actual Last Month',
                                 hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:$}<br>Predicted: %{customdata[0]:$}<br>Delta: %{customdata[2]:.2f}<br>Delta (%): %{customdata[3]:.2f}'))

        # Set x-axis title and format
        fig.update_xaxes(title_text='Date', tickformat='%Y-%m-%d')

        # Set y-axis title
        fig.update_yaxes(title_text='Stock Price')


        # Create a Plotly figure
        fig_future = go.Figure()

        # predicted_values = model_fit.get_prediction(steps=61)
        predicted_values = model_fit.get_prediction(end=30, dynamic=False)
        print(type(predicted_values.predicted_mean), predicted_values.predicted_mean, file=sys.stderr)
        # Add predicted line
        fig_future.add_trace(go.Scatter(x=predicted_values.predicted_mean.index[-30:], y=predicted_values.predicted_mean[-30:],
                                 customdata=predicted_values.predicted_mean,
                                 mode='lines+markers', name='Future Price',
                                 hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: %{customdata:$}<br>'))

        # Set x-axis title and format
        fig_future.update_xaxes(title_text='Date', tickformat='%Y-%m-%d')

        # Set y-axis title
        fig_future.update_yaxes(title_text='Stock Price')

        return render_template('index.html', nifty50_symbols=nifty50_symbols, plot=fig.to_html(), plot_fut=fig_future.to_html())

    return render_template('index.html', nifty50_symbols=nifty50_symbols, plot=None, plot_fut=None)



if __name__ == '__main__':
    app.run(debug=True)


