# импортируем необходимые библиотеки
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns 
import plotly.graph_objects as go                         
import plotly.express as px
from plotly.subplots import make_subplots  
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed, set_log_level
from sklearn.metrics import mean_absolute_percentage_error
import time

from datetime import datetime, timedelta                   
from dateutil.relativedelta import relativedelta   

# устанавливаем формат страницы браузера: на всю ширину
st.set_page_config(page_title='Dynamic of popular stocks from Yahoo finance in the past 7 years', layout="wide")
# -----------------------------------------------------------------------------------------------------
# Реализуем функции необходимые для работы приложения
# декоратор st.cache увеличивает «отзывчивость» приложения и позволяет кэшировать полученные ранее данные, т.о.
# при каждом запуске приложения загрузка данных с ресурса произойдет один раз
@st.cache_data
def read_data(X):
    # сохраним в пересенную полное название компании указанного актива
    stock = yf.Ticker(X).info['longName']
    # создаем список наименований-ключей, с помощью которых получим общую информацию о компании
    presentation = ['website', 'industry', 'sector', 'currentRatio', 'currentPrice', 'revenueGrowth']
    # создаем пустой список для добавления в него общих данных о компании 
    lst = []
    for info in presentation:
        lst.append(yf.Ticker(X).info[info])
    # вычисляем дату начала интересующего нас исторического периода движения акции: из текущей даты вычитаем 7-летний период
    start_date = datetime.now() - relativedelta(years=7)
    # крайняя дата данных - предыдущий день
    end_date = datetime.now() - timedelta(days=1)
    # загружаем данные
    stock_data = yf.download(X, start=start_date, end=end_date, interval='1d')
    # возвращаем полное название компании, список данных о компании и загруженный датасет с показателями динамики акции
    return stock, lst, stock_data

# Заголовок приложения
st.title('Dynamic of popular stocks from Yahoo finance in the last 7 years')
st.markdown('''
### Welcome to the movement of specified stock a few last years!
**This application is a prototype of a predicted system based on a machine learning model - simple neural network for time series NeuralProphet.** \n
To use the application, you need:
1. Enter a stock's ticker that is presented on web-site: https://finance.yahoo.com/ \n
**You will see the most important dynamics of stock's values.** \n
2. Specify for how long you want to predict Closing Price of this stock.''')

# пользователь вводит тикер акции
X = st.text_input('Please enter ticker name (samples: AAPL, NVDA, GOOG, AMD, MSFT, TSLA, INTC)')
# применяем функцию для загрузки данных с сайта
try: 
    stock, lst, stock_data = read_data(X='INTC')
except:
    st.write('В текущий момент ошибка доступа, попробуйте еще раз')
# презентуем данные о компании
st.markdown(f'### <center> Presentation of the company {stock}', unsafe_allow_html=True)
st.write(f'Website: {lst[0]}') 
st.write(f"Comnapy's business activity: {lst[1]}, {lst[2]}")  
st.write(f'CurrentRatio: {lst[3]}')  
st.write(f'Current price of the stock: {lst[4]} - {datetime.now().strftime("%Y-%m-%d")}')  
st.write(f'Revenue of growth: {lst[5]}%')    
# выводим датасет
st.markdown(f'### <center> Dataset of {stock}', unsafe_allow_html=True)       
st.dataframe(stock_data)

# презентуем различные графики, описывающие динамику поведения акции за последние 7 лет
st.markdown('### <center><font color="darkblue">OHLC chart', unsafe_allow_html=True)
# график поведения всех показателей акции: Open (цена открытия торгов), High (максимальная цена торгов), 
# Low (минимальная стоимость акции в течение торгов), Close(цена на момент закрытия торгов)
fig = go.Figure(data=go.Ohlc(x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                # зеленый - рост акции в день торгов, красный - падение
                increasing_line_color= 'green', decreasing_line_color= 'tomato')) 
fig.update_layout(
    title={'text': f"OHLC of stock {X}",'x':0.5,'xanchor': 'center', 'yanchor': 'top'}, 
    width=1500, height=600,                                                             
    autosize=False, margin=dict(t=30, b=30, l=30, r=10),                                
    template='plotly'                                                                   
)
st.write(fig)
st.markdown('''An OHLC chart is a type of bar chart that shows open, high, low, and closing prices for each period.\ 
            OHLC charts are useful since they show the four major data points over a period.\
            When the price rises over a period, the right line will be above the left, since the close is above the open. 
            Often times, these bars are colored either green. 
            If the price falls during a period, the right line will be below the left, since the close is below the open. 
            These bars are typically colored red.''')

st.markdown("#### <center><font color='darkblue'>Closing Price by the fluctuation of Volume", unsafe_allow_html=True)
# создаем линейный график движения Volume
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[2, 1])
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data.Volume, name='Time series of Volume'),row=1, col=1)
fig.update_traces(line_color='magenta', line_width=0.8)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Time series of Adj Closing Price'), row=2, col=1)
fig.update_traces(line_color='blue', line_width=1, row=2, col=1)
fig.update_layout(
    title={'text': f"Comparison of volume and price dynamics",'x':0.5,'xanchor': 'center', 'yanchor': 'top'}, 
    width=1500, height=700,
    autosize=False, margin=dict(t=30, b=60, l=30, r=10),
    template="plotly",
    legend=dict(yanchor="bottom", y=-0.2, orientation='h', xanchor="right", x=0.9)
    )
st.write(fig)
st.markdown('''The Volume Chart shows the number of shares traded between its daily open and close. 
            Volume includes all the share involved in the transactions. For example, 
            if five transactions occurred in one day, and each transaction involved 100 shares,  
            the trading volume for that day would be 500.
            Volume data can give an indication of whether price trends are sustainable. 
            If an uptrend is accompanied by rising volume, this could indicate that a growing
            number of investors are buying the stock. 
            Therefore Volume data allows traders to better understand supply and demand dynamics.''')

# с помощью функции добавим в наши данные признаки месяца, недели 
# и разницы стоимости акции на момент открытия и закрытия торгов
def feature_engineering(data):
    # создаем признак разницы цены акции в день торгов
    data['Diff'] = data.Open-data.Close
    data['Date'] = pd.to_datetime(data.index, errors='coerce')
    # из даты-признака извлекает год, месяц и день недели в раздельные столбцы с помощью аргумента dt
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month_name(locale='English')
    data['week'] = data['Date'].dt.day_name()
    # удаляем преобразованный признак-дату
    data = data.drop('Date', axis=1)
    return data
# применим функцию
feature_engineering(stock_data)
# отобразим результат на графиках
st.markdown("#### <center><font color='darkblue'>Distribution of a price gap weekly, monthly", unsafe_allow_html=True)
sns.set_theme()
fig, ax = plt.subplots(2,1, figsize=(20, 10))
# столбчатая диаграмма распределения разницы Diff по дням недели week
sns.barplot(data=stock_data, x='week', y='Diff', hue='year', palette='bright', ci=False, ax=ax[0])
ax[0].set_title('Difference of trading when stock market is starting and closing daily in week')  # название графика
# столбчатая диаграмма распределения разницы Diff по месяцам month
sns.barplot(data=stock_data, x='month', y='Diff', hue='year', palette='bright', ci=False, ax=ax[1])
ax[1].set_title('Price gap of trading when stock market is starting and closing monthly in year')
# перемещаем легенду для лучшей наглядности
sns.move_legend(ax[0], "center left", bbox_to_anchor=(1, 0.1))
# удаляем лишнюю легенду
ax[1].get_legend().remove()
plt.tight_layout()
st.write(fig)

# создаем список годов
years = stock_data['year'].unique().tolist()
# проходим в цикле по годам
for i, year in enumerate(years):
    fig = go.Figure()
    # указываем маску[i]-год, который мы отображаем на графике для признаков Open и Close, а также для индексов-дат
    fig.add_trace(go.Scatter(x = stock_data[stock_data.year==years[i]].index, y=stock_data[stock_data.year==years[i]]['Open'],
                    mode='lines', marker_color='green', name="Start trading"))
    fig.add_trace(go.Scatter(x = stock_data[stock_data.year==years[i]].index, y=stock_data[stock_data.year==years[i]]['Close'],
                    mode='lines', marker_color='blue', name="Close trading"))
    fig.update_yaxes(title=f'{years[i]}')                                                        # название для оси ординат
    fig.update_layout(
    title={'text': f"Dynamic {stock} {years[i]}",'x':0.5,'xanchor': 'center', 'yanchor': 'top'}, # параметры названия графика
    width=1500, height=300,                                                                       # параметры размера графика
    autosize=False, margin=dict(t=10, b=10, l=10, r=10),
    template="plotly",
    legend=dict(yanchor="bottom",y=1, xanchor="right", x=1)                                      # расположение легенды
    )
    with st.expander(f"Dynamic of stock {year}", False):
        st.write(fig)  


# функция создания индекса RSI приминает временной ряд и число - период для сглаженного усреднения
def relative_strength_idx(data, n=14):
    close = data['Close']
    differenced = close.diff()           # дифференцируем, приводим к стационарности
    differenced = differenced[1:]        # отсекаем первое значение, так как оно будет NaN
    close_up = differenced.copy()      
    close_down = differenced.copy()
    # разделяем, когда цена росла и когда падала
    close_up[close_up < 0] = 0           # обнуляем отрицательные значения
    close_down[close_down > 0] = 0       # обнуляем положительные значения
    rolling_up = close_up.rolling(n).mean()            # скользящее среднее для цены роста
    rolling_down = close_down.abs().rolling(n).mean()  # скользящее среднее для цены снижения
    rs = rolling_up / rolling_down                     # применяем формулу расчета
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi
# применим функцию вычисления индекса RSI
stock_data['RSI'] = relative_strength_idx(stock_data).fillna(0)
def get_macd(data):
    # рассчитываем экспоненциальные скользящие средние с периодами 12 и 26
    EMA_12 = pd.Series(data['Close'].ewm(span=12, min_periods=12).mean())   
    EMA_26 = pd.Series(data['Close'].ewm(span=26, min_periods=26).mean())
    # считаем разницу
    data['MACD'] = pd.Series(EMA_12 - EMA_26)
    # считает скользящую среднюю от разницы
    data['MACD_signal'] = pd.Series(stock_data.MACD.ewm(span=9, min_periods=9).mean())
    return data

get_macd(stock_data)
# отобразим индексы на графиках
st.markdown(f"#### <center><font color='darkblue'>RSI, MACD of {stock}", unsafe_allow_html=True)
fig = make_subplots(rows=3, cols=1, subplot_titles=('Close Price', 'RSI', 'MACD'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data.Close, name='Close'), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data.RSI, name='RSI'), row=2, col=1)
fig.add_hline(y=30, line_width=1.5, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=50, line_width=1.5, line_dash="dash", line_color="blue", row=2, col=1)
fig.add_hline(y=70, line_width=1.5, line_dash="dash", line_color="red", row=2, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD'), row=3, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_signal'], name='Signal line'), row=3, col=1)

fig.update_layout(
    width=1500, height=1000,
    autosize=False, margin=dict(t=30, b=60, l=30, r=10),
    template="plotly",
    legend=dict(yanchor="bottom", y=-0.2, orientation='h', xanchor="right", x=0.9)
    )
st.write(fig)            

# NeuralProphet
@st.cache_data
def build_model(data):
    set_random_seed(seed=42)
    # создаем датафрейм с нужными параметрами
    data = pd.DataFrame({'ds': stock_data.index, 'y': stock_data.Close}).reset_index().drop('Date', axis=1)
    # создаем экземпляр класса NeuralProphet
    prophet = NeuralProphet(
        growth='off',                   # тип роста тренда
        loss_func='MSE',                # тип функции потерь
        yearly_seasonality='auto',      # ежегодный характер сезонности и последующие отдадим на автоподбор
        weekly_seasonality='auto',                     
        daily_seasonality='auto',                    
        learning_rate=0.07,             # темп обучения                  
        epochs=500,                     # количество итераций
        future_regressors_model='neural_nets',
        future_regressors_d_hidden=3,   # количество внутренних слоев нейросети
        n_lags=1,                       # порядок авторегрессии
        collect_metrics=["MSE", "MAE"], # параметры для вычисления   
        n_forecasts=1,
        drop_missing=True         
    )
    # используем специальную функцию библиотеки для разделения данных на тренировочную и тестовую выборки
    train, test = prophet.split_df(data, freq='D', valid_p = 0.2)
    # обучаем тестовый набор, добавим ранюю остановку, если далее снижение функции потерь не происходит
    metrics_train = prophet.fit(train, freq='D', validation_df=test, metrics=True, early_stopping=True)
    # предсказываем
    prophet_predictions = prophet.predict(test)
    mape = round(mean_absolute_percentage_error(prophet_predictions['y'].iloc[1:], prophet_predictions['yhat1'].iloc[1:])*100,2)
    return data, prophet, mape

df_close, prophet, mape = build_model(stock_data)
st.write(f'Accuracy of model - mean absolute error: {mape}%')

n = st.number_input(label='Please enter for how long you want to get forecast', value=30)
def make_prediction(data, n):
    for i in range(n):
        future_df = prophet.make_future_dataframe(data, periods=10, n_historic_predictions=len(data))
        forecast = prophet.predict(future_df)
        period = pd.DataFrame(data={"ds": pd.bdate_range(start=data["ds"].tail(1).values[0], periods=2, freq="B"), "y": np.NaN})
        data = pd.concat([data, period], axis=0).drop_duplicates(subset='ds').reset_index().drop('index', axis=1)
        data['y'].iloc[-1] = forecast['yhat1'].iloc[-1]
    return forecast

forecast = make_prediction(df_close, n=n)
idx = forecast[forecast['ds']==df_close['ds'].iloc[-1]].index[0]
forecast['y'].iloc[idx+1:] = np.NaN

fig = prophet.plot(forecast, figsize=(20,10), plotting_backend='plotly')
st.write(fig)

fig = prophet.plot_parameters(figsize=(20,3), plotting_backend='plotly')
st.write(fig)
# выведем весь датасет с фактическими и предсказанными значениями цены акции на момент закрытия в левой полосе
st.sidebar.markdown('#### Forecast')
# переименуем названия столбцов
forecast = forecast.rename(columns={'ds': 'Date', 'y': 'True Price of stock', 'yhat1': 'Predicted Price'})
st.sidebar.dataframe(forecast[['Date','True Price of stock', 'Predicted Price']])
