# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
#from flask_jsonpify import jsonify
import yfinance as yf

import datetime
import math
import pandas as pd
import numpy as np
import preprocessing

from pandas import Series, DataFrame
import yfinance as yf

from matplotlib import style

import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import json

app = Flask(__name__)
api = Api(app)

CORS(app)

from flask import send_file

# http://localhost:5000/get_image?type=1

# @app.route('/get_images')
class Mavg(Resource):
    def get(self):
       
        return send_file('mavg.png', mimetype='image/gif')
    

# @app.route('/get_image')
class Ret(Resource):
    def get(self):
        
        return send_file('return.png', mimetype='image/gif')


# @app.route('/get_return')
class Comp(Resource):
    def get(self):
        
        return send_file('compare.png', mimetype='image/gif')


class Cor(Resource):
    def get(self):
        
        return send_file('correlation.png', mimetype='image/gif')
    

class RRrate(Resource):
    def get(self):
        
        return send_file('risk-ret-rate.png', mimetype='image/gif')


class Forecast(Resource):
    def get(self):
        
        return send_file('forecast.png', mimetype='image/gif')

class Monte(Resource):
    def get(self):
        
        return send_file('monte.png', mimetype='image/gif')

class Histo(Resource):
    def get(self):
        
        return send_file('histo.png', mimetype='image/gif')


class PortfolioImage(Resource):
    def get(self):
        
        return send_file('portfolio.png', mimetype='image/gif')

# http://localhost:5000/portfolio?c1=AMZN&c2=GOOG&c3=AAPL&c4=FB
class Portfolio(Resource):
    def get(self):

        c1 = request.args.get('c1')
        c2 = request.args.get('c2')
        c3 = request.args.get('c3')
        c4 = request.args.get('c4')
        # startDate = request.args.get('start')

        stock = [c1,c2,c3,c4]#['AMZN', 'GOOG', 'AAPL','FB']#['AAPL', 'GOOG','MSFT','GE', 'IBM']#['ZM','UBER','SWI','RNG','CRWD', 'WORK']
        # ['AAPL', 'GE', 'GOOG', 'IBM']

        data = yf.download(stock,start = '2019-01-27', end='2020-03-27')['Adj Close']

        stock_ret = np.log(data/data.shift(1))

        num_ports = 5000
        all_weights = np.zeros((num_ports, len(data.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for ind in range(num_ports): 
            weights = np.array(np.random.random(4)) 
            weights = weights/np.sum(weights)  
            
            all_weights[ind,:] = weights
            
            ret_arr[ind] = np.sum((stock_ret.mean()*weights)*252)

            vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(stock_ret.cov()*252, weights)))

            sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

        plt.figure(figsize=(8,7))
        plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(vol_arr[sharpe_arr.argmax()], ret_arr[sharpe_arr.argmax()], c='red', s=50, edgecolors='black')
        plt.scatter(vol_arr[vol_arr.argmin()], ret_arr[vol_arr.argmin()], c='blue', s=50, edgecolors='black')

        plt.show()
        plt.savefig('portfolio.png', bbox_inches='tight')
        plt.clf()

        print(vol_arr[sharpe_arr.argmax()])
        print(ret_arr[sharpe_arr.argmax()])
        print(all_weights[sharpe_arr.argmax(),:])

        my_json_string = json.dumps({'Max-Sharpe-Risk': vol_arr[sharpe_arr.argmax()], 'Max-Sharpe-Return': ret_arr[sharpe_arr.argmax()], 'Max-w1':all_weights[sharpe_arr.argmax(),0], 'Max-w2':all_weights[sharpe_arr.argmax(),1], 'Max-w3':all_weights[sharpe_arr.argmax(),2], 'Max-w4':all_weights[sharpe_arr.argmax(),3], 'Min-Risk': vol_arr[vol_arr.argmin()], 'Min-Risk-Return': ret_arr[vol_arr.argmin()], 'Min-w1':all_weights[vol_arr.argmin(),0], 'Min-w2':all_weights[vol_arr.argmin(),1], 'Min-w3':all_weights[vol_arr.argmin(),2], 'Min-w4':all_weights[vol_arr.argmin(),3]})

        print(vol_arr[vol_arr.argmin()])
        print(ret_arr[vol_arr.argmin()])
        print(all_weights[vol_arr.argmin(),:])
        return my_json_string

# @app.route('/add')
class Employees(Resource):
    def get(self):
        # global var
        # global test
        # var += 1
        # test='/test'+str(var)
        # args = request.args
        # print(var)
        
        company = request.args.get('company')
        compare = request.args.get('compare')

        df = yf.download(company, start = '2016-01-01', end='2020-04-20')

        close_px = df['Adj Close']
        mavg = close_px.rolling(window=100).mean()

        print(mavg)

        print(df.head())

        print(df.tail())

        # import matplotlib.pyplot as plt
        # from matplotlib import style

        # # Adjusting the size of matplotlib
        # import matplotlib as mpl
        mpl.rc('figure', figsize=(8, 7))
        mpl.__version__

        # Adjusting the style of matplotlib
        style.use('ggplot')

        close_px.plot(label=company)
        mavg.plot(label='mavg')
        plt.legend()

        plt.savefig('mavg.png', bbox_inches='tight')

        plt.clf()
        # plt.show()

        rets = close_px / close_px.shift(1) - 1
        rets.plot(label='return')
        plt.savefig('return.png', bbox_inches='tight')
        plt.clf()
        
        # plt.show()


        dfcomp = yf.download(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],start = '2016-01-01', end='2020-03-28')['Adj Close']

        print(dfcomp.tail())

        
        retscomp = dfcomp.pct_change()

        corr = retscomp.corr()
        print("hi")

        # cols = [col for col in retscomp.columns if compare in col]
        # print(retscomp[cols])
        print(corr)

        plt.scatter(retscomp[company], retscomp[compare])
        plt.xlabel('Returns-'+company)
        plt.ylabel('Returns-'+compare)

        plt.savefig('compare.png', bbox_inches='tight')
        plt.clf()

        # plt.show()


        #   Error
        #pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));


        plt.imshow(corr, cmap='hot', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns)
        plt.yticks(range(len(corr)), corr.columns)

        # plt.show()
        plt.savefig('correlation.png', bbox_inches='tight')
        plt.clf()


        plt.scatter(retscomp.mean(), retscomp.std())
        plt.xlabel('Expected returns')
        plt.ylabel('Risk')
        # for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        #     plt.annotate(
        #         label, 
        #         xy = (x, y), xytext = (20, -20),
        #         textcoords = 'offset points', ha = 'right', va = 'bottom',
        #         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        # plt.show()
        plt.savefig('risk-ret-rate.png', bbox_inches='tight')
        plt.clf()

        dfreg = df.loc[:,['Adj Close','Volume']]

        a=df['High'] - df['Close']
        print(a)


        dfreg['HL_PCT'] = a / df['Close'] * 100.0
        print("yo --- yo ")

        print(dfreg['HL_PCT'])

        print(df['Close'])
        print(df['Open'])

        b = df['Close'] - df['Open']

        dfreg['PCT_change'] = b / df['Open'] * 100.0

        # import math
        # import numpy as np
        # from sklearn import preprocessing, svm
        # from sklearn.model_selection import train_test_split

        # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)

        print(dfreg.shape)
        # We want to separate 1 percent of the data to forecast
        forecast_out = int(math.ceil(0.01 * len(dfreg)))

        # Separating the label here, we want to predict the AdjClose
        forecast_col = 'Adj Close'
        dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(['label'], 1))

        # Scale the X so that everyone can have the same distribution for linear regression
        X = preprocessing.scale(X)

        # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        # Separate label and identify it as y
        y = np.array(dfreg['label'])
        y = y[:-forecast_out]

        print('Dimension of X',X.shape)
        print('Dimension of y',y.shape)

        # Separation of training and testing of model by cross validation train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # from sklearn.linear_model import LinearRegression
        # from sklearn.neighbors import KNeighborsRegressor

        # from sklearn.linear_model import Ridge
        # from sklearn.preprocessing import PolynomialFeatures
        # from sklearn.pipeline import make_pipeline

        # Linear regression
        clfreg = LinearRegression(n_jobs=-1)
        clfreg.fit(X_train, y_train)


        # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)

        # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)
            
        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)

        confidencereg = clfreg.score(X_test, y_test)
        confidencepoly2 = clfpoly2.score(X_test,y_test)
        confidencepoly3 = clfpoly3.score(X_test,y_test)
        confidenceknn = clfknn.score(X_test, y_test)

        print("The linear regression confidence is ",confidencereg)
        print("The quadratic regression 2 confidence is ",confidencepoly2)
        print("The quadratic regression 3 confidence is ",confidencepoly3)
        print("The knn regression confidence is ",confidenceknn)

        # Printing the forecast
        forecast_set = clfreg.predict(X_lately)
        dfreg['Forecast'] = np.nan
        print(forecast_set, confidencereg, forecast_out)


        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + datetime.timedelta(days=1)

        for i in forecast_set:
            next_date = next_unix
            next_unix += datetime.timedelta(days=1)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
            
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        # plt.show()
        plt.savefig('forecast.png', bbox_inches='tight')
        plt.clf()


        # from scipy.stats import norm

        # data = yf.download("AAPL", start = '2012-01-01', end='2017-01-01')['Adj Close']

        result=[]
        #Define Variables
        S = yf.download(company, start = '2016-01-01', end='2020-03-28')['Adj Close'][-1]#apple['Adj Close'][-1] #starting stock price (i.e. last available real stock price)
        T = 50 #Number of trading days
        days = (df.index[-1] - df.index[0]).days
        cagr = ((((df['Adj Close'][-1]) / df['Adj Close'][1])) ** (365.0/days)) - 1
        mu = cagr# 0.2309 #Return

        df['Returns'] = df['Adj Close'].pct_change()
        vol = df['Returns'].std()*math.sqrt(252)
        # vol = #0.4259 #Volatility


        #choose number of runs to simulate - I have chosen 10,000
        for i in range(100):
            #create list of daily returns using random normal distribution
            daily_returns=np.random.normal((1+mu)**(1/T),vol/math.sqrt(T),T)
            
            #set starting price and create price series generated by above random daily returns
            price_list = [S]
            
            for x in daily_returns:
                price_list.append(price_list[-1]*x)

            #plot data from each individual run which we will plot at the end
            plt.plot(price_list)
            
            #append the ending value of each simulated run to the empty list we created at the beginning
            result.append(price_list[-1])

        #show the plot of multiple price series created above
        # plt.show()
        plt.savefig('monte.png', bbox_inches='tight')
        plt.clf()

        #create histogram of ending stock values for our mutliple simulations
        plt.hist(result,bins=50)
        # plt.show()
        plt.savefig('histo.png', bbox_inches='tight')
        plt.clf()

        print("Mean - ")

        #use numpy mean function to calculate the mean of the result
        print(round(np.mean(result),2))

        mean = json.dumps({'mean': round(np.mean(result),2)})
        return mean 

@app.route("/")
def index():
    return "<h1>Hello Azure!</h1>"

# var=1
# test='/test'+str(var)
test='/test'
api.add_resource(Employees, test) # Route_1
# api.add_resource(R, '/r') 
api.add_resource(Mavg, '/mavg') 
api.add_resource(Ret, '/return') 
api.add_resource(Comp, '/compare') 
api.add_resource(Cor, '/correlation') 
api.add_resource(RRrate, '/RRrate') 
api.add_resource(Forecast, '/forecast') 
api.add_resource(Monte, '/monte') 
api.add_resource(Histo, '/histo') 
api.add_resource(Portfolio, '/portfolio') 
api.add_resource(PortfolioImage, '/portfolioimage') 


if __name__ == '__main__':
    # company = "GE"

    # df = yf.download(company, start = '2016-01-01', end='2020-03-03')#'2017-01-01')#web.DataReader("AAPL", 'yahoo', start, end)
    # close_px = df['Adj Close']
    # mavg = close_px.rolling(window=100).mean()
        
    # print(mavg)
    # print(df.head())
    # print(df.tail())
    # print("hi")
    # mpl.rc('figure', figsize=(8, 7))
    # mpl.__version__
    app.run()






