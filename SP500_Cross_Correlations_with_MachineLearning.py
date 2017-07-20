
# Import Python libraries
#
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
from statistics import mean
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

style.use('ggplot')
# %matplotlib inline

DATA_PATH  = 'C:\\ML_Data\\'
SP500_SYMS_FILE = 'SP500_Ticker_Syms.pickle'
SP500_JOINED_CLOSES_CSV = 'sp500_joined_closes.csv'
SP500_SYMS_DATA_FLDR = 'SP500_syms_data\\'

PCT_CHG_REQ = 0.02
TEST_SIZE = 0.25
NUM_CORR_DAYS = 7 # number of days for correlation

def save_sp500_ticker_syms():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml') # text of the source code for web page

# find the table contain the SP500 list of ticker syms
#
    table = soup.find('table', {'class':'wikitable sortable'})
    ticker_syms = []
    
    for row in table.findAll('tr')[1:]:
        ticker_sym = row.findAll('td')[0].text
        ticker_syms.append(ticker_sym)
    
    with open(DATA_PATH + SP500_SYMS_FILE, 'wb') as f:
        pickle.dump(ticker_syms, f)
    
    print(ticker_syms)
    return 


save_sp500_ticker_syms()

# get closing price data from Yahoo finance
# use our pickled SP500 ticker sym list to
# drive this data loading
#
def get_prices_fr_yahoo(reload_SP500=False):
    
    if reload_SP500:
        ticker_syms = save_sp500_ticker_syms()
    else:
# open our previously saved SP500 sym list
#
        with open(DATA_PATH + SP500_SYMS_FILE, 'rb') as f:
            ticker_syms = pickle.load(f)

# see if our top level SP500 sym data folder exists
# if not, create it
#
    if not os.path.exists(DATA_PATH + SP500_SYMS_DATA_FLDR):
        os.makedirs(DATA_PATH + SP500_SYMS_DATA_FLDR)
    
# define our start and end dates for retrieving
# our stock data
#
    start_dt = dt.datetime(2000,1,1)
    end_dt   = dt.datetime(2016,12,31)

# now iterate through each of the 500 ticker symbols
# in our SP500 sym dataframe
#
    ticker_cnt = 1
    
    for ticker_sym in ticker_syms:
        print("Loading data for: ", ticker_sym, "#: ", ticker_cnt)
        ticker_cnt += 1

        if not os.path.exists(DATA_PATH + SP500_SYMS_DATA_FLDR + ticker_sym + '.csv'):
# get stock pricing data from Yahoo
#
            df = web.DataReader(ticker_sym, 'yahoo', start_dt, end_dt)

# and save it locally to a .CSV file for each SP500 ticker sym
#
            df.to_csv(DATA_PATH + SP500_SYMS_DATA_FLDR + ticker_sym + '.csv')
        else:
            print('{} CSV file already exists'.format(ticker_sym))

# get 16 year daily close pricing data from Yahoo for each SP500 symbol
#
get_prices_fr_yahoo()

# now let's compile all the Adj Close data for
# SP500 symbol into one large dataframe
#
def join_SP500_data():
    with open(DATA_PATH + SP500_SYMS_FILE, 'rb') as f:
        ticker_syms = pickle.load(f)
        
    main_df = pd.DataFrame()
    
    for count,ticker_sym in enumerate(ticker_syms):
        ticker_sym_csv = DATA_PATH + SP500_SYMS_DATA_FLDR + ticker_sym + '.csv'
#        print(ticker_sym_csv)
        df = pd.read_csv(ticker_sym_csv)
        df.set_index('Date', inplace=True)
        
# rename the Adj Close column to be the ticker sym
#
        df.rename(columns = {'Adj Close': ticker_sym}, inplace=True)
    
# and drop all columns except the Adj Close/Sym
#
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)
        
# now join these DFs
#
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')   # append the sym's AdjClose column 
            
        
        if count % 10 == 0:
            print(count, ticker_sym)
        
    print(main_df.head())
    print(main_df.tail())

# and save out the complete main_df
#
    main_df.to_csv(DATA_PATH + SP500_JOINED_CLOSES_CSV)
    

join_SP500_data()

def visualize_SP500_data():
    df = pd.read_csv(DATA_PATH + SP500_JOINED_CLOSES_CSV)
#    df['AAPL'].plot()
#    plt.show()

# create a correlation table of our SP500 dataframe
#
    df_corr = df.corr()
    
#    print(df_corr.head())

# now let's visualize the correlations
# get the inner dataframes
# which is a numpy array of the colunms and rows
#
    data = df_corr.values
    fig = plt.figure()
    axis = fig.add_subplot(1,1,1)  # 1x1 plot #1

# create a heatmap ranging from red (negative) to 
# yellow (neutral) to green (positive) correlations
#
    heatmap = axis.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    
# arrange ticks at half marks
#
    axis.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    axis.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    
    axis.invert_yaxis()
    axis.xaxis.tick_top() # move our X asis to the top

# define our graph labels
#
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    axis.set_xticklabels(column_labels) 
    axis.set_yticklabels(row_labels) 
    plt.xticks(rotation=90)

# normalize graph scale to -1 -> + 1
#
    heatmap.set_clim(-1, 1) 
    
    plt.tight_layout() # clean things up a bit
    plt.show()

    

visualize_SP500_data()



# all stocks are a: Buy, Sell or Hold
# Train: did the price within the next 7 trading
# days, did the price:
#
# go up > 2%: Buy
# go down < 2%: Sell
# did neither up or down 2%: Hold
#
# Features define, labels are our target
#
# Each model that we gen is going to be on a per company/sym basis
# 
# Each company is going is going to take into
# account the pricing data for all the other SP500 companies
#
def process_data_for_labels(ticker_sym):
    num_days = NUM_CORR_DAYS
    
    df = pd.read_csv(DATA_PATH + SP500_JOINED_CLOSES_CSV, index_col=0)
    
    ticker_syms = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    
    for i in range(1, num_days+1):
        
# get % change over 7 days to generate a new column
#
        df['{}_{}d'.format(ticker_sym, i)] = (df[ticker_sym].shift(-i) 
                                             - df[ticker_sym]) / df[ticker_sym]
    df.fillna(0, inplace=True)
    
    return ticker_syms, df

process_data_for_labels('AAPL')


# function to map to pandas dataframe columns:
# Labels: Buy, Sell, Hold
# Features: the % price change for all companies
#
# Our machine learning goal will be: better than 33% accuracy
#
# we'll be passing the whole week of % changes
#
def buy_sell_hold(*args):  # pass any # of args
    cols = [c for c in args]
    
    pct_chg_requirem = PCT_CHG_REQ
    
    for col in cols:
        if col > pct_chg_requirem:
            return 1  # Buy
        if col < -pct_chg_requirem:
            return -1 # Sell
    return 0 # Hold


def extract_featuresets(ticker_sym):
    ticker_syms, df = process_data_for_labels(ticker_sym)

# define a new column that's mapped to 
# Buy, Sell or Hold (our class)
#
    df['{}_target'.format(ticker_sym)] = list(map (buy_sell_hold,
                                                   df['{}_1d'.format(ticker_sym)],
                                                   df['{}_2d'.format(ticker_sym)],
                                                   df['{}_3d'.format(ticker_sym)],
                                                   df['{}_4d'.format(ticker_sym)],
                                                   df['{}_5d'.format(ticker_sym)],
                                                   df['{}_6d'.format(ticker_sym)],
                                                   df['{}_7d'.format(ticker_sym)]
                                                  ))
    
    vals = df['{}_target'.format(ticker_sym)].values.tolist()
    str_vals = [str(i) for i in vals]

# get our distribution
#
    print(ticker_sym, ' Data Class Counts: ', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
# want to be very explicit when it comes to
# to choosing which columns should be in the features set
#
    df_vals = df[[ticker_sym for ticker_sym in ticker_syms]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

# now define our featuresets and labels
# featuresets is : pct chgs daily
# the target, class is y
#
    X = df_vals.values
    y = df['{}_target'.format(ticker_sym)].values

# return featuresets, labels, and the dataframe
#
    return X, y, df

extract_featuresets('MMM')


# now do the machine learning with classification
# for a single ticker sym
#
def do_machine_learning(ticker_sym):
    X, y, df = extract_featuresets(ticker_sym)
    
# create our training and testing
#

# old deprecated version of train_test_split
#
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
#                                                                         y,
#                                                                         test_size = TEST_SIZE)

# new model_selection method of train, test, split
#
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=0)
    
# define our classifier where X is our % change
# data for all the companies, y is our target:
# -1, 0, +1 (Buy, Sell, Hold)
#
# our classifier: clf can also pickled for later re-use
#

#    clf = neighbors.KNeighborsClassifier()
    
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn',  neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    
    clf.fit(X_train, y_train)
#    confidence = clf.score(X_test, y_test)
    confidence = mean(cross_val_score(clf, X, y, cv=2))

    print(ticker_sym, ' Accuracy: ',confidence)
   
#    predictions = clf.predict(X_test)

    predictions = cross_val_predict(clf, X, y, cv=2)

    print(ticker_sym, ' Predicted class counts:',Counter(predictions))
    
#    return confidence
    return

do_machine_learning('MMM')


#from sklearn.model_selection import cross_val_score
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#
#from sklearn.model_selection import cross_val_predict
#predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)


# Output for 'BAC'
# Data Spread:  Counter({'1': 1797, '-1': 1651, '0': 829})
# Accuracy:  0.405607476636
# predicted class counts: Counter({-1: 620, 1: 423, 0: 27})
# Predicted Spread:  Counter({-1: 620, 1: 423, 0: 27})


# now do the machine learning with classification
# for all SP500 ticker syms
#
def do_machine_learning_for_all_SP500():
    
    with open(DATA_PATH + SP500_SYMS_FILE,"rb") as f:
        ticker_syms = pickle.load(f)

    accuracies = []
    for count,ticker_sym in enumerate(ticker_syms):

        if count%10==0:
            print('#: ', count)
        
        accuracy = do_machine_learning(ticker_sym)
        accuracies.append(accuracy)
#        print("{} accuracy: {}. Average accuracy:{}".format(ticker_sym, accuracy, mean(accuracies)))
        print(ticker_sym, ' Accuracy: ', accuracy, 'Average accuracy: ', mean(accuracies))
    
do_machine_learning_for_all_SP500()

# output:
#
"""
#:  0
MMM  Data Class Counts:  Counter({'1': 1651, '-1': 1338, '0': 1288})
MMM  Predicted class counts: Counter({-1: 392, 0: 364, 1: 314})
MMM  Accuracy:  0.385046728972 Average accuracy:  0.385046728972
ABT  Data Class Counts:  Counter({'1': 1690, '-1': 1483, '0': 1104})
ABT  Predicted class counts: Counter({-1: 597, 1: 355, 0: 118})
ABT  Accuracy:  0.361682242991 Average accuracy:  0.373364485981
ABBV  Data Class Counts:  Counter({'0': 3428, '1': 484, '-1': 365})
ABBV  Predicted class counts: Counter({0: 972, -1: 78, 1: 18})
ABBV  Accuracy:  0.824906367041 Average accuracy:  0.523878446335
ACN  Data Class Counts:  Counter({'1': 1772, '-1': 1420, '0': 1085})
ACN  Predicted class counts: Counter({-1: 501, 1: 448, 0: 119})
ACN  Accuracy:  0.40543071161 Average accuracy:  0.494266512654
ATVI  Data Class Counts:  Counter({'1': 2101, '-1': 1805, '0': 371})
ATVI  Predicted class counts: Counter({-1: 582, 1: 488})
ATVI  Accuracy:  0.471028037383 Average accuracy:  0.489618817599
"""
