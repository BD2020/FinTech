# Import Python libraries
#
import pandas as pd
import numpy as np
import csv
pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn: statistical data visualization
%matplotlib inline
plt.rcParams['figure.figsize'] = (12,8)

DATA_PATH = 'C:\\ML_Data\\_Lending Club\\'

# Lending Club data is at:
# https://www.lendingclub.com/info/download-data.action
#
# 2007-2011_LoanStats.csv
# 2007-2012_RejectStats.csv
# LCDataDictionary.xls
# 
# LoanStats_2007-2011
# LoanStats_2014

# skip row 1 so pandas can parse the data properly
#
loans_2007 = pd.read_csv(DATA_PATH + 'LoanStats_2007-2011.csv', 
                         skiprows=1, 
                         low_memory=False)

"""

Here are the column descriptions and example data for loans 2007-2011:

name 	dtypes 	first value 	description
0 	id 	object 	1077501 	A unique LC assigned ID for the loan listing.
1 	member_id 	float64 	1.2966e+06 	A unique LC assigned Id for the borrower member.
2 	loan_amnt 	float64 	5000 	The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
3 	funded_amnt 	float64 	5000 	The total amount committed to that loan at that point in time.
4 	funded_amnt_inv 	float64 	4975 	The total amount committed by investors for that loan at that point in time.
5 	term 	object 	36 months 	The number of payments on the loan. Values are in months and can be either 36 or 60.
6 	int_rate 	object 	10.65% 	Interest Rate on the loan
7 	installment 	float64 	162.87 	The monthly payment owed by the borrower if the loan originates.
8 	grade 	object 	B 	LC assigned loan grade
9 	sub_grade 	object 	B2 	LC assigned loan subgrade
10 	emp_title 	object 	NaN 	The job title supplied by the Borrower when applying for the loan.*
11 	emp_length 	object 	10+ years 	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
12 	home_ownership 	object 	RENT 	The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
13 	annual_inc 	float64 	24000 	The self-reported annual income provided by the borrower during registration.
14 	verification_status 	object 	Verified 	Indicates if income was verified by LC, not verified, or if the income source was verified
15 	issue_d 	object 	Dec-2011 	The month which the loan was funded
16 	loan_status 	object 	Fully Paid 	Current status of the loan
17 	pymnt_plan 	object 	n 	Indicates if a payment plan has been put in place for the loan
18 	purpose 	object 	credit_card 	A category provided by the borrower for the loan request.
19 	title 	object 	Computer 	The loan title provided by the borrower
20 	zip_code 	object 	860xx 	The first 3 numbers of the zip code provided by the borrower in the loan application.
21 	addr_state 	object 	AZ 	The state provided by the borrower in the loan application
22 	dti 	float64 	27.65 	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
23 	delinq_2yrs 	float64 	0 	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
24 	earliest_cr_line 	object 	Jan-1985 	The month the borrower's earliest reported credit line was opened
25 	inq_last_6mths 	float64 	1 	The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
26 	open_acc 	float64 	3 	The number of open credit lines in the borrower's credit file.
27 	pub_rec 	float64 	0 	Number of derogatory public records
28 	revol_bal 	float64 	13648 	Total credit revolving balance
29 	revol_util 	object 	83.7% 	Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
30 	total_acc 	float64 	9 	The total number of credit lines currently in the borrower's credit file
31 	initial_list_status 	object 	f 	The initial listing status of the loan. Possible values are – W, F
32 	out_prncp 	float64 	0 	Remaining outstanding principal for total amount funded
33 	out_prncp_inv 	float64 	0 	Remaining outstanding principal for portion of total amount funded by investors
34 	total_pymnt 	float64 	5863.16 	Payments received to date for total amount funded
35 	total_pymnt_inv 	float64 	5833.84 	Payments received to date for portion of total amount funded by investors
36 	total_rec_prncp 	float64 	5000 	Principal received to date
37 	total_rec_int 	float64 	863.16 	Interest received to date
38 	total_rec_late_fee 	float64 	0 	Late fees received to date
39 	recoveries 	float64 	0 	post charge off gross recovery
40 	collection_recovery_fee 	float64 	0 	post charge off collection fee
41 	last_pymnt_d 	object 	Jan-2015 	Last month payment was received
42 	last_pymnt_amnt 	float64 	171.62 	Last total payment amount received
43 	last_credit_pull_d 	object 	Dec-2016 	The most recent month LC pulled credit for this loan
44 	collections_12_mths_ex_med 	float64 	0 	Number of collections in 12 months excluding medical collections
45 	policy_code 	float64 	1 	publicly available policy_code=1\nnew products not publicly available policy_code=2
46 	application_type 	object 	INDIVIDUAL 	Indicates whether the loan is an individual application or a joint application with two co-borrowers
47 	acc_now_delinq 	float64 	0 	The number of accounts on which the borrower is now delinquent.
48 	chargeoff_within_12_mths 	float64 	0 	Number of charge-offs within 12 months
49 	delinq_amnt 	float64 	0 	The past-due amount owed for the accounts on which the borrower is now delinquent.
50 	pub_rec_bankruptcies 	float64 	0 	Number of public record bankruptcies
51 	tax_liens 	float64 	0 	Number of tax liens
"""

half_count = len(loans_2007) / 2

# Drop any column with more than 50% missing values
#
loans_2007 = loans_2007.dropna(thresh=half_count,axis=1)

# These columns are not useful for our purposes
#
loans_2007 = loans_2007.drop(['url','desc'],axis=1)      

loans_2007.head(3)

loans_2007.shape

# Convert the LC Data Dictionary from .xls to .csv format
#
#data_dictionary = pd.read_csv('C:\\ML_Data\\_Lending Club\\LCDataDictionary.csv')

data_dictionary = pd.read_excel(DATA_PATH + 'LCDataDictionary.xlsx')

# df = pandas.read_excel(open('your_xls_xlsx_filename','rb'), sheetname='Sheet 1')

#print(data_dictionary.shape[0])
#print(data_dictionary.columns.tolist())

# rename the data dictionary column headers
#

data_dictionary = data_dictionary.rename(columns={'LoanStatNew': 'name',
                                                  'Description': 'description'})

data_dictionary.head(5)

# join the first row of loans_2007 to the data_dictionary 
# DataFrame to give us a preview DataFrame with the following columns:
#
# name - contains the column names of loans_2007.
# dtypes - contains the data types of the loans_2007 columns.
# first value - contains the values of loans_2007 first row.
# description - explains what each column in loans_2007 represents
#
loans_2007_dtypes = pd.DataFrame(loans_2007.dtypes,columns=['dtypes'])
loans_2007_dtypes = loans_2007_dtypes.reset_index()
loans_2007_dtypes['name'] = loans_2007_dtypes['index']
loans_2007_dtypes = loans_2007_dtypes[['name','dtypes']]

loans_2007_dtypes['first value'] = loans_2007.loc[0].values
preview = loans_2007_dtypes.merge(data_dictionary, on='name',how='left')

#preview.head()

#preview[:19]

# a column is considered leaking information when 
# especially it won’t be available at the time we 
# use our model - in this case when we use our model on future loans
#
# We don't want our modeling to use any 'leaking'
# information, so - we can drop 'leaky' data columns
# from loans_2007

#print(data_dictionary.columns.tolist())

# the following features can be removed:
#
# id - randomly generated field by Lending Club for unique identification purposes only.
# member_id - also randomly generated field by Lending Club for identification purposes only.
# funded_amnt - leaks information from the future(after the loan is already started to be funded).
# funded_amnt_inv - also leaks data from the future.
# sub_grade - contains redundant information that is already in the grade column (more below).
# int_rate - also included within the grade column.
# emp_title - requires other data and a lot of processing to become potentially useful
# issued_d - leaks data from the future.
#

# grading: it segments borrowers based on their credit score 
# and other behaviors, which is we should keep the grade column 
# and drop interest int_rate and sub_grade

# Drop this first list un-needed columns:
#
drop_list = ['id','member_id','funded_amnt','funded_amnt_inv',
             'int_rate','sub_grade','emp_title','issue_d']

loans_2007 = loans_2007.drop(drop_list,axis=1)


# Let’s analyze the next 19 columns:
#
#preview[19:38]

# We can drop the following columns:
#
# zip_code - mostly redundant with the addr_state column 
#          since only the first 3 digits of the 5 digit zip code are visible.
# out_prncp - leaks data from the future.
# out_prncp_inv - also leaks data from the future.
# total_pymnt - also leaks data from the future.
# total_pymnt_inv - also leaks data from the future.
#
drop_cols = [ 'zip_code','out_prncp','out_prncp_inv',
             'total_pymnt','total_pymnt_inv']
loans_2007 = loans_2007.drop(drop_cols, axis=1)


# preview[38:]

# drop the following, all of which leak data from the future:
#
# total_rec_prncp
# total_rec_int
# total_rec_late_fee
# recoveries
# collection_recovery_fee
# last_pymnt_d
# last_pymnt_amnt
#
drop_cols = ['total_rec_prncp','total_rec_int', 'total_rec_late_fee',
             'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
             'last_pymnt_amnt']

loans_2007 = loans_2007.drop(drop_cols, axis=1)

#print(loans_2007['fico_range_low'].unique())
#print(loans_2007['fico_range_high'].unique())

fico_columns = ['fico_range_high','fico_range_low']

print(loans_2007.shape[0])
loans_2007.dropna(subset=fico_columns,inplace=True)
print(loans_2007.shape[0])

#loans_2007[fico_columns].plot.hist(alpha=0.5,bins=20);

# create a column for the average of fico_range_low 
# and fico_range_high columns and name it fico_average

loans_2007['fico_average'] = (loans_2007['fico_range_high'] + loans_2007['fico_range_low']) / 2

cols = ['fico_range_low','fico_range_high','fico_average']
loans_2007[cols].head()

# Now that we have the mean calculated columns, 
# we can drop fico_range_low, fico_range_high, 
# last_fico_range_low, and last_fico_range_high columns

drop_cols = ['fico_range_low','fico_range_high','last_fico_range_low',
             'last_fico_range_high']
loans_2007 = loans_2007.drop(drop_cols, axis=1)
#loans_2007.shape

# we've now been able to reduce the number of columns from 56 to 33
#

# decide the appropriate column to use as a target column 
# for modeling - with the main goal being: to predict who 
# will pay off a loan and who will default.
#
# loan_status is the only field in the main dataset that 
# describes a loan status, so let’s use this column as the target column

preview[preview.name == 'loan_status']

# this column contains text values that need 
# to be converted to numerical values to be able 
# use for training a model

# use the DataFrame method value_counts() to return 
# the frequency of the unique values in the loan_status column.
#
loans_2007["loan_status"].value_counts()

# loan status has nine different possible values
#

# Let’s learn about these unique values to determine 
# the ones that best describe the final outcome of a loan, 
# and also the kind of classification problem we’ll be dealing with
#

# Can read about most of the different loan statuses on 
# the Lending Club website as well as these posts on the 
# Lend Academy and Orchard forums
#
# http://kb.lendingclub.com/investor/articles/Investor/What-do-the-different-Note-statuses-mean/
#
# http://www.orchardplatform.com/blog/2014520loans-issued-under-previous-credit-policies/
#
#
status_meaning = [
    "Loan has been fully paid off.",
    "Loan for which there is no longer a reasonable expectation of further payments.",
    "While the loan was paid off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
    "While the loan was charged off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
    "Loan is up to date on current payments.",
    "The loan is past due but still in the grace period of 15 days.",
    "Loan hasn't been paid in 31 to 120 days (late on the current payment).",
    "Loan hasn't been paid in 16 to 30 days (late on the current payment).",
    "Loan is defaulted on and no payment has been made for more than 121 days."]

status, count = loans_2007["loan_status"].value_counts().index, loans_2007["loan_status"].value_counts().values

loan_statuses_explanation = pd.DataFrame({'Loan Status': status,
                                          'Count': count,
                                          'Meaning': status_meaning})[['Loan Status','Count','Meaning']]
loan_statuses_explanation


# our goal is to build a machine learning model 
# that can learn from past loans in trying to 
# predict which loans will be paid off and which won’t.
#
# only the Fully Paid and Charged Off values 
# describe the final outcome of a loan

# we should use only samples where the loan_status 
# column is 'Fully Paid' or 'Charged Off'
#
# We’re not interested in any loan_status that indicates
# that the loan is ongoing or in progress
#
# we’re interested in being able to predict which 
# of these 2 loan_status values a loan will fall under, 
# so we can treat the problem as binary classification
#
# remove all the loans that don’t have a loan_status
# of either 'Fully Paid' or 'Charged Off' , then transform 
# the 'Fully Paid' values to 1 for the positive case 
# and the 'Charged Off' values to 0 for the negative case
#
# Out of the ~42,000 rows we have, 3,000 will be removed
#

# First, redefine our loans_2007 data frame to contain
# only rows where loan_status is "Fully Paid" or "Charged Off"

loans_2007 = loans_2007[(loans_2007["loan_status"] == "Fully Paid") |
                            (loans_2007["loan_status"] == "Charged Off")]


# then redefine loan_status to now be either 1 or 0
#
mapping_dictionary = {"loan_status": {"Fully Paid": 1, "Charged Off": 0}}
loans_2007 = loans_2007.replace(mapping_dictionary)


#
loans_2007.head()
"""
 	loan_amnt 	term 	installment 	grade 	emp_length 	home_ownership 	annual_inc 	verification_status 	loan_status 	pymnt_plan 	purpose 	title 	addr_state 	dti 	delinq_2yrs 	earliest_cr_line 	inq_last_6mths 	open_acc 	pub_rec 	revol_bal 	revol_util 	total_acc 	initial_list_status 	last_credit_pull_d 	collections_12_mths_ex_med 	policy_code 	application_type 	acc_now_delinq 	chargeoff_within_12_mths 	delinq_amnt 	pub_rec_bankruptcies 	tax_liens 	fico_average
0 	5000.0 	36 months 	162.87 	B 	10+ years 	RENT 	24000.0 	Verified 	1 	n 	credit_card 	Computer 	AZ 	27.65 	0.0 	Jan-1985 	1.0 	3.0 	0.0 	13648.0 	83.7% 	9.0 	f 	Sep-2016 	0.0 	1.0 	INDIVIDUAL 	0.0 	0.0 	0.0 	0.0 	0.0 	737.0
1 	2500.0 	60 months 	59.83 	C 	< 1 year 	RENT 	30000.0 	Source Verified 	0 	n 	car 	bike 	GA 	1.00 	0.0 	Apr-1999 	5.0 	3.0 	0.0 	1687.0 	9.4% 	4.0 	f 	Sep-2016 	0.0 	1.0 	INDIVIDUAL 	0.0 	0.0 	0.0 	0.0 	0.0 	742.0
2 	2400.0 	36 months 	84.33 	C 	10+ years 	RENT 	12252.0 	Not Verified 	1 	n 	small_business 	real estate business 	IL 	8.72 	0.0 	Nov-2001 	2.0 	2.0 	0.0 	2956.0 	98.5% 	10.0 	f 	Sep-2016 	0.0 	1.0 	INDIVIDUAL 	0.0 	0.0 	0.0 	0.0 	0.0 	737.0
3 	10000.0 	36 months 	339.31 	C 	10+ years 	RENT 	49200.0 	Source Verified 	1 	n 	other 	personel 	CA 	20.00 	0.0 	Feb-1996 	1.0 	10.0 	0.0 	5598.0 	21% 	37.0 	f 	Apr-2016 	0.0 	1.0 	INDIVIDUAL 	0.0 	0.0 	0.0 	0.0 	0.0 	692.0
5 	5000.0 	36 months 	156.46 	A 	3 years 	RENT 	36000.0 	Source Verified 	1 	n
"""


filtered_loans = loans_2007

# Visualize the Target Column Outcomes
#
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='loan_status', data=filtered_loans, ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")

filtered_loans.loan_status.value_counts().plot(x=None,
                                               y=None, 
                                               kind='pie', 
                                               ax=axs[1],
                                               autopct='%1.2f%%')

axs[1].set_title("Percentage of each Loan status")
plt.show()


# a significant number of borrowers in our dataset
# paid off their loan - 85.59% of loan borrowers 
# paid off amount borrowed, while 14.41% unfortunately 
# defaulted. From our loan data it is these ‘defaulters’ 
# that we’re more interested in filtering out as much 
# as possible to reduce loses on investment returns.
#

# look for any columns that contain only one unique 
# value and remove them. These columns won’t be useful 
# for the model since they don’t add any information 
# to each loan application

# nunique() returns the number of unique values, 
# excluding any null values
#
loans_2007 = loans_2007.loc[:,loans_2007.apply(pd.Series.nunique) != 1]

loans_2007.shape

# there may be some columns with more than one unique 
# values but one of the values has insignificant frequency 
# in the dataset. Let’s find out and drop such column(s)
#
for col in loans_2007.columns:
    if (len(loans_2007[col].unique()) < 4):
        pass
#        print(loans_2007[col].value_counts())
#        print()

"""
Output:

 36 months    29096
 60 months    10143
Name: term, dtype: int64

Not Verified       16845
Verified           12526
Source Verified     9868
Name: verification_status, dtype: int64

1    33586
0     5653
Name: loan_status, dtype: int64

n    39238
y        1
Name: pymnt_plan, dtype: int64
"""

# we can drop the pymnt_plan column since
# it almost completely all: 'y'
#
print(loans_2007.shape[1])
loans_2007 = loans_2007.drop('pymnt_plan', axis=1)
print("We've been able to reduced the features to => {}".format(loans_2007.shape[1]))

# lets save our work a CSV file
#
# loans_2007.to_csv("processed_data/filtered_loans_2007.csv",index=False)

loans_2007.to_csv(DATA_PATH + 'LoanStatsFiltered_2007-2011.csv', 
                  index=False,
                  encoding='latin-1')

#loans_2007.to_excel(DATA_PATH + 'LoanStatsFiltered_2007-2011.xlsx')

"""
 ew = pandas.ExcelWriter('test.xlsx',options={'encoding':'utf-8'})
 sampleList = ['Miño', '1', '2', 'señora']
 dataframe = pandas.DataFrame(sampleList)
 dataframe.to_excel(ew)
 ew.save()
"""

# now prepare the LoanStatsFiltered_2007-2011.csv data 
# for machine learning. Focusing on handling missing 
# values, converting categorical columns to numeric 
# columns and removing any other extraneous columns
# 
# We need to handle missing values and categorical 
# features before feeding the data into a machine 
# learning algorithm, because the mathematics 
# underlying most machine learning models assumes 
# that the data is numerical and contains no missing values
#
# To reinforce this requirement, scikit-learn returns 
# an error if we try to train a model using data that 
# contain missing values or non-numeric values when 
# working with models like linear regression and logistic regression
#
# Next:
# 
#    Handle Missing Values
#    Investigate Categorical Columns
#        Convert Categorical Columns To Numeric Features
#            Map Ordinal Values To Integers
#            Encode Nominal Values As Dummy Variables
#

# First, re-load the filtered loan stats data
#

filtered_loans = pd.read_csv(DATA_PATH + 'LoanStatsFiltered_2007-2011.csv', encoding='latin-1') 


print(filtered_loans.shape)
filtered_loans.head()

# return the number of missing values across the DataFrame by:
#
# First, use the Pandas DataFrame method isnull() 
# to return a DataFrame containing Boolean values:
#   True if the original value is null
#   False if the original value isn’t null
#
# Then, use the Pandas DataFrame method sum() to 
# calculate the number of null values in each column
#
null_counts = filtered_loans.isnull().sum()
# print("Number of null values in each column:\n{}".format(null_counts))

"""
Number of null values in each column:
loan_amnt                 0
term                      0
installment               0
grade                     0
emp_length                0
home_ownership            0
annual_inc                0
verification_status       0
loan_status               0
purpose                   0
title                    10
addr_state                0
dti                       0
delinq_2yrs               0
earliest_cr_line          0
inq_last_6mths            0
open_acc                  0
pub_rec                   0
revol_bal                 0
revol_util               50
total_acc                 0
last_credit_pull_d        2
pub_rec_bankruptcies    697
fico_average              0
dtype: int64
"""

# remove columns entirely where more than 1% (392) 
# of the rows for that column contain a null value. 
# n addition, we’ll remove the remaining rows containing null values
#

# we’ll keep the title and revol_util columns, 
# just removing rows containing missing values, 
# but drop the pub_rec_bankruptcies column entirely 
# since more than 1% of the rows have a missing value for this column
#
# Use the drop method to remove the pub_rec_bankruptcies column from filtered_loans.
# Use the dropna method to remove all rows from filtered_loans containing any missing values.

filtered_loans = filtered_loans.drop("pub_rec_bankruptcies",axis=1)
filtered_loans = filtered_loans.dropna()

# next is to have all the columns as numeric columns 
# (int or float data type), and containing no missing values.
#
print("Data types and their frequency\n{}".format(filtered_loans.dtypes.value_counts()))

"""
Data types and their frequency
float64    11
object     11
int64       1
"""

# select just the object columns using the DataFrame 
# method select_dtype, then display a sample row to 
# get a better sense of how the values in each column are formatted.
#
object_columns_df = filtered_loans.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])

"""
term                     36 months
grade                            B
emp_length               10+ years
home_ownership                RENT
verification_status       Verified
purpose                credit_card
title                     Computer
addr_state                      AZ
earliest_cr_line          Jan-1985
revol_util                   83.7%
last_credit_pull_d        Sep-2016
Name: 0, dtype: object
"""

# revol_util is a revolving line utilization 
# rate or the amount of credit the borrower 
# is using relative to all available credit
#

# Since revol_util is a string % amt, to
# convert to numeric:
# Use the str.rstrip() string method to strip the right trailing percent sign (%).
# use the astype() method to convert to the type float.
# Assign the new Series of float values back to revol_util
#
filtered_loans['revol_util'] = filtered_loans['revol_util'].str.rstrip('%').astype('float')

# these columns seem to represent categorical values:
#
#   home_ownership - home ownership status, can only be 1 of 4 categorical values according to the data dictionary.
#   verification_status - indicates if income was verified by Lending Club.
#   emp_length - number of years the borrower was employed upon time of application.
#   term - number of payments on the loan, either 36 or 60.
#   addr_state - borrower’s state of residence.
#   grade - LC assigned loan grade based on credit score.
#   purpose - a category provided by the borrower for the loan request.
#   title - loan title provided the borrower.

# values for both earliest_cr_line and last_credit_pull_d 
# columns contain date values that would require a 
# good amount of feature engineering for them to be potentially useful:
#
#    earliest_cr_line - The month the borrower’s earliest reported credit line was opened
#    last_credit_pull_d - The most recent month Lending Club pulled credit for this loan
#
# We’ll remove these date columns from the DataFrame

cols = ['home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state']
for name in cols:
    pass
#    print(name,':')
#    print(object_columns_df[name].value_counts(),'\n')

"""
home_ownership :
RENT        18677
MORTGAGE    17381
OWN          3020
OTHER          96
NONE            3
Name: home_ownership, dtype: int64 

grade :
B    11873
A    10062
C     7970
D     5194
E     2760
F     1009
G      309
Name: grade, dtype: int64 

verification_status :
Not Verified       16809
Verified           12515
Source Verified     9853
Name: verification_status, dtype: int64 

emp_length :
10+ years    8715
< 1 year     4542
2 years      4344
3 years      4050
4 years      3385
5 years      3243
1 year       3207
6 years      2198
7 years      1738
8 years      1457
9 years      1245
n/a          1053
Name: emp_length, dtype: int64 

term :
 36 months    29041
 60 months    10136
Name: term, dtype: int64 

"""

# Most of these coumns contain discrete categorical 
# values which we can encode as dummy variables and keep. 
# The addr_state column, however,contains too many 
# unique values, so it’s better to drop this.
#
for name in ['purpose','title']:
    pass
#    print("Unique Values in column: {}\n".format(name))
#    print(filtered_loans[name].value_counts(),'\n')

"""
Unique Values in column: purpose

debt_consolidation    18355
credit_card            5073
other                  3921
home_improvement       2944
major_purchase         2178
small_business         1792
car                    1534
wedding                 940
medical                 688
moving                  580
vacation                377
house                   372
educational             320
renewable_energy        103
Name: purpose, dtype: int64 
"""


# It appears the purpose and title columns do 
# contain overlapping information, but the purpose 
# column contains fewer discrete values and is cleaner, 
# so we’ll keep it and drop title
#
drop_cols = ['last_credit_pull_d','addr_state','title','earliest_cr_line']
filtered_loans = filtered_loans.drop(drop_cols, axis=1)


# There are two types of categorical features in the dataset 
# that need to be converted to numerical features
#
# 1. Ordinal values: these categorical values are in natural order
# that can be sorted in ascending or descending order
# For example, loans are graded: from A to G, where grade A
# is less risky than grade B, etc
#
# 2. Nominal Values: these are regular categorical values that
# can't be sorted. For example, purpose can't be ordered
#
# These are the columns we now have in our dataset:
# 
#   Ordinal Values
#     grade
#     emp_length
#   Nominal Values _ home_ownership
#     verification_status
#     purpose
#     term
#

# First map both grade and emp_length to appropriate 
# numeric values
#


mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
 
    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}

filtered_loans = filtered_loans.replace(mapping_dict)
filtered_loans[['emp_length','grade']].head()

# The approach to converting nominal features into 
# numerical features is to encode them as dummy variables. 
# The process will be:
#
#  Use pandas’ get_dummies() method to return a new DataFrame 
#      containing a new column for each dummy variable
#  Use the concat() method to add these dummy columns back to the original DataFrame
#  Then drop the original columns entirely using the drop method
#
nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(filtered_loans[nominal_columns])
filtered_loans = pd.concat([filtered_loans, dummy_df], axis=1)
filtered_loans = filtered_loans.drop(nominal_columns, axis=1)

filtered_loans.head()

# output:
# loan_amnt 	installment 	grade 	emp_length 	annual_inc 	loan_status 	dti 	delinq_2yrs 	inq_last_6mths 	open_acc 	pub_rec 	revol_bal 	revol_util 	total_acc 	fico_average 	home_ownership_MORTGAGE 	home_ownership_NONE 	home_ownership_OTHER 	home_ownership_OWN 	home_ownership_RENT 	verification_status_Not Verified 	verification_status_Source Verified 	verification_status_Verified 	purpose_car 	purpose_credit_card 	purpose_debt_consolidation 	purpose_educational 	purpose_home_improvement 	purpose_house 	purpose_major_purchase 	purpose_medical 	purpose_moving 	purpose_other 	purpose_renewable_energy 	purpose_small_business 	purpose_vacation 	purpose_wedding 	term_ 36 months 	term_ 60 months
# 0 	5000.0 	162.87 	2 	10 	24000.0 	1 	27.65 	0.0 	1.0 	3.0 	0.0 	13648.0 	83.7 	9.0 	737.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	1.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0
# 1 	2500.0 	59.83 	3 	0 	30000.0 	0 	1.00 	0.0 	5.0 	3.0 	0.0 	1687.0 	9.4 	4.0 	742.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	1.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0
# 2 	2400.0 	84.33 	3 	10 	12252.0 	1 	8.72 	0.0 	2.0 	2.0 	0.0 	2956.0 	98.5 	10.0 	737.0 	0.0 	0.0 	0.0 	0.0 	1.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	1.0 	0.0
# 3 	10000.0 	339.31 	3 	10 	49200.0 	1 	20.00 	0.0 	1.0 	10.0 	0.0 	5598.0 	21.0 	37.0 	692.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0
# 4 	5000.0 	156.46 	1 	3 	36000.0 	1 	11.20 	0.0 	3.0 	9.0 	0.0

# Check that all features are of the same length, 
# contain no null value, and are numericals
#
filtered_loans.info()

# output:
#
"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 39177 entries, 0 to 39238
Data columns (total 39 columns):
loan_amnt                              39177 non-null float64
installment                            39177 non-null float64
grade                                  39177 non-null int64
emp_length                             39177 non-null int64
annual_inc                             39177 non-null float64
loan_status                            39177 non-null int64
dti                                    39177 non-null float64
delinq_2yrs                            39177 non-null float64
inq_last_6mths                         39177 non-null float64
open_acc                               39177 non-null float64
pub_rec                                39177 non-null float64
revol_bal                              39177 non-null float64
revol_util                             39177 non-null float64
total_acc                              39177 non-null float64
fico_average                           39177 non-null float64
home_ownership_MORTGAGE                39177 non-null float64
home_ownership_NONE                    39177 non-null float64
home_ownership_OTHER                   39177 non-null float64
home_ownership_OWN                     39177 non-null float64
home_ownership_RENT                    39177 non-null float64
verification_status_Not Verified       39177 non-null float64
verification_status_Source Verified    39177 non-null float64
verification_status_Verified           39177 non-null float64
purpose_car                            39177 non-null float64
purpose_credit_card                    39177 non-null float64
purpose_debt_consolidation             39177 non-null float64
purpose_educational                    39177 non-null float64
purpose_home_improvement               39177 non-null float64
purpose_house                          39177 non-null float64
purpose_major_purchase                 39177 non-null float64
purpose_medical                        39177 non-null float64
purpose_moving                         39177 non-null float64
purpose_other                          39177 non-null float64
purpose_renewable_energy               39177 non-null float64
purpose_small_business                 39177 non-null float64
purpose_vacation                       39177 non-null float64
purpose_wedding                        39177 non-null float64
term_ 36 months                        39177 non-null float64
term_ 60 months                        39177 non-null float64
dtypes: float64(36), int64(3)
memory usage: 12.0 MB
"""

# Now save out filtered loan data for upcoming
# binary classification machine learning analysis
#
filtered_loans.to_csv(DATA_PATH + 'LoanStatsCleaned_2007-2011.csv', 
                  index=False,
                  encoding='latin-1')

print("Done with cleaning Lending Club loan stats data")
