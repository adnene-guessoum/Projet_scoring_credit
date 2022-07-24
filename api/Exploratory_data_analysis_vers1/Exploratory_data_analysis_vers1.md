<h1>EXPLORATORY DATA ANALYSIS - HOME CREDIT DEFAULT RISK</h1>

<h2>Introduction</h2>

<b>Home Credit</b> is an international consumer finance provider which operates
in 9 countries. It provides point of sales loans, cash loans and revolving
loans to underserved borrowers. The term undeserved borrower here refers
to those who earn regular income from their job or businesses, but have
little or no credit history and find it difficult to get credits from other
traditional lending organizations. They believe that the credit history should
not be a barrier for a borrower to fulfill their dreams.<br><br>
Over 22 years of track record, they have accumulated a large amount of
borrower behavioural data which they leverage to provide financial
assistance to such customers. They have built predictive models that help
them to efficiently analyze the risk associated with a given client and also
estimate the safe credit amount to be lent to customers, even with no credit
history.
<br><br>


<h2>Dataset </h2>

<h3>Data Overview</h3>

<b>Home Credit Group</b> has generously provided a vast dataset to motivate machine learning engineers and researchers to come up with techniques to build a predictive model for default risk prediction. Generally, the data in the field of Finances tend to be very much variant and collecting such data can be very tedious task, but in this case, Home Credit has done most of the heavy lifting to provide us as clean of a data as possible. <br><br>
The dataset provided contains a vast number of details about the borrower.
It is separated into several relational tables, which contain applicants’ static
data such as their gender, age, number of family members, occupation, and
other necessary fields, applicant’s previous credit history obtained from the
credit bureau department, and the applicant’s past credit history within the
Home Credit Group itself. The dataset is an imbalanced dataset, where the negative class dominates
the positive class, as there are only a few number of defaulters among all
the applicants.<br><br>
The Dataset can be downloaded from the Kaggle link: <a href = https://www.kaggle.com/c/home-credit-default-risk/data>Home Credit Default Risk Dataset</a><br><br>

<h3>Data Specifications</h3>
<pre>
There are 10 .csv files in total. They are:

HomeCredit_columns_description.csv - 36.51 KB
POS_CASH_balance.csv -               374.51 MB
application_test.csv -               25.34 MB
application_train.csv -              158.44 MB
bureau.csv -                         162.14 MB
bureau_balance.csv -                 358.19 MB
credit_card_balance.csv -            404.91 MB
installments_payments.csv -          689.62 MB
previous_application.csv -           386.21 MB
sample_submission.csv -              523.63 KB</pre>

<h3>Structure of Relational Tables

<br><br>
<h4>Brief Description of Each Table</h4>

<h5>application_{train|test}.csv</h5>

<ul><li>This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
<li>Static data for all applications. One row represents one loan in our data sample. </ul>

<h5>bureau.csv</h5>

<ul><li>All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
<li>For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
    </ul>
    
<h5>bureau_balance.csv</h5>

<ul><li>Monthly balances of previous credits in Credit Bureau.
<li>This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
    </ul>
    
<h5>POS_CASH_balance.csv</h5>

<ul><li>Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
<li>This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
    </ul>
    
<h5>credit_card_balance.csv</h5>

<ul><li>Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
    </ul>

<h5>previous_application.csv</h5>

<ul><li>All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.
    </ul>

<h5>installments_payments.csv</h5>

<ul><li>Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
There is a) one row for every payment that was made plus b) one row each for missed payment.
One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
    </ul>

<h5>HomeCredit_columns_description.csv</h5>

<ul><li>This file contains descriptions for the columns in the various data files.
    </ul>
Source: Home Credit Group (<a href = https://www.kaggle.com/c/home-credit-default-risk/data>Kaggle</a>)

<h2>Preliminary Steps</h2>

### Loading the libraries and modules

Let's start by loading the essential libraries and modules. We will also set the max columns and max rows display limit to None for Pandas DataFrames to be able to see the whole DataFrame.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from analysis_functions import *

import warnings
warnings.filterwarnings('ignore')


import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime

#metrics
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#pandas DataFrame column and row display limits
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

#for 100% jupyter notebook cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

```


<style>.container { width:100% !important; }</style>


<h2>EDA</h2>

For the data analysis, we will follow following steps:

<ol><li>For each table, we will first check basic stats like the number of records in tables, number of features, number of NaN values, etc.
    <li>Next, we will explore some of the features with respect to the target variable for each table.
        We will be employing the following plots
        <ul><li>For Categorical Features, we will mostly be using <b>Bar Plots</b> and <b>Pie Charts</b>.
            <li>For continuous/numeric features, we will be using <b>Box-Plots</b>, <b>PDFs</b>, <b>CDF</b>, and <b>Violin-Plots</b>.   
        </ul>
    <li>We will be drawing observations from each plot and note important insights generated from the plots.
        </ol>
        

<h5>Note 1:</h5>
<ul><li>For Categorical Variables, we will be plotting the Bar Plots and Pie Plots. The use of Bar and Pie Plot will be based on the Number of unique categories present in a feature. If a feature has too many categories, displaying them on Pie Plots can be cumbersome, and Bar Plot does a better job of showing each category. Also Bar Plot will be preferred when the proportions of all the categories are more or less the same to identify small differences</li>
    <li>We will be following the below mentioned strategy for plotting for Categorical Features in the whole notebook:
        <ul><li>First, we will be plotting the distribution of each category in the whole data, in the first subplot.</li>
            <li>Next, in the second subplot, we will be plotting the Percentage of Defaulters from each category, i.e. with Target = 1. </li>
            <li>For example, say if a feature contains Gender, viz. Male and Female, so for first subplot we will plot the number of occurrences of each of Male and Female in our dataset. <br>In the second subplot, we will be plotting that out of the counts of Male present in the dataset, how many or what percentage of Males were found to Default. Similarly we will do this for Female. <br>
                This is being done because there will be few categories which will be dominant over others, and their Default characteristics would not be identifiable if we look at just the counts.</li></ul></li></ul>

<h5>Note 2:</h5>

For the analysing the continuous variables, we will use four kinds of plots as and when needed, which are Distplot, CDF, Box-plots and Violin Plots.
<ul><li><b>DistPlots:</b><br>The distplot will be used when we want to see the PDFs of the continuous variable. This PDF will help us to analyze where most of our data is lying.</li>
    <li><b>CDF:</b><br>CDFs can be used as an extention to PDFs to see what percentage of points lie below a certain threshold value. This would give us a good estimate of the distribution of majority of data.
    <li><b>Box-Plots:</b><br>Box-plots are helpful when we want to analyze the whole range of values that our continuous variable has. It shows the 25th, 50th and 75th percentile in a single plot. Moreover, it also gives some ideas related to presence of outliers in a given set of values.</li>
    <li><b>Violin-Plots:</b><br>Violin-plots tend to combine the features of both Distplots and Box-Plots. Vertically they mimic the box-plot and show the quantiles, range of values, and horizontally they show the PDF of the continuous variable.

### Loading all the Tables


```python
def load_all_tables(directory_path = '', verbose = True):
    
    '''
    Function to load all the tables required
    
    Input:
        directory_path: str, default = ''
            Path of directory in which tables are stored in
        verbose: bool, default = True
            Whether to keep verbosity or not
        
    '''
    
    if verbose:
        print("Loading all the tables...")
        #start = datetime.now()
    
    #making all the variables global to be used anywhere in the notebook
    global application_train, application_test, bureau, bureau_balance, cc_balance, installments_payments, POS_CASH_balance, previous_application
    
    application_train = pd.read_csv(directory_path + 'application_train.csv')
    if verbose:
        print("Loaded 1 table.")
       
    application_test = pd.read_csv(directory_path + 'application_test.csv')
    if verbose:
        print("Loaded 2 tables.")

    bureau = pd.read_csv(directory_path + 'bureau.csv')
    if verbose:
        print("Loaded 3 tables.")

    bureau_balance = pd.read_csv(directory_path + 'bureau_balance.csv')
    if verbose:
        print("Loaded 4 tables.")

    cc_balance = pd.read_csv(directory_path + 'credit_card_balance.csv')
    if verbose:
        print("Loaded 5 tables.")

    installments_payments = pd.read_csv(directory_path + 'installments_payments.csv')
    if verbose:
        print("Loaded 6 tables.")

    POS_CASH_balance = pd.read_csv(directory_path + 'POS_CASH_balance.csv')
    if verbose:
        print("Loaded 7 tables.")

    previous_application = pd.read_csv(directory_path + 'previous_application.csv')
    if verbose:
        print("Loaded 8 tables.")
        print("Done.")
        #print(f'Time Taken to load 8 tables = {datetime.now() - start}')
    
```


```python
load_all_tables(directory_path = '../../')
```

    Loading all the tables...
    Loaded 1 table.
    Loaded 2 tables.
    Loaded 3 tables.
    Loaded 4 tables.
    Loaded 5 tables.
    Loaded 6 tables.
    Loaded 7 tables.
    Loaded 8 tables.
    Done.
    


```python
application_train = pd.read_csv('../../application_train.csv')
```


```python
application_test.shape
```




    (48744, 121)



### application_train.csv and application_test.csv

##### Description:

The application_train.csv table consists of static data relating to the Borrowers with labels. Each row represents one loan application. <br>
The application_test.csv contains the testing dataset, and is similar to application_train.csv, except that the TARGET column has been omitted, which has to be predicted with the help of Statistical and Machine Learning Predictive Models.

#### Basic Stats


```python
print('-'*100)
print(f'The shape of application_train.csv is: {application_train.shape}')
print('-'*100)
print(f'Number of duplicate values in application_train: {application_train.shape[0] - application_train.duplicated().shape[0]}')
print('-'*100)
display(application_train.head())
```

    ----------------------------------------------------------------------------------------------------
    The shape of application_train.csv is: (307511, 122)
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in application_train: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>LANDAREA_AVG</th>
      <th>LIVINGAPARTMENTS_AVG</th>
      <th>LIVINGAREA_AVG</th>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_MODE</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_MODE</th>
      <th>ENTRANCES_MODE</th>
      <th>FLOORSMAX_MODE</th>
      <th>FLOORSMIN_MODE</th>
      <th>LANDAREA_MODE</th>
      <th>LIVINGAPARTMENTS_MODE</th>
      <th>LIVINGAREA_MODE</th>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>APARTMENTS_MEDI</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>COMMONAREA_MEDI</th>
      <th>ELEVATORS_MEDI</th>
      <th>ENTRANCES_MEDI</th>
      <th>FLOORSMAX_MEDI</th>
      <th>FLOORSMIN_MEDI</th>
      <th>LANDAREA_MEDI</th>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.018801</td>
      <td>-9461</td>
      <td>-637</td>
      <td>-3648.0</td>
      <td>-2120</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
      <td>0.0247</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6192</td>
      <td>0.0143</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0369</td>
      <td>0.0202</td>
      <td>0.0190</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0252</td>
      <td>0.0383</td>
      <td>0.9722</td>
      <td>0.6341</td>
      <td>0.0144</td>
      <td>0.0000</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0377</td>
      <td>0.022</td>
      <td>0.0198</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0250</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6243</td>
      <td>0.0144</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0375</td>
      <td>0.0205</td>
      <td>0.0193</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003541</td>
      <td>-16765</td>
      <td>-1188</td>
      <td>-1186.0</td>
      <td>-291</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>School</td>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>NaN</td>
      <td>0.0959</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>0.0605</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0130</td>
      <td>0.0773</td>
      <td>0.0549</td>
      <td>0.0039</td>
      <td>0.0098</td>
      <td>0.0924</td>
      <td>0.0538</td>
      <td>0.9851</td>
      <td>0.8040</td>
      <td>0.0497</td>
      <td>0.0806</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0128</td>
      <td>0.079</td>
      <td>0.0554</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0968</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7987</td>
      <td>0.0608</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0132</td>
      <td>0.0787</td>
      <td>0.0558</td>
      <td>0.0039</td>
      <td>0.01</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-19046</td>
      <td>-225</td>
      <td>-4260.0</td>
      <td>-2531</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.555912</td>
      <td>0.729567</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008019</td>
      <td>-19005</td>
      <td>-3039</td>
      <td>-9833.0</td>
      <td>-2437</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.650442</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19932</td>
      <td>-3038</td>
      <td>-4311.0</td>
      <td>-3458</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Religion</td>
      <td>NaN</td>
      <td>0.322738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('-'*100)
print(f'The shape of application_test.csv is: {application_test.shape}')
print('-'*100)
print(f'Number of duplicate values in application_test: {application_test.shape[0] - application_test.duplicated().shape[0]}')
print('-'*100)
display(application_test.head())
```

    ----------------------------------------------------------------------------------------------------
    The shape of application_test.csv is: (48744, 121)
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in application_test: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>LANDAREA_AVG</th>
      <th>LIVINGAPARTMENTS_AVG</th>
      <th>LIVINGAREA_AVG</th>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_MODE</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_MODE</th>
      <th>ENTRANCES_MODE</th>
      <th>FLOORSMAX_MODE</th>
      <th>FLOORSMIN_MODE</th>
      <th>LANDAREA_MODE</th>
      <th>LIVINGAPARTMENTS_MODE</th>
      <th>LIVINGAREA_MODE</th>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>APARTMENTS_MEDI</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>COMMONAREA_MEDI</th>
      <th>ELEVATORS_MEDI</th>
      <th>ENTRANCES_MEDI</th>
      <th>FLOORSMAX_MEDI</th>
      <th>FLOORSMIN_MEDI</th>
      <th>LANDAREA_MEDI</th>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>568800.0</td>
      <td>20560.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018850</td>
      <td>-19241</td>
      <td>-2329</td>
      <td>-5170.0</td>
      <td>-812</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Kindergarten</td>
      <td>0.752614</td>
      <td>0.789654</td>
      <td>0.159520</td>
      <td>0.0660</td>
      <td>0.0590</td>
      <td>0.9732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1379</td>
      <td>0.125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0505</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0672</td>
      <td>0.0612</td>
      <td>0.9732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1379</td>
      <td>0.125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0526</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0666</td>
      <td>0.0590</td>
      <td>0.9732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1379</td>
      <td>0.125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>block of flats</td>
      <td>0.0392</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1740.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>99000.0</td>
      <td>222768.0</td>
      <td>17370.0</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-18064</td>
      <td>-4469</td>
      <td>-9118.0</td>
      <td>-1623</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Low-skill Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Self-employed</td>
      <td>0.564990</td>
      <td>0.291656</td>
      <td>0.432962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>663264.0</td>
      <td>69777.0</td>
      <td>630000.0</td>
      <td>NaN</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019101</td>
      <td>-20038</td>
      <td>-4458</td>
      <td>-2175.0</td>
      <td>-3503</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Transport: type 3</td>
      <td>NaN</td>
      <td>0.699787</td>
      <td>0.610991</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-856.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>315000.0</td>
      <td>1575000.0</td>
      <td>49018.5</td>
      <td>1575000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-13976</td>
      <td>-1866</td>
      <td>-2000.0</td>
      <td>-4208</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>0.525734</td>
      <td>0.509677</td>
      <td>0.612704</td>
      <td>0.3052</td>
      <td>0.1974</td>
      <td>0.9970</td>
      <td>0.9592</td>
      <td>0.1165</td>
      <td>0.32</td>
      <td>0.2759</td>
      <td>0.375</td>
      <td>0.0417</td>
      <td>0.2042</td>
      <td>0.2404</td>
      <td>0.3673</td>
      <td>0.0386</td>
      <td>0.08</td>
      <td>0.3109</td>
      <td>0.2049</td>
      <td>0.9970</td>
      <td>0.9608</td>
      <td>0.1176</td>
      <td>0.3222</td>
      <td>0.2759</td>
      <td>0.375</td>
      <td>0.0417</td>
      <td>0.2089</td>
      <td>0.2626</td>
      <td>0.3827</td>
      <td>0.0389</td>
      <td>0.0847</td>
      <td>0.3081</td>
      <td>0.1974</td>
      <td>0.9970</td>
      <td>0.9597</td>
      <td>0.1173</td>
      <td>0.32</td>
      <td>0.2759</td>
      <td>0.375</td>
      <td>0.0417</td>
      <td>0.2078</td>
      <td>0.2446</td>
      <td>0.3739</td>
      <td>0.0388</td>
      <td>0.0817</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.3700</td>
      <td>Panel</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1805.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>180000.0</td>
      <td>625500.0</td>
      <td>32067.0</td>
      <td>625500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-13040</td>
      <td>-2191</td>
      <td>-4000.0</td>
      <td>-4262</td>
      <td>16.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Business Entity Type 3</td>
      <td>0.202145</td>
      <td>0.425687</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-821.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


##### Observations and Conclusions:

<ol><li><b>application_train.csv:</b>
    <ul>
    <li>The application_train.csv file has approx 307k records, and 122 features. These features contain the personal statistics belonging to a particular customer such as his Age, Income, Type of Loan, Apartment Stats, etc.</li>
    <li>There are 307k unique SK_ID_CURR which represent unique loan applications.</li>
    <li>The TARGET field represents the Loan-Default Status, 0 stands for Non-Defaulter, and 1 for Defaulter.</li>
    </ul></li><br>
    <li><b>application_test.csv:</b><ul>
        <li>The application_test.csv file has approx 48.7k records, and 121 features. These features are exactly those which are in application_train.csv, except that these are the training sets.</li>
        <li>There are 48.7k unique SK_ID_CURR which represent unique loan applications.</li>
        <li>The TARGET column has been omitted and needs to be predicted by the help of Predictive Statistical and Machine Learning Models.</li>
        </ul></li></ol>

#### NaN Columns and Percentages


```python
plot_nan_percent(nan_df_create(application_train), 'application_train', grid = True)
```

    Number of columns having NaN values: 67 columns
    


    
![png](output_24_1.png)
    



```python
plot_nan_percent(nan_df_create(application_test), 'application_test', grid = True)
```

    Number of columns having NaN values: 64 columns
    


    
![png](output_25_1.png)
    


##### Observations and Conclusions:

<ol><li><b>application_train.csv:</b>
    <ul><li>It can be seen from the above plot that there are 67 columns out of 122 features which contain NaN values. If there were just one or two columns which had NaN values, we could have gotten away with just eliminating those columns, but for such large number of columns, we cannot remove them as is, as loss of information could be very high.<br>
    <li>We see that some columns like relating to "COMMONAREA", "NONLIVINGAPARTMENT", etc. have close to 70% missing values. We would have to come up with techniques to handle these many missing values and see what would work best for our data.
    <li>Another thing to note here is that most of the columns which have more than 50% missing values are related to the Apartments Statistics of the borrower. It is very likely that these values were not recorded during data entry, and could be optional.</ul><br>
    </li>
    <li><b>application_test.csv:</b><ul>
        <li>There are very similar number of columns with NaN values (64) as were with the application_train.</li>
        <li>The percentages of NaN values are also quite similar to the ones present in training dataset. This means that the training and test sets are pretty much of similar distribution.</li>
        </ul>

#### Distribution of Target Variable


```python
target_distribution = application_train.TARGET.value_counts()
labels = ['Non-Defaulter', 'Defaulter']

fig = go.Figure(data = [
        go.Pie(values = target_distribution, labels = labels, textinfo = 'label+percent+value' , pull = [0,0.04])], 
         layout = go.Layout(title = 'Distribution of Target Variable'))

fig.show()
```


<div>                            <div id="baf3b328-942d-4740-b333-50f3223bc38a" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("baf3b328-942d-4740-b333-50f3223bc38a")) {                    Plotly.newPlot(                        "baf3b328-942d-4740-b333-50f3223bc38a",                        [{"labels":["Non-Defaulter","Defaulter"],"pull":[0,0.04],"textinfo":"label+percent+value","values":[282686,24825],"type":"pie"}],                        {"title":{"text":"Distribution of Target Variable"},"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('baf3b328-942d-4740-b333-50f3223bc38a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


##### Observations and Conclusions:

<ul><li>From the distribution of Target variable, one thing that we can quickly notice is the Data Imbalance. There are only 8.07% of the total loans that had actually been Defaulted. This means that Defaulters is the minority class. 
    <li>On the other hand, there are 91.9% loans which were not Defaulted. Thus, Non-Defaulters will be our majority class.
    <li>The Defaulters have been assigned a Target variable of 1 and Non-Defaulters have been assigned Target Variable 0.
    <li>For imbalanced dataset, during building the model, we cannot feed the data as is to some algorithms, which are imbalance sensitive.
    <li>Similar is the case with the Performance Metrics. For such a dataset, Accuracy is usually not the right metric as the Accuracy would generally be biased to majority class. We can use other metrics such as ROC-AUC Score, Log-Loss, F1-Score, Confusion Matrix for better evaluation of model.
    <li>One more important thing to note here is that there are very few people who actually default, and they tend to show some sort of different behaviour. Thus in such cases of Fraud, Default and Anamoly Detection, we need to focus on outliers too, and we cannot remove them, as they could be the differentiating factor between Defaulter and Non-Defaulter.

#### Phi-K matrix

We will plot a heatmap of the values of Phi-K Correlation Coefficient between each of the feature with the other. <br>
The Phi-K coefficient is similar to Correlation Coefficient except that it can be used with a pair of categorical features to check if one feature shows some sort of association with the other categorical feature. It's max value can be 1 which would show a maximum association between two categorical variables.


```python
categorical_columns = ['TARGET','FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                                   'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                                  'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                                   'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
                                'LIVE_CITY_NOT_WORK_CITY'] + ['FLAG_DOCUMENT_' + str(i) for i in range(2,22)] + application_train.dtypes[
                                    application_train.dtypes == 'object'].index.tolist()
plot_phik_matrix(application_train, categorical_columns, figsize = (15,15), fontsize = 8)
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_32_1.png)
    


    ----------------------------------------------------------------------------------------------------
    Categories with highest values of Phi-K Correlation value with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>OCCUPATION_TYPE</td>
      <td>0.102846</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ORGANIZATION_TYPE</td>
      <td>0.089164</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NAME_INCOME_TYPE</td>
      <td>0.084831</td>
    </tr>
    <tr>
      <th>12</th>
      <td>REG_CITY_NOT_WORK_CITY</td>
      <td>0.079946</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FLAG_EMP_PHONE</td>
      <td>0.072087</td>
    </tr>
    <tr>
      <th>11</th>
      <td>REG_CITY_NOT_LIVE_CITY</td>
      <td>0.069588</td>
    </tr>
    <tr>
      <th>15</th>
      <td>FLAG_DOCUMENT_3</td>
      <td>0.069525</td>
    </tr>
    <tr>
      <th>41</th>
      <td>NAME_FAMILY_STATUS</td>
      <td>0.056043</td>
    </tr>
    <tr>
      <th>42</th>
      <td>NAME_HOUSING_TYPE</td>
      <td>0.051107</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LIVE_CITY_NOT_WORK_CITY</td>
      <td>0.050956</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol><li>From the above heatmap of Phi-K Correlation, we see that most of the categorical features are not correlated to each other, however some of them show strong correlation.</li>
    <li>Some of the highly correlated Category pairs are:
        <ul><li>REGION_RATING_CLIENT_W_CITY and REGION_RATING_CLIENT -  This is understandable as they would more or less tell a similar story.</li>
            <li>LIVE_REGION_NOT_WORK_REGION and REG_REGION_NOT_WORK_REGION</li>
            <li>NAME_INCOME_TYPE, ORGANIZATION_TYPE and FLAG_EMP_PHONE</li>
        </ul>
    </li>
    <li>We can also see some correlation between the Organization type and the income type of a client. Similarly we see a correlation between the Occupation Type and the Organization Type too.</li>
    <li>We find that the category OCCUPATION_TYPE, ORGANIZATION_TYPE, NAME_INCOME_TYPE, REG_CITY_NOT_WORK_CITY are some of the highest correlated categories with the TARGET variable. These maybe important in the classification task, and would need further EDA</li></ol>
    

#### Correlation Matrix of Features

We will plot a heatmap of the correlation of each numeric feature with respect to other features. We have excluded the column SK_ID_CURR, as it does not have any relevance. This heatmap will help us identify the highly correlated numeric features and will also help us to identify features which are highly correlated with Target Variable.


```python
columns_to_drop = ['SK_ID_CURR'] + list(set(categorical_columns) - set(['TARGET']))
corr_mat = correlation_matrix(application_train, columns_to_drop, figsize = (17,17), fontsize = 8, cmap = 'inferno')
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_36_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in application_train table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-"*100)
```

    interval columns not set, guessing: ['TARGET', 'CNT_CHILDREN']
    interval columns not set, guessing: ['TARGET', 'AMT_INCOME_TOTAL']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT']
    interval columns not set, guessing: ['TARGET', 'AMT_ANNUITY']
    interval columns not set, guessing: ['TARGET', 'AMT_GOODS_PRICE']
    interval columns not set, guessing: ['TARGET', 'REGION_POPULATION_RELATIVE']
    interval columns not set, guessing: ['TARGET', 'DAYS_BIRTH']
    interval columns not set, guessing: ['TARGET', 'DAYS_EMPLOYED']
    interval columns not set, guessing: ['TARGET', 'DAYS_REGISTRATION']
    interval columns not set, guessing: ['TARGET', 'DAYS_ID_PUBLISH']
    interval columns not set, guessing: ['TARGET', 'OWN_CAR_AGE']
    interval columns not set, guessing: ['TARGET', 'CNT_FAM_MEMBERS']
    interval columns not set, guessing: ['TARGET', 'HOUR_APPR_PROCESS_START']
    interval columns not set, guessing: ['TARGET', 'EXT_SOURCE_1']
    interval columns not set, guessing: ['TARGET', 'EXT_SOURCE_2']
    interval columns not set, guessing: ['TARGET', 'EXT_SOURCE_3']
    interval columns not set, guessing: ['TARGET', 'APARTMENTS_AVG']
    interval columns not set, guessing: ['TARGET', 'BASEMENTAREA_AVG']
    interval columns not set, guessing: ['TARGET', 'YEARS_BEGINEXPLUATATION_AVG']
    interval columns not set, guessing: ['TARGET', 'YEARS_BUILD_AVG']
    interval columns not set, guessing: ['TARGET', 'COMMONAREA_AVG']
    interval columns not set, guessing: ['TARGET', 'ELEVATORS_AVG']
    interval columns not set, guessing: ['TARGET', 'ENTRANCES_AVG']
    interval columns not set, guessing: ['TARGET', 'FLOORSMAX_AVG']
    interval columns not set, guessing: ['TARGET', 'FLOORSMIN_AVG']
    interval columns not set, guessing: ['TARGET', 'LANDAREA_AVG']
    interval columns not set, guessing: ['TARGET', 'LIVINGAPARTMENTS_AVG']
    interval columns not set, guessing: ['TARGET', 'LIVINGAREA_AVG']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAPARTMENTS_AVG']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAREA_AVG']
    interval columns not set, guessing: ['TARGET', 'APARTMENTS_MODE']
    interval columns not set, guessing: ['TARGET', 'BASEMENTAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'YEARS_BEGINEXPLUATATION_MODE']
    interval columns not set, guessing: ['TARGET', 'YEARS_BUILD_MODE']
    interval columns not set, guessing: ['TARGET', 'COMMONAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'ELEVATORS_MODE']
    interval columns not set, guessing: ['TARGET', 'ENTRANCES_MODE']
    interval columns not set, guessing: ['TARGET', 'FLOORSMAX_MODE']
    interval columns not set, guessing: ['TARGET', 'FLOORSMIN_MODE']
    interval columns not set, guessing: ['TARGET', 'LANDAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'LIVINGAPARTMENTS_MODE']
    interval columns not set, guessing: ['TARGET', 'LIVINGAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAPARTMENTS_MODE']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'APARTMENTS_MEDI']
    interval columns not set, guessing: ['TARGET', 'BASEMENTAREA_MEDI']
    interval columns not set, guessing: ['TARGET', 'YEARS_BEGINEXPLUATATION_MEDI']
    interval columns not set, guessing: ['TARGET', 'YEARS_BUILD_MEDI']
    interval columns not set, guessing: ['TARGET', 'COMMONAREA_MEDI']
    interval columns not set, guessing: ['TARGET', 'ELEVATORS_MEDI']
    interval columns not set, guessing: ['TARGET', 'ENTRANCES_MEDI']
    interval columns not set, guessing: ['TARGET', 'FLOORSMAX_MEDI']
    interval columns not set, guessing: ['TARGET', 'FLOORSMIN_MEDI']
    interval columns not set, guessing: ['TARGET', 'LANDAREA_MEDI']
    interval columns not set, guessing: ['TARGET', 'LIVINGAPARTMENTS_MEDI']
    interval columns not set, guessing: ['TARGET', 'LIVINGAREA_MEDI']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAPARTMENTS_MEDI']
    interval columns not set, guessing: ['TARGET', 'NONLIVINGAREA_MEDI']
    interval columns not set, guessing: ['TARGET', 'TOTALAREA_MODE']
    interval columns not set, guessing: ['TARGET', 'OBS_30_CNT_SOCIAL_CIRCLE']
    interval columns not set, guessing: ['TARGET', 'DEF_30_CNT_SOCIAL_CIRCLE']
    interval columns not set, guessing: ['TARGET', 'OBS_60_CNT_SOCIAL_CIRCLE']
    interval columns not set, guessing: ['TARGET', 'DEF_60_CNT_SOCIAL_CIRCLE']
    interval columns not set, guessing: ['TARGET', 'DAYS_LAST_PHONE_CHANGE']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_HOUR']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_DAY']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_WEEK']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_MON']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_QRT']
    interval columns not set, guessing: ['TARGET', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>EXT_SOURCE_3</td>
      <td>0.247680</td>
    </tr>
    <tr>
      <th>13</th>
      <td>EXT_SOURCE_1</td>
      <td>0.217846</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EXT_SOURCE_2</td>
      <td>0.213965</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DAYS_BIRTH</td>
      <td>0.102378</td>
    </tr>
    <tr>
      <th>63</th>
      <td>DAYS_LAST_PHONE_CHANGE</td>
      <td>0.073218</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DAYS_EMPLOYED</td>
      <td>0.072095</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DAYS_ID_PUBLISH</td>
      <td>0.067766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AMT_GOODS_PRICE</td>
      <td>0.059094</td>
    </tr>
    <tr>
      <th>23</th>
      <td>FLOORSMAX_AVG</td>
      <td>0.058826</td>
    </tr>
    <tr>
      <th>51</th>
      <td>FLOORSMAX_MEDI</td>
      <td>0.058595</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The heatmap does a good job of showing the value or level of correlation that each particular feature has with all other features.</li>
    <li>It can be observed that most of the heatmap contains a purple-ish color, which indicates a very small value of correlation. This implies that most of the features are indeed not correlated to others.</li>
    <li>However, we can see contrasting shades at the middle of the heatmap. These shades depict a high value of correlation between the features. These are the features which are related to the stats of the apartments.<br>
        If we look at the features of application_train, we can clearly see that the statistics of apartments are given in terms of Mean, Median and Mode, so it can be expected for the mean, median and mode to be correlated with each other. One more thing to note is that the features among particular category, for example Mean are also correlated with other mean features, such as Number of Elevators, Living Area, Non-Living Area, Basement Area, etc.</li>
    <li>We also see some high correlation between AMT_GOODS_PRICE and AMT_CREDIT, between DAYS_EMPLOYED and DAYS_BIRTH. </li>
    <li>We would not want highly correlated features as they increase the time complexity of the model without adding much value to it. Hence, we would be removing the inter-correlated features.
    <li>Among all the features, we see some high correlation for EXT_SOURCE features with respect to Target Variable. These might be important for our classification task.</ol>

#### Plotting Categorical Variables

<b><u>Distribution of Categorical Variable NAME_CONTRACT_TYPE</u></b>

This column contains information about the type of loan for the given applicant. As per the documentation provided by Home Credit, there are two types of loans, i.e. Revolving Loans and Cash Loans.


```python
#let us first see the unique categories of 'NAME_CONTRACT_TYPE'
print_unique_categories(application_train, 'NAME_CONTRACT_TYPE')

#plotting the Pie Plot for the column
plot_categorical_variables_pie(application_train, 'NAME_CONTRACT_TYPE', hole = 0.5)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'NAME_CONTRACT_TYPE' are:
    ['Cash loans' 'Revolving loans']
    ----------------------------------------------------------------------------------------------------
    


<div>                            <div id="77f866cb-6dee-452a-9c0a-486ba46dbdb6" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("77f866cb-6dee-452a-9c0a-486ba46dbdb6")) {                    Plotly.newPlot(                        "77f866cb-6dee-452a-9c0a-486ba46dbdb6",                        [{"hole":0.5,"labels":["Cash loans","Revolving loans"],"textinfo":"label+percent","textposition":"inside","values":[278232,29279],"type":"pie","domain":{"x":[0.0,0.45],"y":[0.0,1.0]}},{"hole":0.5,"hoverinfo":"label+value","labels":["Cash loans","Revolving loans"],"textinfo":"label+value","values":[8.35,5.48],"type":"pie","domain":{"x":[0.55,1.0],"y":[0.0,1.0]}}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"annotations":[{"font":{"size":16},"showarrow":false,"text":"Distribution of NAME_CONTRACT_TYPE for all Targets","x":0.225,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Percentage of Defaulters for each category of NAME_CONTRACT_TYPE","x":0.775,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"title":{"text":"Distribution of NAME_CONTRACT_TYPE"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('77f866cb-6dee-452a-9c0a-486ba46dbdb6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above plot, we can draw following observations and conclusions:
<ol><li>From the first subplot, i.e. the overall distribution:
        <ul><li>It can be seen that most of the loans that the customers take are Cash Loans.</li>
            <li>Only 9.52% of the people have taken Revolving Loans.</li></ul></li>
<li>From the second subplot, i.e. Percentage of Defaulters:
        <ul><li>We see is that there are more percentage of people who have defaulted with Cash Loans (8.35%) as compared to those who defaulted with Revolving Loans (5.48%).</li></ul></li></ol>

<b><u>Distribution of Categorical Variable CODE_GENDER</u></b>

This column contains information about the Gender of the Client/Applicant.<br>
Here <b>M</b> stands for <b>Male</b> and <b>F</b> for <b>Female</b>.


```python
#let us first see the unique categories of 'CODE_GENDER'
print_unique_categories(application_train, 'CODE_GENDER', show_counts = True)

#plotting the Pie Plot for the Column
plot_categorical_variables_pie(application_train, 'CODE_GENDER', hole = 0.5)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'CODE_GENDER' are:
    ['M' 'F' 'XNA']
    ----------------------------------------------------------------------------------------------------
    Counts of each category are:
    F      202448
    M      105059
    XNA         4
    Name: CODE_GENDER, dtype: int64
    ----------------------------------------------------------------------------------------------------
    


<div>                            <div id="c5feba14-674a-437f-89a9-7d99ad4725ec" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c5feba14-674a-437f-89a9-7d99ad4725ec")) {                    Plotly.newPlot(                        "c5feba14-674a-437f-89a9-7d99ad4725ec",                        [{"hole":0.5,"labels":["F","M","XNA"],"textinfo":"label+percent","textposition":"inside","values":[202448,105059,4],"type":"pie","domain":{"x":[0.0,0.45],"y":[0.0,1.0]}},{"hole":0.5,"hoverinfo":"label+value","labels":["F","M"],"textinfo":"label+value","values":[7.0,10.14],"type":"pie","domain":{"x":[0.55,1.0],"y":[0.0,1.0]}}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"annotations":[{"font":{"size":16},"showarrow":false,"text":"Distribution of CODE_GENDER for all Targets","x":0.225,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Percentage of Defaulters for each category of CODE_GENDER","x":0.775,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"title":{"text":"Distribution of CODE_GENDER"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c5feba14-674a-437f-89a9-7d99ad4725ec');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

The Distribution of CODE_GENDER feature seems interesting. Here are some talking points:
<ol><li>First point to observe is that there are 4 rows in the application_train table which have 'XNA' genders, which dont make much sense, and can be counted as NaN values. Since they are only 4 in Number and only belong to Non-Defaulter Category. So, we can eliminate those rows without much hesitation.</li>
    <li>From the subplot 1 we see that for the given dataset, there are more number of Female applicants (65.8%) than Male applicants (34.2%). </li>
    <li>However, contrary to the number of Female applicants, from the second plot we note that it has been seen that Male applicants tend to default more (10.14%) as compared to Female applicants (7%). </li></ol>
Thus, it can be said that Male have more tendency to default than Female as per the given dataset.

<b><u>Distribution of Categorical Variable FLAG_EMP_PHONE</u></b>

This column is a boolean column, which tells whether if the client provided his Work Phone Number or not.<br>
Here <b>1</b> stands for <b>Yes</b> and <b>0</b> stands for <b>No</b>.


```python
#let us first see the unique categories of 'FLAG_EMP_PHONE'
print_unique_categories(application_train, 'FLAG_EMP_PHONE')

#plotting the Pie Plot for the Column
plot_categorical_variables_pie(application_train, column_name = 'FLAG_EMP_PHONE', hole = 0.5)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'FLAG_EMP_PHONE' are:
    [1 0]
    ----------------------------------------------------------------------------------------------------
    


<div>                            <div id="111217ce-327e-4ef8-8b31-d787f604f0ee" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("111217ce-327e-4ef8-8b31-d787f604f0ee")) {                    Plotly.newPlot(                        "111217ce-327e-4ef8-8b31-d787f604f0ee",                        [{"hole":0.5,"labels":[1,0],"textinfo":"label+percent","textposition":"inside","values":[252125,55386],"type":"pie","domain":{"x":[0.0,0.45],"y":[0.0,1.0]}},{"hole":0.5,"hoverinfo":"label+value","labels":[1,0],"textinfo":"label+value","values":[8.66,5.4],"type":"pie","domain":{"x":[0.55,1.0],"y":[0.0,1.0]}}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"annotations":[{"font":{"size":16},"showarrow":false,"text":"Distribution of FLAG_EMP_PHONE for all Targets","x":0.225,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Percentage of Defaulters for each category of FLAG_EMP_PHONE","x":0.775,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"title":{"text":"Distribution of FLAG_EMP_PHONE"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('111217ce-327e-4ef8-8b31-d787f604f0ee');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

This feature contains two categories, i.e. if the client had provided his Work Phone Number during registration/application process or not.
<ol><li>From the first subplot we see that most of the applicants do not provide their Work Phone Number (82%) and only 18% have provided their Work Phone Number.
    <li>It can also be seen that the Default tendency for those who do provide Work Phone Number is more than those who do not provide Work Phone Number.<br>
        This is characteristic could be attributed to the fact that the Defaulters might be providing their Work Phone Numbers so that they don't get disturbed on their personal phone.

<b><u>Distribution of Categorical Variable REGION_RATING_CLIENT_W_CITY</u></b>

This feature is the rating provided by the Home Credit to each client's region based on the surveys that they might have done. This rating also takes into account the City in which the client lives. <br>
Taking City into account is important because even if some regions have a good rating in a particular City, but that City doesn't have high rating, then applicant would be given a medium rating and not a high rating.<br>
It contains values in the range from 1 to 3.


```python
#let us first see the unique categories of 'REGION_RATING_CLIENT_W_CITY'
print_unique_categories(application_train, 'REGION_RATING_CLIENT_W_CITY')

#plotting the Pie Plot for the Column
plot_categorical_variables_pie(application_train, column_name = 'REGION_RATING_CLIENT_W_CITY')
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'REGION_RATING_CLIENT_W_CITY' are:
    [2 1 3]
    ----------------------------------------------------------------------------------------------------
    


<div>                            <div id="9f2bdcf7-044c-4ae4-ac07-48043912b18c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9f2bdcf7-044c-4ae4-ac07-48043912b18c")) {                    Plotly.newPlot(                        "9f2bdcf7-044c-4ae4-ac07-48043912b18c",                        [{"hole":0,"labels":[2,3,1],"textinfo":"label+percent","textposition":"inside","values":[229484,43860,34167],"type":"pie","domain":{"x":[0.0,0.45],"y":[0.0,1.0]}},{"hole":0,"hoverinfo":"label+value","labels":[2,3,1],"textinfo":"label+value","values":[7.92,11.4,4.84],"type":"pie","domain":{"x":[0.55,1.0],"y":[0.0,1.0]}}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"annotations":[{"font":{"size":16},"showarrow":false,"text":"Distribution of REGION_RATING_CLIENT_W_CITY for all Targets","x":0.225,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Percentage of Defaulters for each category of REGION_RATING_CLIENT_W_CITY","x":0.775,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"title":{"text":"Distribution of REGION_RATING_CLIENT_W_CITY"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('9f2bdcf7-044c-4ae4-ac07-48043912b18c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above plots, we can draw following insights:
<ol><li>From the first subplot, we see that most of the clients (74.6%) have a region rating of 2. This is the middle value which is for most of the applicants.<br>Very few applicants have a region rating of 1 (only 11.1%) and some have a rating of 3 (14.3%).</li>
    <li>Among the Defaulters, it is seen that most of the defaulters have a region rating of 3 (11.4%) which is comparably higher to the other two ratings, i.e. clients with rating of 1 have a Defaulting percentage of just 4.84% and with rating 2 have a percentage of 7.92%.</li>
    </ol>
This shows that the rating 3 could be an important attribute for making a decision on Defaulting Characteristics.

<b><u>Distribution of Categorical Variable NAME_EDUCAtION_TYPE</u></b>

This feature descibes/enlists the Highest Education that the client had achieved.


```python
#let us first see the unique categories of 'NAME_EDUCATION_TYPE'
print_unique_categories(application_train, 'NAME_EDUCATION_TYPE', show_counts = True)

#plotting the Bar Plot for the Column
plot_categorical_variables_bar(application_train, column_name = 'NAME_EDUCATION_TYPE', rotation = 45, horizontal_adjust = 0.25)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'NAME_EDUCATION_TYPE' are:
    ['Secondary / secondary special' 'Higher education' 'Incomplete higher'
     'Lower secondary' 'Academic degree']
    ----------------------------------------------------------------------------------------------------
    Counts of each category are:
    Secondary / secondary special    218391
    Higher education                  74863
    Incomplete higher                 10277
    Lower secondary                    3816
    Academic degree                     164
    Name: NAME_EDUCATION_TYPE, dtype: int64
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of NAME_EDUCATION_TYPE = 5
    


    
![png](output_53_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

Looking at the above plots, we can conclude the following:
<ol><li>About 71% of people have had their education only till Secondary/Secondary Special, along with 24.34% clients having done Higher Education. This suggests that most of the clients/borrowers don't have a high education level.</li>
    <li>From the second plot, we see that the people who have had their studies till only Lower Secondary have the highest Defaulting Characterists, with Secondary and Incomplete higher having similar defaulting tendencies.</li>
    <li>The group of people with Higher Education have comparably lower defaulting tendency, which is logical too. Also, people with Academic Degree show the least Defaulting Rate. However, the Academic Degree group are very few in numbers, so it might not be very useful.</li></ol>
   

<b><u>Distribution of Categorical Variable OCCUPATION_TYPE</u></b>

This feature tells about the type of Occupation that the client has. This can be a very important feature which could describe the Defaulting Characteristics of a client. Let us see the plots for them.


```python
#let us first see the unique categories of 'OCCUPATION_TYPE'
print_unique_categories(application_train, 'OCCUPATION_TYPE')

#plotting the Bar Plot for the Column
plot_categorical_variables_bar(application_train, column_name = 'OCCUPATION_TYPE', figsize = (20,6), rotation = 90)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'OCCUPATION_TYPE' are:
    ['Laborers' 'Core staff' 'Accountants' 'Managers' nan 'Drivers'
     'Sales staff' 'Cleaning staff' 'Cooking staff' 'Private service staff'
     'Medicine staff' 'Security staff' 'High skill tech staff'
     'Waiters/barmen staff' 'Low-skill Laborers' 'Realty agents' 'Secretaries'
     'IT staff' 'HR staff']
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of OCCUPATION_TYPE = 19
    


    
![png](output_56_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the plots of Occupation Type, we can draw following observations:
<ol><li>Among the applicants, the most common type of Occupation is Laborers contributing to close to 26% applications. The next most frequent occupation is Sales Staff, followed by Core Staff and Managers.</li>
    <li>The Defaulting Rate for Low-Skill Laborers is the highest among all the occupation types (~17.5%). This is followed by Drivers, Waiters, Security Staff, Laborers, Cooking Staff, etc. All the jobs are low-level jobs. This shows that low-level Jobs people tend to have higher default rate.</li>
    <li>The lowest Defaulting Rate are among Accountants, Core Staff, Managers, High skill tech staff, HR staff, etc. which are from medium to high level jobs.</li></ol>

Thus it can be concluded that Low-level job workers tend to have a higher defaulting tendency compared to medium-high level jobs.

<b><u>Distribution of Categorical Variable ORGANIZATION_TYPE</u></b>

Similar to Occupation Type, Organization Type that the client belongs to could also be an important feature for predicting the Default Risk of that client. Let us visualize this feature in more detail.


```python
print(f"Total Number of categories of ORGANIZATION_TYPE = {len(application_train.ORGANIZATION_TYPE.unique())}")

plt.figure(figsize = (25,16))
sns.set(style = 'whitegrid', font_scale = 1.2)
plt.subplots_adjust(wspace=0.25)

plt.subplot(1,2,1)
count_organization = application_train.ORGANIZATION_TYPE.value_counts().sort_values(ascending = False)
sns.barplot(x = count_organization, y = count_organization.index)
plt.title('Distribution of ORGANIZATION_TYPE', pad = 20)
plt.xlabel('Counts')
plt.ylabel('ORGANIZATION_TYPE')

plt.subplot(1,2,2)
percentage_default_per_organization = application_train[application_train.TARGET == 1].ORGANIZATION_TYPE.value_counts() * 100 / count_organization
percentage_default_per_organization = percentage_default_per_organization.dropna().sort_values(ascending = False)
sns.barplot(x = percentage_default_per_organization, y = percentage_default_per_organization.index)
plt.title('Percentage of Defaulters for each category of ORGANIZATION_TYPE', pad = 20)
plt.xlabel('Percentage of Defaulters per category')
plt.ylabel('ORGANIZATION_TYPE')

plt.show()
```

    Total Number of categories of ORGANIZATION_TYPE = 58
    


    
![png](output_59_1.png)
    


##### Observations and Conclusions:

There are a lots of organization types which the client belongs to, 58 to be precise. The plots above give the following observations:
<ol><li>From the first plot we see that most of the applicants work in Organizations of Type 'Business Entity Type3', 'XNA' or 'Self Employed'. The Organization Type 'XNA' could probably denote unclassified Organization TYpe.</li>
    <li>From the second plot, we notice that the applicants belonging to 'Transport: type 3' have the highest defaulting tendency as compared to the rest. They are followed by organizations of types: 'Industry: type 13', 'Industry: type 8', 'Restaurant', 'Construction', etc.</li>
    <li>The organizations which show lowest default rates are 'Trade: type 4', 'Industry: type 12', etc.</li>
</ol>
These type numbers also would say something more about the Organization, however, we don't have any information related to that, so we will stick with the naming provided to us only.

<b><u>Distribution of Categorical Variable REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY</u></b>
<br><br>
<i>REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY:</i><br>
These columns include flags whether if the the client's permanent address matches with his Contact Address or Work Address or not at region level <br><br>
<i>LIVE_CITY_NOT_WORK_CITY</i>
This column indicates whether if the client's permanent address matches with his Contact Address at city level or not.<br><br>
Here 1 indicates different addresses and 0 indicates same addresses.


```python
print('-'*100)
plot_categorical_variables_bar(application_train, column_name = 'REG_CITY_NOT_LIVE_CITY', figsize = (14, 4), horizontal_adjust = 0.33)
print('-'*100)
plot_categorical_variables_bar(application_train, column_name = 'REG_CITY_NOT_WORK_CITY', figsize = (14, 4), horizontal_adjust = 0.33)
print('-'*100)
plot_categorical_variables_bar(application_train, column_name = 'LIVE_CITY_NOT_WORK_CITY', figsize = (14, 4), horizontal_adjust = 0.33)
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of REG_CITY_NOT_LIVE_CITY = 2
    


    
![png](output_62_1.png)
    


    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of REG_CITY_NOT_WORK_CITY = 2
    


    
![png](output_62_3.png)
    


    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of LIVE_CITY_NOT_WORK_CITY = 2
    


    
![png](output_62_5.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above 3 plots, following insights can be drawn:
<ol><li>Of all the applicants there are only a minority of applicants whose addresses do not match.
    <ul><li>Firstly, there are only 7.52% people who have different permanent address from their contact address at region level.</li>
        <li>Secondly, there are around 23.05% people who have different permanent address from their work address at region level. This higher number is explainable, because it is possible that they work in different region as compared to their permanent address.</li>
        <li>Lastly, there are around 17.96% people who have different permanent address from their contact address at city level.</li>
    </ul>
    <li>
        <ul><li>If we look at the defaulting characteristics, we find that there is maximum defaulting tendency of those people who have their permanent and contact addresses different at region level, which is followed by different permanent and work address and lastly different permanent and contact address at city level.</li>
            <li>For all the cases it is seen that the Defaulting tendency of those people who have different addresses is higher than those who have same address. This means that somewhere, this difference in address may suggest Defaulting Risk.</li>
        </ul>
    </li>
    </ol>

<b><u>Distribution of Categorical Variables FLAG_DOCUMENT_3</u></b>


This columns contains the flag about a document that was to be submitted by the applicant. It's value is 0 if the client had provided the document and 1 if not.


```python
plot_categorical_variables_bar(application_train, column_name = 'FLAG_DOCUMENT_3', figsize = (14, 4), horizontal_adjust = 0.33)
```

    Total Number of unique categories of FLAG_DOCUMENT_3 = 2
    


    
![png](output_66_1.png)
    


##### Observations and Conclusions:

From the above plot, we see that:
<ol><li>From the first subplot, we see that most of the clients hadn't provided this document (~71%) and only 29% did. </li>
    <li>From the second subplot, we see that those who had provided this document had a higher defaulting rate than those who didn't. This means that the applicants who had provided the Document_3 tend to default more than those who don't. </li>
    <li>Perhaps this could be something related to BPL certificate or something, but we haven't been provided with that information, so we cannot say anything about the kind of document it could have been.</li>
</ol>

#### Plotting Continuous Variables

<b><u>Distribution of Continuous Variable Age of Applicant</u></b>

For the given dataset, the age is given in Days, which can be harder to interpret. Thus, we will create a latent variable to store the ages in Years, which would be easier to analyse and interpret.


```python
application_train['AGE_YEARS'] = application_train['DAYS_BIRTH'] * -1 / 365
plot_continuous_variables(application_train, 'AGE_YEARS', plots = ['distplot','box'])
_ = application_train.pop('AGE_YEARS')
```


    
![png](output_70_0.png)
    


##### Observations and Conclusions:

From the above two plots, we can draw some important insights.
<ol><li>From the distplot, we can observe the peak of Age of people who Default to be close to 30 years. Also, at this point, the Non-Defaulters have a quite smaller PDF. One more thing to note is that the PDF of Age for Defaulters starts a bit left from the Non-Defaulters, and also is a bit throughtout the range. This means that the Defaulters are usually younger than Non-Defaulters.</li>
    <li>From CDF too we see that the probability
    <li>From the box-plot too, we can better visualize the same thing. The Age of Defaulters is usually lesser than the Non-Defaulters. All the quantiles of ages of Defaulters is lesser than Non-Defaulters. The 75th percentile value of Non-Defaulters is around 54 years while for Defaulters it is near to 49 years.</li></ol>
These observations imply that the Defaulters are usually younger than Non-Defaulters.

<b><u>Distribution of Continuous Variables with DAYS features</u><b>



<b>DAYS_EMPLOYED</b><br>

This feature tells about the number of days from the day of application the applicant had been employed. For easy interpretation, we will convert the days to years.


```python
application_train['YEARS_EMPLOYED'] = application_train.DAYS_EMPLOYED * -1 / 365
print_percentiles(application_train, 'DAYS_EMPLOYED')
plot_continuous_variables(application_train, 'YEARS_EMPLOYED', plots = ['box'], scale_limits = [0,70], figsize = (10,8))
_ = application_train.pop('YEARS_EMPLOYED')
```

    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_EMPLOYED is -17912.0
    The 25th percentile value of DAYS_EMPLOYED is -2760.0
    The 50th percentile value of DAYS_EMPLOYED is -1213.0
    The 75th percentile value of DAYS_EMPLOYED is -289.0
    The 90th percentile value of DAYS_EMPLOYED is 365243.0
    The 92th percentile value of DAYS_EMPLOYED is 365243.0
    The 94th percentile value of DAYS_EMPLOYED is 365243.0
    The 96th percentile value of DAYS_EMPLOYED is 365243.0
    The 98th percentile value of DAYS_EMPLOYED is 365243.0
    The 100th percentile value of DAYS_EMPLOYED is 365243.0
    ----------------------------------------------------------------------------------------------------
    


    
![png](output_74_1.png)
    


##### Observations and Conclusions:

<ol><li>We see that the DAYS_EMPLOYED column contains some erroneous datapoints with values 365243. These seem like some erroneous/non-sensicle values.</li>
    <li>From the box plot we observe that the Defaulters seem to have less number of years being employed as compared to Non-Defaulters. All the 25th, 50th and 75th quantile for Defaulters are lesser than those of Non-Defaulters.

<b>DAYS_ID_PUBLISH</b><br>

This columns tells about how many days ago from the day of registration did the client change his Identity Document with which he applied for loan.


```python
plot_continuous_variables(application_train, 'DAYS_ID_PUBLISH', plots = ['box'], figsize = (10,8))
```


    
![png](output_77_0.png)
    


##### Observations and Conclusions:

From the above box plot, we see a similar trend as seen with DAYS_REGISTRATION, in which the Defaulters usually had lesser number of days since they changed their identity. The Non-Defaulters show to have more number of days for all the quantiles since they changed their identity document.

<b><u>Distribution of EXT_SOURCES</u></b>

There are three EXT_SOURCES columns, which contain values between 0 and 1. They are normalized scores from different sources


```python
print('-'*100)
plot_continuous_variables(application_train, 'EXT_SOURCE_1', plots = ['distplot', 'box'], figsize = (16,8))
print('-'*100)
plot_continuous_variables(application_train, 'EXT_SOURCE_2', plots = ['distplot', 'box'], figsize = (16,8))
print('-'*100)
plot_continuous_variables(application_train, 'EXT_SOURCE_3', plots = ['distplot', 'box'], figsize = (16,8))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_81_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


    
![png](output_81_3.png)
    


    ----------------------------------------------------------------------------------------------------
    


    
![png](output_81_5.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above three plots, we can draw following conclusions:
<ol><li>If we look at the box-plots, we can clearly see a similar trend for all three EXT_SOURCE columns, which is that the Defaulters tend to have considerably lower values.
    <li>This trend can also be seen with the PDFs. The Non-Defaulters show a higher peak at high EXT_SOURCE values, and the Probability Densities are very low for low values. This implies that Non-Defaulters generally have high values of these scores.
    <li>It is interesting to note that the median value for defaulters is almost equal to or lower than 25th percentile values of Non-Defaulters.</li>
    <li>EXT_SOURCE_1 and EXT_SOURCE_3 columns tend to show better discrimination/separability as compared to EXT_SOURCE_2.
    <li>These 3 features look to be best separating the Defaulters and Non-Defaulters linearly among all the features so far.

<b><u>Distribution of FLOORSMAX_AVG and FLOORSMIN_MODE</u></b>

These columns describe the normalized scores of Average of Maximum number of Floors and Mode of Minimum number of Floors in applicant's building 


```python
plot_continuous_variables(application_train, 'FLOORSMAX_AVG', plots = ['box'], figsize = (10,8))
```


    
![png](output_84_0.png)
    



```python
plot_continuous_variables(application_train, 'FLOORSMIN_MODE', plots = ['box'], figsize = (10,8))
```


    
![png](output_85_0.png)
    


##### Observations and Conclusions

From the above plot, we can draw the following insights:
<ol><li>The defaulters have lower median value of FLOORSMAX_AVG feature as compared to Non-Defaulters. The 75th percentile values of both the Defaulters and Non-Defaulters is more or less the same, but the 25th percentile value of Non-Defaulters is almost more than the median of Defaulters, thus this could be an important feature.</li>
    <li>The Non-Defaulters also tend to show a higher value of FLLORSMIN_MODE as compared to Defaulters. The 75th percentile value of Non-Defaulters is significantly higher than the 75th percentile value of Defaulters.

### bureau.csv

##### Description

This table consists of all client's previous credit records with financial institutions other than Home Credit Group which were reported by the the Credit Bureau.

#### Basic Stats


```python
print(f'The shape of bureau.csv is: {bureau.shape}')
print('-'*100)
print(f'Number of unique SK_ID_BUREAU in bureau.csv are: {len(bureau.SK_ID_BUREAU.unique())}')
print(f'Number of unique SK_ID_CURR in bureau.csv are: {len(bureau.SK_ID_CURR.unique())}')
print(f'Number of overlapping SK_ID_CURR in application_train.csv and bureau.csv are: {len(set(application_train.SK_ID_CURR.unique()).intersection(set(bureau.SK_ID_CURR.unique())))}')
print(f'Number of overlapping SK_ID_CURR in application_test.csv and bureau.csv are: {len(set(application_test.SK_ID_CURR.unique()).intersection(set(bureau.SK_ID_CURR.unique())))}')
print('-'*100)
print(f'Number of duplicate values in bureau: {bureau.shape[0] - bureau.duplicated().shape[0]}')
print('-'*100)
display(bureau.head(5))
```

    The shape of bureau.csv is: (1716428, 17)
    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_BUREAU in bureau.csv are: 1716428
    Number of unique SK_ID_CURR in bureau.csv are: 305811
    Number of overlapping SK_ID_CURR in application_train.csv and bureau.csv are: 263491
    Number of overlapping SK_ID_CURR in application_test.csv and bureau.csv are: 42320
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in bureau: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215354</td>
      <td>5714462</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-497</td>
      <td>0</td>
      <td>-153.0</td>
      <td>-153.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91323.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215354</td>
      <td>5714463</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-208</td>
      <td>0</td>
      <td>1075.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>225000.0</td>
      <td>171342.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215354</td>
      <td>5714464</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>528.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>464323.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>215354</td>
      <td>5714465</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>215354</td>
      <td>5714466</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-629</td>
      <td>0</td>
      <td>1197.0</td>
      <td>NaN</td>
      <td>77674.5</td>
      <td>0</td>
      <td>2700000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-21</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


<h5>Observations and Conclusions:</h5>

<ol><li>The bureau.csv file contains close to 1.7M datapoints, with 17 features. Out of these 17 features, two are SK_ID_CURR and SK_ID_BUREAU.
    <ul><li>SK_ID_BUREAU is the loan ID of the client's previous loan from other financial institutions. There may be multiple previous loans corresponding to a single SK_ID_CURR which depends on client's borrowing pattern.
    <li>SK_ID_CURR is the loan ID of client's current loan with Home Credit.
    <li>The rest of the features contain other stats such as DAYS_CREDIT, AMT_CREDIT_SUM, CREDIT_TYPE, etc.
    </ul></li>
    <li>There are 305k unique SK_ID_CURR in bureau out of which:
        <ul><li>There are 263k SK_ID_CURR in bureau which are present in application_train out of total of total 307k of application_train's unique SK_ID_CURR. This means that some of the applicants in current loan application with Home Credit Group do not have any previous Credit history with Credit Bureau Department.<br>
        <li>Similarly, there are 42.3k SK_ID_CURR in bureau which are present in application_test, out of total 48k of application_test's unique SK_ID_CURR.</ul>

#### NaN Columns and Percentages


```python
nan_df_bureau = nan_df_create(bureau)
print("-"*100)
plot_nan_percent(nan_df_bureau, 'bureau', tight_layout = False, figsize = (10,5))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    Number of columns having NaN values: 7 columns
    


    
![png](output_93_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol><li>Out of 17 features, there are 7 features which contain NaN values. 
    <li>The highest NaN values are observed with the column AMT_ANNUITY which has over 70% missing values.</ol>

<b>Merging the TARGETS from application_train to bureau table.</b>


```python
print("-"*100)
print("Merging TARGET with bureau Table")
bureau_merged = application_train.iloc[:,:2].merge(bureau, on = 'SK_ID_CURR', how = 'left')
print("-"*100)
```

    ----------------------------------------------------------------------------------------------------
    Merging TARGET with bureau Table
    ----------------------------------------------------------------------------------------------------
    

#### Phi-K Matrix


```python
cols_for_phik = ['TARGET','CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE']
plot_phik_matrix(bureau_merged, cols_for_phik,  figsize = (5,5))
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_98_1.png)
    


    ----------------------------------------------------------------------------------------------------
    Categories with highest values of Phi-K Correlation value with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CREDIT_ACTIVE</td>
      <td>0.064481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CREDIT_TYPE</td>
      <td>0.049954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CREDIT_CURRENCY</td>
      <td>0.004993</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

The above heatmap shows the Phi-K Correlation values between categorical values.
<ol>
<li>From the Phi-K Correlation Coefficient, we see that the variable CREDIT_TYPE shows some association with the variable CREDIT_ACTIVE.</li>
<li>We see that the Categorical Variables don't really have a high association with TARGET variable, especially the CREDIT_CURRENCY feature.</li>
    </ol>

#### Correlation Matrix of Features


```python
corr_mat = correlation_matrix(bureau_merged, ['SK_ID_CURR','SK_ID_BUREAU'], cmap = 'Blues', figsize = (12,10))
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_101_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in bureau table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-"*100)
```

    interval columns not set, guessing: ['TARGET', 'DAYS_CREDIT']
    interval columns not set, guessing: ['TARGET', 'CREDIT_DAY_OVERDUE']
    interval columns not set, guessing: ['TARGET', 'DAYS_CREDIT_ENDDATE']
    interval columns not set, guessing: ['TARGET', 'DAYS_ENDDATE_FACT']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_MAX_OVERDUE']
    interval columns not set, guessing: ['TARGET', 'CNT_CREDIT_PROLONG']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_SUM']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_SUM_DEBT']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_SUM_LIMIT']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_SUM_OVERDUE']
    interval columns not set, guessing: ['TARGET', 'DAYS_CREDIT_UPDATE']
    interval columns not set, guessing: ['TARGET', 'AMT_ANNUITY']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAYS_CREDIT</td>
      <td>0.088651</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAYS_CREDIT_ENDDATE</td>
      <td>0.018980</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AMT_CREDIT_SUM_OVERDUE</td>
      <td>0.005654</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AMT_CREDIT_SUM_LIMIT</td>
      <td>0.005192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AMT_CREDIT_MAX_OVERDUE</td>
      <td>0.004280</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNT_CREDIT_PROLONG</td>
      <td>0.003862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CREDIT_DAY_OVERDUE</td>
      <td>0.002528</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DAYS_CREDIT_UPDATE</td>
      <td>0.002219</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AMT_CREDIT_SUM_DEBT</td>
      <td>0.001695</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AMT_CREDIT_SUM</td>
      <td>0.000670</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The heatmap above shows the correlation between each feature in bureau table with the rest of the features.</li>
    <li>It can be observed that most of the heatmap has light colors, which shows little to no correlation.</li>
    <li>However, we can see some dark shades which represent high correlation.</li>
    <li>The high correlation is particularly observed for features:
        <ol><li>DAYS_CREDIT and DAYS_CREDIT_UPDATE</li>
            <li>DAYS_ENDDATE_FACT and DAYS_CREDIT_UPDATE</li>
            <li>AMT_CREDIT_SUM and AMT_CREDIT_SUM_DEBT</li>
            <li>DAYS_ENDDATE_FACT and DAYS_CREDIT</li></ol></li>
    <li>We can also see that the features don't particularly show good/high correlation with Target as such, except for DAYS_CREDIT feature. This implies that there isn't a direct linear relation between Target and the features.</li></ol>

#### Plotting Categorical Variables

We will now plot some of the Categorical Variables of the table bureau, and see their impact on the Target Variable.

<b><u>Distribution of Categorical Variable CREDIT_ACTIVE</u></b>

This column describes the Status of the previous loan reported from Credit Bureau.


```python
#let us first see the unique categories of 'CREDIT_ACTIVE'
print_unique_categories(bureau_merged, 'CREDIT_ACTIVE', show_counts = True)

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(bureau_merged, column_name = 'CREDIT_ACTIVE', horizontal_adjust = 0.3, fontsize_percent = 'x-small')
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'CREDIT_ACTIVE' are:
    ['Closed' 'Active' nan 'Sold' 'Bad debt']
    ----------------------------------------------------------------------------------------------------
    Counts of each category are:
    Closed      917733
    Active      541919
    Sold          5653
    Bad debt        20
    Name: CREDIT_ACTIVE, dtype: int64
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of CREDIT_ACTIVE = 5
    


    
![png](output_107_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above plot, we can draw the following insights:
<ol><li>From the first subplot, we see that a majority of the previous loans from other financial institutions are Closed Loans (62.63%), followed by 36.98% active loans. The sold and Bad-Debt Loans are very less in number.</li>
    <li>If we look at the Defaulters Percentage per category, we see that about 20% of people from Bad-Debt defaulted, which is the highest default rate. This is followed by Sold loans and Active Loans. The lowest default rate is for Closed Loans, which show a good history about a client. Thus the patten observed here is quite logical and expected.

#### Plotting Continuous Variables

<u><b>Distribution of Continuous Variable with DAYS Features</b></u>

<b>DAYS_CREDIT</b>

This column describes about the number of days before current application when the client applied for Credit Bureau Credit. For ease of interpretability, we will convert these days to years.


```python
bureau_merged['YEARS_CREDIT'] = bureau_merged['DAYS_CREDIT'] / -365
plot_continuous_variables(bureau_merged, 'YEARS_CREDIT', plots = ['distplot', 'box'], figsize = (15,8))
_ = bureau_merged.pop('YEARS_CREDIT')
```


    
![png](output_112_0.png)
    


##### Observations and Conclusions:

From the above plots, we see that:
<ol><li>From the PDF, we see that the Defaulters tend to have higher peaks compared to Non-Defaulters when the number of years are less.. This implies that the applicants who had applied for loans from Credit Bureau recently showed more defaulting tendency than those who had applied long ago. The PDF of Defaulters is also a bit towards left as compared to Non-Defaulters.</li>
    <li>Fro the box-plot as well, we see that Defaulters usually had less YEARS_CREDIT as compared to Non-Defaulters.

<b>DAYS_CREDIT_ENDDATE</b>

This column tells about the remaining duration of Credit Bureau credit at the time of application for loan in Home Credit.


```python
print_percentiles(bureau_merged, 'DAYS_CREDIT_ENDDATE', percentiles = list(range(0,11,2)) + [25,50,75,100])
plot_continuous_variables(bureau_merged, 'DAYS_CREDIT_ENDDATE', plots = ['box'], figsize = (8,6))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_CREDIT_ENDDATE is -42060.0
    The 2th percentile value of DAYS_CREDIT_ENDDATE is -2487.0
    The 4th percentile value of DAYS_CREDIT_ENDDATE is -2334.0
    The 6th percentile value of DAYS_CREDIT_ENDDATE is -2202.0
    The 8th percentile value of DAYS_CREDIT_ENDDATE is -2073.9199999999983
    The 10th percentile value of DAYS_CREDIT_ENDDATE is -1939.0
    The 25th percentile value of DAYS_CREDIT_ENDDATE is -1144.0
    The 50th percentile value of DAYS_CREDIT_ENDDATE is -334.0
    The 75th percentile value of DAYS_CREDIT_ENDDATE is 473.0
    The 100th percentile value of DAYS_CREDIT_ENDDATE is 31199.0
    ----------------------------------------------------------------------------------------------------
    


    
![png](output_115_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above percentile values, and looking at the box-plot, we see that there seems to be erroneous value for DAYS_CREDIT_ENDDATE, where the 0th percentile value dates back to as long as 42060 days or 115 years. This does not make much sense as this implies that the previous loan the client had dates back to 115 years ago. This could be inherited loan too, but we cannot comment so surely about that. We would try to remove these values in the data preprocessing stage.

<b>DAYS_ENDDATE_FACT</b>

This column tells about the the number of days ago that the Credit Bureau credit had ended at the time of application for loan in Home Credit. These values are only for Closed Credits.


```python
print_percentiles(bureau_merged, 'DAYS_ENDDATE_FACT', percentiles = list(range(0,11,2)) + [25,50,75,100])
plot_continuous_variables(bureau_merged, 'DAYS_ENDDATE_FACT', plots = ['box'], figsize = (8,8), scale_limits = [-40000, 0])
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_ENDDATE_FACT is -42023.0
    The 2th percentile value of DAYS_ENDDATE_FACT is -2561.0
    The 4th percentile value of DAYS_ENDDATE_FACT is -2450.0
    The 6th percentile value of DAYS_ENDDATE_FACT is -2351.0
    The 8th percentile value of DAYS_ENDDATE_FACT is -2265.0
    The 10th percentile value of DAYS_ENDDATE_FACT is -2173.0
    The 25th percentile value of DAYS_ENDDATE_FACT is -1503.0
    The 50th percentile value of DAYS_ENDDATE_FACT is -900.0
    The 75th percentile value of DAYS_ENDDATE_FACT is -427.0
    The 100th percentile value of DAYS_ENDDATE_FACT is 0.0
    ----------------------------------------------------------------------------------------------------
    


    
![png](output_118_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol><li>Just like previous column, we see that the 0th percentile for this column also seems erroneous, which is 42023 days or ~115 years. We would have to remove these values, as they don't make much sense.</li>
    <li>Looking at the box-plot, we see that the Defaulters tend to have lesser number of days since their Credit Bureau credit had ended. The Non-Defaulters usually have their previous credits ended longer before than Defaulters.

<b>DAYS_CREDIT_UPDATE</b>

This column tells about the the number of days ago that the information from Credit Bureau credit had come at the time of application for loan in Home Credit.


```python
print_percentiles(bureau_merged, 'DAYS_CREDIT_UPDATE', percentiles = list(range(0,11,2)) + [25,50,75,100])
plot_continuous_variables(bureau_merged, 'DAYS_CREDIT_UPDATE', plots = ['box'], figsize = (8,8), scale_limits = [-40000, 400])
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_CREDIT_UPDATE is -41947.0
    The 2th percentile value of DAYS_CREDIT_UPDATE is -2415.0
    The 4th percentile value of DAYS_CREDIT_UPDATE is -2213.0
    The 6th percentile value of DAYS_CREDIT_UPDATE is -2002.0
    The 8th percentile value of DAYS_CREDIT_UPDATE is -1766.0
    The 10th percentile value of DAYS_CREDIT_UPDATE is -1582.0
    The 25th percentile value of DAYS_CREDIT_UPDATE is -904.0
    The 50th percentile value of DAYS_CREDIT_UPDATE is -406.0
    The 75th percentile value of DAYS_CREDIT_UPDATE is -33.0
    The 100th percentile value of DAYS_CREDIT_UPDATE is 372.0
    ----------------------------------------------------------------------------------------------------
    


    
![png](output_121_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol><li>The trend of erroneious values is again very similar to the other days column where this 0th percentile value seems to be erroneous. Also since only the 0th percentile value is so odd, and the rest seem to be fine, thus this value is definitely erroneous. We will be removing this value too. 
    <li>From the box-plot, we can say that the Defaulters tend to have a lesser number of days since their Information about the Credit Bureau Credit were received. Their median, 75th percentile values all are lesser than those for Non-Defaulters.</ol>

### bureau_balance.csv

##### Description

This table consists of Monthly balance of each credit for each of the previous credit that the client had with financial institutions other than Home Credit.

<h4>Basic Stats</h4>


```python
print(f'The shape of bureau_balance.csv is: {bureau_balance.shape}')
print('-'*100)
print(f'Number of duplicate values in bureau_balance: {bureau_balance.shape[0] - bureau_balance.duplicated().shape[0]}')
print('-'*100)
display(bureau_balance.head(5))
```

    The shape of bureau_balance.csv is: (27299925, 3)
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in bureau_balance: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_BUREAU</th>
      <th>MONTHS_BALANCE</th>
      <th>STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5715448</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5715448</td>
      <td>-1</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5715448</td>
      <td>-2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5715448</td>
      <td>-3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5715448</td>
      <td>-4</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("-"*100)
print(f'Number of unique SK_ID_BUREAU in bureau_balance.csv are: {len(bureau_balance.SK_ID_BUREAU.unique())}')
print('-'*100)
print(f'Number of unique values for STATUS are: {len(bureau_balance.STATUS.unique())}')
print(f"Unique values of STATUS are:\n{bureau_balance.STATUS.unique()}")
print('-'*100)
print(f"Max number of months for Months Balance: {np.abs(bureau_balance.MONTHS_BALANCE.min())}")
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_BUREAU in bureau_balance.csv are: 817395
    ----------------------------------------------------------------------------------------------------
    Number of unique values for STATUS are: 8
    Unique values of STATUS are:
    ['C' '0' 'X' '1' '2' '3' '5' '4']
    ----------------------------------------------------------------------------------------------------
    Max number of months for Months Balance: 96
    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

<ol><li>The bureau_balance.csv table contains approximately 27.29M rows, and 3 columns.</li>
    <li>This table contains the monthly status for each of the previous loan for a particular applicant reported by the Credit Bureau Department.
    <li>There are 8 unique values for the STATUS which are encoded. Each of them have a special meaning. <br>
        C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,… 5 means DPD 120+ or sold or written off.
     <li>The most earliest month's balance that we have is the 96 months back status, i.e. the Status has been provided upto 8 years of history for loans for which those exist.

<h4>NaN Columns and Percentages</h4>


```python
plot_nan_percent(nan_df_create(bureau_balance), 'bureau_balance')
```

    The dataframe bureau_balance does not contain any NaN values.
    

### previous_application.csv

##### Description

This table contains the static data of the previous loan which the client had with Home Credit.

#### Basic Stats


```python
print(f'The shape of previous_application.csv is: {previous_application.shape}')
print('-'*100)
print(f'Number of unique SK_ID_PREV in previous_application.csv are: {len(previous_application.SK_ID_PREV.unique())}')
print(f'Number of unique SK_ID_CURR in previous_application.csv are: {len(previous_application.SK_ID_CURR.unique())}')
print('-'*100)
print(f'Number of overlapping SK_ID_CURR in application_train.csv and previous_application.csv are: {len(set(application_train.SK_ID_CURR.unique()).intersection(set(previous_application.SK_ID_CURR.unique())))}')
print(f'Number of overlapping SK_ID_CURR in application_test.csv and previous_application.csv are: {len(set(application_test.SK_ID_CURR.unique()).intersection(set(previous_application.SK_ID_CURR.unique())))}')
print('-'*100)
print(f'Number of duplicate values in previous_application: {previous_application.shape[0] - previous_application.duplicated().shape[0]}')
print('-'*100)
display(previous_application.head(5))
```

    The shape of previous_application.csv is: (1670214, 37)
    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_PREV in previous_application.csv are: 1670214
    Number of unique SK_ID_CURR in previous_application.csv are: 338857
    ----------------------------------------------------------------------------------------------------
    Number of overlapping SK_ID_CURR in application_train.csv and previous_application.csv are: 291057
    Number of overlapping SK_ID_CURR in application_test.csv and previous_application.csv are: 47800
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in previous_application: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>FLAG_LAST_APPL_PER_CONTRACT</th>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <th>RATE_DOWN_PAYMENT</th>
      <th>RATE_INTEREST_PRIMARY</th>
      <th>RATE_INTEREST_PRIVILEGED</th>
      <th>NAME_CASH_LOAN_PURPOSE</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>DAYS_DECISION</th>
      <th>NAME_PAYMENT_TYPE</th>
      <th>CODE_REJECT_REASON</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_CLIENT_TYPE</th>
      <th>NAME_GOODS_CATEGORY</th>
      <th>NAME_PORTFOLIO</th>
      <th>NAME_PRODUCT_TYPE</th>
      <th>CHANNEL_TYPE</th>
      <th>SELLERPLACE_AREA</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>Y</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.182832</td>
      <td>0.867336</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-73</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>Mobile</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Country-wide</td>
      <td>35</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>Y</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-164</td>
      <td>XNA</td>
      <td>XAP</td>
      <td>Unaccompanied</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Contact center</td>
      <td>-1</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>Y</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-301</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Spouse, partner</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Credit and cash offices</td>
      <td>-1</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>Y</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-512</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Credit and cash offices</td>
      <td>-1</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>Y</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Repairs</td>
      <td>Refused</td>
      <td>-781</td>
      <td>Cash through the bank</td>
      <td>HC</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>-1</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


##### Observations and Conclusions:

<ol><li>The table previous_application.csv consists of 1.67M rows in total. Each row corresponds to each of the previous loan that the client had with previously with Home Credit Group. It is possible for a single client of current application to have multiple previous loans with Home Credit Group.</li>
    <li>There are 37 columns in previous_application.csv, which contain the details about the previous loan.</li>
    <li>There are 338k unique SK_ID_CURR in previous_application, of which 291k correspond to the application_train SK_ID_CURRs and 47.8k correspond to application_test SK_ID_CURRs.</li></ol>

#### NaN Columns and Percentages


```python
previous_application_nan = nan_df_create(previous_application)
print('-' * 100)
plot_nan_percent(previous_application_nan, 'previous_application', tight_layout = False, figsize = (13,5))
print('-' * 100)
del previous_application_nan
```

    ----------------------------------------------------------------------------------------------------
    Number of columns having NaN values: 16 columns
    


    
![png](output_137_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

<ol><li>There are 16 columns out of the 37 columns which contain NaN values.</li>
    <li>Two of these columns have 99.64% missing values, which is very high, and we will have to come up with some smart way to handle such high NaN values. We cannot directly discard any feature at this point.</li>
    <li>Other than these two columns, rest of the columns also contain > 40% NaN values, except for 5 columns. </li></ol>

<b>Merging the TARGETS from application_train to previous_application table.</b>


```python
print("-"*100)
print("Merging TARGET with previous_application Table")
prev_merged = application_train.iloc[:,:2].merge(previous_application, on = 'SK_ID_CURR', how = 'left')
print("-"*100)
```

    ----------------------------------------------------------------------------------------------------
    Merging TARGET with previous_application Table
    ----------------------------------------------------------------------------------------------------
    

#### Phi-K Matrix


```python
cols_for_phik = ['TARGET'] + prev_merged.dtypes[prev_merged.dtypes == 'object'].index.tolist() + ['NFLAG_INSURED_ON_APPROVAL']
plot_phik_matrix(prev_merged, cols_for_phik, cmap = 'Blues', figsize = (11,9), fontsize = 9)
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_142_1.png)
    


    ----------------------------------------------------------------------------------------------------
    Categories with highest values of Phi-K Correlation value with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>NAME_CONTRACT_STATUS</td>
      <td>0.088266</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PRODUCT_COMBINATION</td>
      <td>0.063839</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CODE_REJECT_REASON</td>
      <td>0.062771</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NAME_CONTRACT_TYPE</td>
      <td>0.050859</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CHANNEL_TYPE</td>
      <td>0.050302</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NAME_GOODS_CATEGORY</td>
      <td>0.042951</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NAME_CASH_LOAN_PURPOSE</td>
      <td>0.040305</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NAME_PAYMENT_TYPE</td>
      <td>0.039752</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NAME_SELLER_INDUSTRY</td>
      <td>0.038077</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NAME_YIELD_GROUP</td>
      <td>0.034626</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above heatmap of values of Phi-K Coefficient for Categorical Features, we observe that:

<ol><li>The feature PRODUCT_COMBINATION shows association with lots of other features such as NAME_CONTRACT_TYPE, NAME_PRODUCT_TYPE, NAME_PORTFOLIO, etc.</li>
    <li>The feature NAME_GOODS_CATEGORY is also highly associated with NAME_SELLER_INDUSTRY</li>
    <li>If we look at the association with TARGET variable, we see that the features NAME_CONTRACT_STATUS, PRODUCT_COMBINATION, CODE_REJECT_REASON are some of the highest associated features, and would need further investigation</li></ol>
   

#### Correlation Matrix of Features


```python
corr_mat = correlation_matrix(prev_merged, ['SK_ID_CURR','SK_ID_PREV','NFLAG_INSURED_ON_APPROVAL'], cmap = 'Blues', figsize = (14,12))
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_145_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in previous_applications table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-"*100)
```

    interval columns not set, guessing: ['TARGET', 'AMT_ANNUITY']
    interval columns not set, guessing: ['TARGET', 'AMT_APPLICATION']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT']
    interval columns not set, guessing: ['TARGET', 'AMT_DOWN_PAYMENT']
    interval columns not set, guessing: ['TARGET', 'AMT_GOODS_PRICE']
    interval columns not set, guessing: ['TARGET', 'HOUR_APPR_PROCESS_START']
    interval columns not set, guessing: ['TARGET', 'NFLAG_LAST_APPL_IN_DAY']
    interval columns not set, guessing: ['TARGET', 'RATE_DOWN_PAYMENT']
    interval columns not set, guessing: ['TARGET', 'RATE_INTEREST_PRIMARY']
    interval columns not set, guessing: ['TARGET', 'RATE_INTEREST_PRIVILEGED']
    interval columns not set, guessing: ['TARGET', 'DAYS_DECISION']
    interval columns not set, guessing: ['TARGET', 'SELLERPLACE_AREA']
    interval columns not set, guessing: ['TARGET', 'CNT_PAYMENT']
    interval columns not set, guessing: ['TARGET', 'DAYS_FIRST_DRAWING']
    interval columns not set, guessing: ['TARGET', 'DAYS_FIRST_DUE']
    interval columns not set, guessing: ['TARGET', 'DAYS_LAST_DUE_1ST_VERSION']
    interval columns not set, guessing: ['TARGET', 'DAYS_LAST_DUE']
    interval columns not set, guessing: ['TARGET', 'DAYS_TERMINATION']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>CNT_PAYMENT</td>
      <td>0.056639</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DAYS_DECISION</td>
      <td>0.053694</td>
    </tr>
    <tr>
      <th>13</th>
      <td>DAYS_FIRST_DRAWING</td>
      <td>0.048993</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RATE_DOWN_PAYMENT</td>
      <td>0.039592</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HOUR_APPR_PROCESS_START</td>
      <td>0.038121</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RATE_INTEREST_PRIVILEGED</td>
      <td>0.028204</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DAYS_LAST_DUE_1ST_VERSION</td>
      <td>0.027878</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DAYS_LAST_DUE</td>
      <td>0.027320</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DAYS_TERMINATION</td>
      <td>0.026479</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AMT_ANNUITY</td>
      <td>0.013808</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The heatmap above shows the correlation between each feature in previous_application table with the rest of the features.</li>
    <li>From the heatmap, we can see some highly correlated features which have a darker colour.</li>
    <li>The high correlation is particularly observed for features:
        <ol><li>DAYS_TERMINATION and DAYS_LAST_DUE</li>
            <li>AMT_CREDIT and AMT_APPLICATION</li>
            <li>AMT_APPLICATION and AMT_GOODS_PRICE</li>
            <li>AMT_CREDIT and AMT_ANNUITY</li>
            <li>AMT_ANNUITY and AMT_CREDIT</li>
            <li>AMT_CREDIT and AMT_GOODS_PRICE</li></ol></li>
    <li>We can also see that the features don't particularly show good/high correlation with Target as such. This implies that there isn't much of a direct linear relation between Target and the features.</li></ol>

#### Plotting Categorical Variables

Let us now plot some of the Categorical Variables of table previous_application and see how they impact the Target Variable.

<b><u>Distribution of Categorical Variable NAME_CONTRACT_TYPE</u></b>

This column descibes the type of the Contract of the previous loan with the Home Credit Group.


```python
#let us first see the unique categories of 'NAME_CONTRACT_TYPE'
print_unique_categories(prev_merged, 'NAME_CONTRACT_TYPE', show_counts = True)

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(prev_merged, 'NAME_CONTRACT_TYPE', horizontal_adjust = 0.3, figsize = (20, 6))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'NAME_CONTRACT_TYPE' are:
    ['Consumer loans' 'Cash loans' 'Revolving loans' nan 'XNA']
    ----------------------------------------------------------------------------------------------------
    Counts of each category are:
    Cash loans         626764
    Consumer loans     625256
    Revolving loans    161368
    XNA                   313
    Name: NAME_CONTRACT_TYPE, dtype: int64
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of NAME_CONTRACT_TYPE = 5
    


    
![png](output_151_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above plots, we can observe the following:
<ol><li>From the first subplot, we see that most of the previous loans have been either Cash Loans or Consumer Loans, which correspond to roughly 44% of loans each. The remaining 11.41% corresponds to Revolving Loans, and there are some loans named XNA whose types are actually not known, but they are very few in numbers.</li>
    <li>Looking at the second subplot, we see that the Percentage of Defaulters for XNA type of loan are the highest, at 20% Default rate. The next highest Default Rate is among Revolving Loans, which is close to 10.5%.</li>
    <li>The Cash Loans have lesser default rates, roughly 9% while the consumer loans tend to have the lowest Percentage of Defaulters, which is close to 7.5%.</li></ol>

<b><u>Distribution of Categorical Variable NAME_CONTRACT_STATUS</u></b>

This column describes the status of the contract of the previous loan with Home Credit, i.e. whether it is active or closed, etc.


```python
#let us first see the unique categories of 'NAME_CONTRACT_STATUS'
print_unique_categories(prev_merged, 'NAME_CONTRACT_STATUS')

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(prev_merged, 'NAME_CONTRACT_STATUS', horizontal_adjust = 0.25, figsize = (20, 6), fontsize_percent = 'small')
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'NAME_CONTRACT_STATUS' are:
    ['Approved' 'Canceled' 'Refused' nan 'Unused offer']
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of NAME_CONTRACT_STATUS = 5
    


    
![png](output_154_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

From the above plots, we see that:
<ol><li>The most common type of Contract Status is the Approved Status. About 63% of the previous Credits have an Approved Status. The next two common status are Canceled and Refused, which both correspond to about 18% of the loans. This implies that most of the loans get approved and only some fraction of them do not. The least occurring type of contract status is Unused Offer which corresponds to just 1.61% of all the loans.</li>
    <li>Looking at the second subplot for percentage of defaulters, we see that the those loans which previously had Refused Status tend to have defaulted the highest in the current loans. They correspond to about 12% of Defaulters from that category. These are followed by Canceled Status which correspond to close to 9% of Default Rate. This behavious is quite expected logically, as these people must have been refused due to not having adequate profile. The least default rate is observed for Contract Status of Approved.</li></ol>

<b><u>Distribution of Categorical Variable CODE_REJECT_REASON</u></b>

This column describes the reason of the rejection of previously applied loan in Home Credit Group.


```python
#let us first see the unique categories of 'CODE_REJECT_REASON'
print_unique_categories(prev_merged, 'CODE_REJECT_REASON', show_counts = True)

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(prev_merged, 'CODE_REJECT_REASON', horizontal_adjust = 0.18, figsize = (20, 6))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'CODE_REJECT_REASON' are:
    ['XAP' 'LIMIT' nan 'HC' 'SCO' 'SCOFR' 'VERIF' 'CLIENT' 'XNA' 'SYSTEM']
    ----------------------------------------------------------------------------------------------------
    Counts of each category are:
    XAP       1145533
    HC         145984
    LIMIT       47773
    SCO         32636
    CLIENT      22771
    SCOFR       10875
    XNA          4378
    VERIF        3079
    SYSTEM        672
    Name: CODE_REJECT_REASON, dtype: int64
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of CODE_REJECT_REASON = 10
    


    
![png](output_157_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

The above plot shows the distribution of the Categorical variable CODE_REJECT_REASON. Following insights can be generated from the above plot:
<ol><li>The most common type of reason of rejection is XAP, which is about ~81%. The other reasons form only a small part of the rejection reasons. HC is the second highest rejection reason with just 10.33% of occurrences.</li>
    <li>The distribution of percentage of defaulters for each category of CODE_REJECT_REASON in quite interesting. Those applicants who had their previous applications rejected by Code SCOFT have the highest percentage of Defaulters among them (~21%). This is followed by LIMIT and HC which have around 12.5% and 12% of Defaulters.</li>
    <li>The most common occurring rejection reason XAP corresponds to only 7.5% of Defaulters of all, and is the second lowest percentage of Defaulters after SYSTEM code.</li></ol>

<b><u>Distribution of Categorical Variable CHANNEL_TYPE</u></b>

This column describes the channel through which the client was acquired for the previous loan in Home Credit.


```python
#let us first see the unique categories of 'CHANNEL_TYPE'
print_unique_categories(prev_merged, 'CHANNEL_TYPE')

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(prev_merged, 'CHANNEL_TYPE', horizontal_adjust = 0.15, rotation = 45, figsize = (20, 6), fontsize_percent = 'x-small')
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'CHANNEL_TYPE' are:
    ['Stone' 'Credit and cash offices' 'Country-wide' 'Regional / Local'
     'AP+ (Cash loan)' 'Contact center' nan 'Channel of corporate sales'
     'Car dealer']
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of CHANNEL_TYPE = 9
    


    
![png](output_160_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

The above two plots show the distribution of CHANNEL_TYPE for previous loans in Home Credit.
<ol><li>From the first subplot we see that most of the applications were acquired through the Credit and cash offices which were roughly 42.47% applications, which were followed by Country-wide channel corresponding to 29.93% applications. Rest of the channel types corresponded to only a select number of applications.</li>
    <li>The highest Defaulting Percentage was seen among applications who had a channel type of AP+ (Cash loan) which corresponded to about 13% defaulters in that category. The rest of the channels had lower default percentages than this one. The channel Car Dealer showed a lowest Percentage of Defaulters in that category (only 5%).

<b><u>Distribution of Categorical Variable PRODUCT_COMBINATION</u></b>

This column gives details about the product combination of the previous applications.


```python
#let us first see the unique categories of 'PRODUCT_COMBINATION'
print_unique_categories(prev_merged, 'PRODUCT_COMBINATION')

# plotting the Bar Plot for the Column
plot_categorical_variables_bar(prev_merged, 'PRODUCT_COMBINATION', rotation = 90, figsize = (20, 6))
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The unique categories of 'PRODUCT_COMBINATION' are:
    ['POS other with interest' 'Cash X-Sell: low' 'POS industry with interest'
     'POS household with interest' 'POS mobile without interest' 'Card Street'
     'Card X-Sell' 'Cash X-Sell: high' 'Cash' 'Cash Street: high'
     'Cash X-Sell: middle' 'POS mobile with interest'
     'POS household without interest' 'POS industry without interest'
     'Cash Street: low' nan 'Cash Street: middle'
     'POS others without interest']
    ----------------------------------------------------------------------------------------------------
    Total Number of unique categories of PRODUCT_COMBINATION = 18
    


    
![png](output_163_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

From the distribution of PRODUCT_COMBINATION, we can generate following insights:
<ol><li>The 3 most common types of Product Combination are Cash, POS household with interest and POS mobile with interest. They correspond to roughly 50% of all the applications. </li>
    <li>Looking at the Percentage of Defaulters per category plot, we see a highest defaulting tendency among Cash Street: mobile category, Cash X-sell: high, Cash Street: high and Card Street which all are near about 11-11.5% defaulters per category. The lowest Percentage of Defaulters are in the POS Industry without interest Category, which correspond to about 4.5% Defaulters.

#### Plotting Continuous Variables

<u><b>Distribution of Continuous Variable with DAYS Features</b></u>

<b>DAYS_DECISION</b>

This column tells about the number of days relative to the current application when the decision was made about previous application.


```python
plot_continuous_variables(prev_merged, 'DAYS_DECISION', plots = ['distplot', 'box'], figsize = (15,8))
```


    
![png](output_168_0.png)
    


##### Observations and Conclusions

From the above plot, we notice that for Defaulters, the number of days back when the decision was made is a bit lesser than that for Non-Defaulters. This implies that the Defaulters usually had the decision on their previous applications made more recently as compared to Non-Defaulters.

<b>DAYS_FIRST_DRAWING</b>

This column tells about the number of days back from current application that the first disbursement of the previous application was made.


```python
print_percentiles(prev_merged, 'DAYS_FIRST_DRAWING', percentiles = list(range(0,11)) + list(range(20,101,20)))
plot_continuous_variables(prev_merged, 'DAYS_FIRST_DRAWING', plots = ['box'], figsize = (8,6), scale_limits = [-3000,0])
print('-'*100)
```

    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_FIRST_DRAWING is -2922.0
    The 1th percentile value of DAYS_FIRST_DRAWING is -2451.0
    The 2th percentile value of DAYS_FIRST_DRAWING is -1179.0
    The 3th percentile value of DAYS_FIRST_DRAWING is -674.0
    The 4th percentile value of DAYS_FIRST_DRAWING is -406.0
    The 5th percentile value of DAYS_FIRST_DRAWING is -262.0
    The 6th percentile value of DAYS_FIRST_DRAWING is -156.0
    The 7th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 8th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 9th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 10th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 20th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 40th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 60th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 80th percentile value of DAYS_FIRST_DRAWING is 365243.0
    The 100th percentile value of DAYS_FIRST_DRAWING is 365243.0
    ----------------------------------------------------------------------------------------------------
    


    
![png](output_171_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol><li>Looking at the percentile values of DAYS_FIRST_DRAWING, it seems like most of the values are erroneous, starting from 7th percentile values itself. These erroneous values will needed to be dopped.</li>
    <li>If we try to analyze the distribution of this column by removing the erroneous ponts, we see that most of the Defaulters had their First Drawing on previous credit more recently as compared to Non-Defaulters. The 75th percentile value for Defaulters is also significantly lesser than that of Non-Defaulters.

<b>DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE, and DAYS_TERMINATION</b>

These columns also decribe about the number of days ago from the current application that certain activities happened. 


```python
print('-'*100)
print("Percentile Values for DAYS_FIRST_DUE")
print_percentiles(prev_merged, 'DAYS_FIRST_DUE', percentiles = list(range(0,11,2)) + [20,40,60,80,100])
print("Percentile Values for DAYS_LAST_DUE_1ST_VERSION")
print_percentiles(prev_merged, 'DAYS_LAST_DUE_1ST_VERSION', percentiles = list(range(0,11,2)) + [20,40,60,80,100])
print("Percentile Values for DAYS_LAST_DUE")
print_percentiles(prev_merged, 'DAYS_LAST_DUE', percentiles = list(range(0,11,2)) + [20,40,60,80,100])
print("Percentile Values for DAYS_TERMINATION")
print_percentiles(prev_merged, 'DAYS_TERMINATION', percentiles = list(range(0,11,2)) + [20,40,60,80,100])
```

    ----------------------------------------------------------------------------------------------------
    Percentile Values for DAYS_FIRST_DUE
    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_FIRST_DUE is -2892.0
    The 2th percentile value of DAYS_FIRST_DUE is -2759.0
    The 4th percentile value of DAYS_FIRST_DUE is -2648.0
    The 6th percentile value of DAYS_FIRST_DUE is -2555.0
    The 8th percentile value of DAYS_FIRST_DUE is -2471.0
    The 10th percentile value of DAYS_FIRST_DUE is -2388.0
    The 20th percentile value of DAYS_FIRST_DUE is -1882.0
    The 40th percentile value of DAYS_FIRST_DUE is -1070.0
    The 60th percentile value of DAYS_FIRST_DUE is -647.0
    The 80th percentile value of DAYS_FIRST_DUE is -329.0
    The 100th percentile value of DAYS_FIRST_DUE is 365243.0
    ----------------------------------------------------------------------------------------------------
    Percentile Values for DAYS_LAST_DUE_1ST_VERSION
    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2801.0
    The 2th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2516.0
    The 4th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2380.0
    The 6th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2267.0
    The 8th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2159.0
    The 10th percentile value of DAYS_LAST_DUE_1ST_VERSION is -2045.0
    The 20th percentile value of DAYS_LAST_DUE_1ST_VERSION is -1498.0
    The 40th percentile value of DAYS_LAST_DUE_1ST_VERSION is -644.0
    The 60th percentile value of DAYS_LAST_DUE_1ST_VERSION is -146.0
    The 80th percentile value of DAYS_LAST_DUE_1ST_VERSION is 273.0
    The 100th percentile value of DAYS_LAST_DUE_1ST_VERSION is 365243.0
    ----------------------------------------------------------------------------------------------------
    Percentile Values for DAYS_LAST_DUE
    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_LAST_DUE is -2889.0
    The 2th percentile value of DAYS_LAST_DUE is -2534.0
    The 4th percentile value of DAYS_LAST_DUE is -2400.0
    The 6th percentile value of DAYS_LAST_DUE is -2290.0
    The 8th percentile value of DAYS_LAST_DUE is -2186.0
    The 10th percentile value of DAYS_LAST_DUE is -2079.0
    The 20th percentile value of DAYS_LAST_DUE is -1554.0
    The 40th percentile value of DAYS_LAST_DUE is -784.0
    The 60th percentile value of DAYS_LAST_DUE is -333.0
    The 80th percentile value of DAYS_LAST_DUE is 365243.0
    The 100th percentile value of DAYS_LAST_DUE is 365243.0
    ----------------------------------------------------------------------------------------------------
    Percentile Values for DAYS_TERMINATION
    ----------------------------------------------------------------------------------------------------
    The 0th percentile value of DAYS_TERMINATION is -2874.0
    The 2th percentile value of DAYS_TERMINATION is -2523.0
    The 4th percentile value of DAYS_TERMINATION is -2383.0
    The 6th percentile value of DAYS_TERMINATION is -2270.0
    The 8th percentile value of DAYS_TERMINATION is -2161.0
    The 10th percentile value of DAYS_TERMINATION is -2048.0
    The 20th percentile value of DAYS_TERMINATION is -1508.0
    The 40th percentile value of DAYS_TERMINATION is -743.0
    The 60th percentile value of DAYS_TERMINATION is -297.0
    The 80th percentile value of DAYS_TERMINATION is 365243.0
    The 100th percentile value of DAYS_TERMINATION is 365243.0
    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

From all of the above percentile values, we realise that all the Days columns have these erroneous values somewhere or the other. Thus these values need to be replaced so that our model doesn't get affected by these.

### installments_payments.csv

##### Description

This table lists out the repayment history of each of the loan that the applicant had with Home Credit Group. The table contains features like the amount of instalment, how much did the client pay for each instalments, etc.

<h4>Basic Stats</h4>


```python
print(f'The shape of installments_payments.csv is: {installments_payments.shape}')
print('-'*100)
print(f'Number of unique SK_ID_PREV in installments_payments.csv are: {len(installments_payments.SK_ID_PREV.unique())}')
print(f'Number of unique SK_ID_CURR in installments_payments.csv are: {len(installments_payments.SK_ID_CURR.unique())}')
print('-'*100)
print(f'Number of overlapping SK_ID_CURR in application_train.csv and installments_payments.csv are: {len(set(application_train.SK_ID_CURR.unique()).intersection(set(installments_payments.SK_ID_CURR.unique())))}')
print(f'Number of overlapping SK_ID_CURR in application_test.csv and installments_payments.csv are: {len(set(application_test.SK_ID_CURR.unique()).intersection(set(installments_payments.SK_ID_CURR.unique())))}')
print('-'*100)
print(f'Number of duplicate values in installments_payments: {installments_payments.shape[0] - installments_payments.duplicated().shape[0]}')
print('-'*100)
display(installments_payments.head(5))
```

    The shape of installments_payments.csv is: (13605401, 8)
    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_PREV in installments_payments.csv are: 997752
    Number of unique SK_ID_CURR in installments_payments.csv are: 339587
    ----------------------------------------------------------------------------------------------------
    Number of overlapping SK_ID_CURR in application_train.csv and installments_payments.csv are: 291643
    Number of overlapping SK_ID_CURR in application_test.csv and installments_payments.csv are: 47944
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in installments_payments: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NUM_INSTALMENT_VERSION</th>
      <th>NUM_INSTALMENT_NUMBER</th>
      <th>DAYS_INSTALMENT</th>
      <th>DAYS_ENTRY_PAYMENT</th>
      <th>AMT_INSTALMENT</th>
      <th>AMT_PAYMENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1054186</td>
      <td>161674</td>
      <td>1.0</td>
      <td>6</td>
      <td>-1180.0</td>
      <td>-1187.0</td>
      <td>6948.360</td>
      <td>6948.360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1330831</td>
      <td>151639</td>
      <td>0.0</td>
      <td>34</td>
      <td>-2156.0</td>
      <td>-2156.0</td>
      <td>1716.525</td>
      <td>1716.525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2085231</td>
      <td>193053</td>
      <td>2.0</td>
      <td>1</td>
      <td>-63.0</td>
      <td>-63.0</td>
      <td>25425.000</td>
      <td>25425.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2452527</td>
      <td>199697</td>
      <td>1.0</td>
      <td>3</td>
      <td>-2418.0</td>
      <td>-2426.0</td>
      <td>24350.130</td>
      <td>24350.130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2714724</td>
      <td>167756</td>
      <td>1.0</td>
      <td>2</td>
      <td>-1383.0</td>
      <td>-1366.0</td>
      <td>2165.040</td>
      <td>2160.585</td>
    </tr>
  </tbody>
</table>
</div>


##### Observations and Conclusions

<ol><li>There are about 13.6M datapoints in the table installments_payments.csv. Each row represents each installment history related to the a particular loan that the client previously had with Home Credit Group.</li>
    <li>There are 997k unique previous loans in the installments_payments. These belong to 339k unique SK_ID_CURR, which are ID of applicants of current loan.</li>
    <li>Out of these 339k SK_ID_CURR, 291k belong to the training dataset, and 47.9k belong to the test dataset. This implies that almost out of 307k unique SK_ID_CURR in application_train, 291k previously had some form of loan with Home Credit. Similarly for 48.7k of those in test dataset, 47.9k had loan previously with Home Credit.</li>
    <li>The table has 8 unique features, 6 of which describe the statistics of each installment for previous loan. </li></ol>

#### NaN Columns and Percentages


```python
print('-'*100)
print("Columns with NaN values and their percentages:")
installments_payments_nan = nan_df_create(installments_payments)
display(installments_payments_nan[installments_payments_nan.percent != 0])
print('-'*100)
del installments_payments_nan
```

    ----------------------------------------------------------------------------------------------------
    Columns with NaN values and their percentages:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>DAYS_ENTRY_PAYMENT</td>
      <td>0.021352</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AMT_PAYMENT</td>
      <td>0.021352</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

<ol><li>There are only 2 columns which contain NaN values of the 8 columns from installments_payments.</li>
    <li>These columns also contain very minimal proportion of NaN values, i.e only 0.02%, so it is not of much concern.</li></ol>

<b>Merging the TARGETS from application_train to installments_payments table.</b>


```python
print("-"*100)
print("Merging TARGET with installments_payments Table")
installments_merged = application_train.iloc[:,:2].merge(installments_payments, on = 'SK_ID_CURR', how = 'left')
print("-"*100)
```

    ----------------------------------------------------------------------------------------------------
    Merging TARGET with installments_payments Table
    ----------------------------------------------------------------------------------------------------
    

#### Correlation Matrix of Features


```python
corr_mat = correlation_matrix(installments_merged, ['SK_ID_CURR','SK_ID_PREV'], figsize = (8,7))
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_187_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in installments_payments table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-"*100)
```

    interval columns not set, guessing: ['TARGET', 'NUM_INSTALMENT_VERSION']
    interval columns not set, guessing: ['TARGET', 'NUM_INSTALMENT_NUMBER']
    interval columns not set, guessing: ['TARGET', 'DAYS_INSTALMENT']
    interval columns not set, guessing: ['TARGET', 'DAYS_ENTRY_PAYMENT']
    interval columns not set, guessing: ['TARGET', 'AMT_INSTALMENT']
    interval columns not set, guessing: ['TARGET', 'AMT_PAYMENT']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>DAYS_INSTALMENT</td>
      <td>0.046824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DAYS_ENTRY_PAYMENT</td>
      <td>0.033128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NUM_INSTALMENT_NUMBER</td>
      <td>0.022993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AMT_INSTALMENT</td>
      <td>0.004125</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AMT_PAYMENT</td>
      <td>0.003084</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NUM_INSTALMENT_VERSION</td>
      <td>0.002198</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The heatmap above shows the correlation between each feature in installments_payments table with the rest of the features.</li>
    <li>From the heatmap of correlation matrix, we see a couple of highly correlated features. These are:
        <ul><li>AMT_INSTALMENT and AMT_PAYMENT</li>
            <li>DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT</li>
        </ul></li>
    <li>These two sets of correlated features are understandable, as they are actually the features as to when the installment was due to be paid vs when it was paid and also the amount that was due vs the amount that was paid.</li>
    <li>These features will be useful for creating new sets of completely uncorrelated features.</li>
    <li>The correlation of features with Target isn't noticeable, this shows the absence of a linear relationship between the feature and the target variable.</li></ol>

#### Plotting Continuous Variables

Firstly we will group by the 'SK_ID_PREV' field and aggregate with mean, so that we get an averaged row for each of the previous loan that the client had.


```python
installments_merged = installments_merged.groupby('SK_ID_PREV').mean()
```

<b><u>Distribution of Continuous Vairable DAYS_INSTALMENT</u></b>

This column lists the days when the installment of previous credit was to be paid.


```python
plot_continuous_variables(installments_merged, 'DAYS_INSTALMENT', plots = ['box'], figsize = (8,8))
```


    
![png](output_194_0.png)
    


<b><u>Distribution of Continuous Vairable DAYS_ENTRY_PAYMENT</u></b>

This column lists the days when the installment of previous credit was actually paid.


```python
plot_continuous_variables(installments_merged, 'DAYS_ENTRY_PAYMENT', plots = ['box'], figsize = (8,8))
del installments_merged
```


    
![png](output_196_0.png)
    


##### Observations and Conclusions

From the above two plots, we can see a similar pattern, where the Defaulters tend to have lesser number of days since their last payment, while Non-Defaulters have more number of days since their last payments. All quantiles of Defaulters have more recent days than those of Non-Defaulters. Thus, Non-Defaulters usually have more gap in their payments from the day of application as compared to Defaulters.

### POS_CASH_balance.csv

##### Description

This table contains the Monthly Balance Snapshots of previous Point of Sales and Cash Loans that the applicant had with Home Credit Group. The table contains columns like the status of contract, the number of installments left, etc.

#### Basic Stats


```python
print(f'The shape of POS_CASH_balance.csv is: {POS_CASH_balance.shape}')
print('-'*100)
print(f'Number of unique SK_ID_PREV in POS_CASH_balance.csv are: {len(POS_CASH_balance.SK_ID_PREV.unique())}')
print(f'Number of unique SK_ID_CURR in POS_CASH_balance.csv are: {len(POS_CASH_balance.SK_ID_CURR.unique())}')
print('-'*100)
print(f'Number of overlapping SK_ID_CURR in application_train.csv and POS_CASH_balance.csv are: {len(set(application_train.SK_ID_CURR.unique()).intersection(set(POS_CASH_balance.SK_ID_CURR.unique())))}')
print(f'Number of overlapping SK_ID_CURR in application_test.csv and POS_CASH_balance.csv are: {len(set(application_test.SK_ID_CURR.unique()).intersection(set(POS_CASH_balance.SK_ID_CURR.unique())))}')
print('-'*100)
print(f'Number of duplicate values in POS_CASH_balance: {POS_CASH_balance.shape[0] - POS_CASH_balance.duplicated().shape[0]}')
print('-'*100)
display(POS_CASH_balance.head())
```

    The shape of POS_CASH_balance.csv is: (10001358, 8)
    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_PREV in POS_CASH_balance.csv are: 936325
    Number of unique SK_ID_CURR in POS_CASH_balance.csv are: 337252
    ----------------------------------------------------------------------------------------------------
    Number of overlapping SK_ID_CURR in application_train.csv and POS_CASH_balance.csv are: 289444
    Number of overlapping SK_ID_CURR in application_test.csv and POS_CASH_balance.csv are: 47808
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in POS_CASH_balance: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>CNT_INSTALMENT</th>
      <th>CNT_INSTALMENT_FUTURE</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1803195</td>
      <td>182943</td>
      <td>-31</td>
      <td>48.0</td>
      <td>45.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1715348</td>
      <td>367990</td>
      <td>-33</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1784872</td>
      <td>397406</td>
      <td>-32</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1903291</td>
      <td>269225</td>
      <td>-35</td>
      <td>48.0</td>
      <td>42.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2341044</td>
      <td>334279</td>
      <td>-35</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


##### Observations and Conclusions

<ol><li>This table contains around 10M datapoints, where each row corresponds to the monthly snapshot of the status of the previous POS and Cash Loan that the client had with Home Credit Group. It consists of 8 columns, two of which are SK_ID_CURR and SK_ID_PREV.</li>
    <li>There are 936k unique previous loan IDs in the table, which correspond to 337k unique current applicants (SK_ID_CURR).</li>
    <li>Out of these 337k SK_ID_CURR, 289k belong to training set and 47.8k belong to test set.</li></ol>

#### NaN Columns and Percentages


```python
print('-'*100)
print("Columns with NaN values and their percentages:")
POS_CASH_nan = nan_df_create(POS_CASH_balance)
display(POS_CASH_nan[POS_CASH_nan.percent != 0])
print('-'*100)
del POS_CASH_nan
```

    ----------------------------------------------------------------------------------------------------
    Columns with NaN values and their percentages:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>CNT_INSTALMENT_FUTURE</td>
      <td>0.260835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNT_INSTALMENT</td>
      <td>0.260675</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions

<ol><li>There are only 2 columns which contain NaN values of the 8 columns from POS_CASH_balance. These columns are the Counts of Installments remaining and the term of the loan.</li>
    <li>These columns also contain very minimal proportion of NaN values, i.e only 0.26%%, so it is also not of much concern.</li></ol>

<b>Merging the TARGETS from application_train to POS_CASH_balance table.</b>


```python
print("-"*100)
print("Merging TARGET with POS_CASH_balance Table")
pos_cash_merged = application_train.iloc[:,:2].merge(POS_CASH_balance, on = 'SK_ID_CURR', how = 'left')
print("-"*100)
```

    ----------------------------------------------------------------------------------------------------
    Merging TARGET with POS_CASH_balance Table
    ----------------------------------------------------------------------------------------------------
    

#### Correlation Matrix of Features


```python
corr_mat = correlation_matrix(pos_cash_merged, ['SK_ID_CURR','SK_ID_PREV'], figsize = (7,6))
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_209_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in POS_CASH_balance table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-"*100)
```

    interval columns not set, guessing: ['TARGET', 'MONTHS_BALANCE']
    interval columns not set, guessing: ['TARGET', 'CNT_INSTALMENT']
    interval columns not set, guessing: ['TARGET', 'CNT_INSTALMENT_FUTURE']
    interval columns not set, guessing: ['TARGET', 'SK_DPD']
    interval columns not set, guessing: ['TARGET', 'SK_DPD_DEF']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CNT_INSTALMENT_FUTURE</td>
      <td>0.033194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CNT_INSTALMENT</td>
      <td>0.030947</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MONTHS_BALANCE</td>
      <td>0.027391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SK_DPD</td>
      <td>0.012773</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SK_DPD_DEF</td>
      <td>0.010539</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The above heatmap shows the correlation between the fetures</li>
    <li>From the heatmap of correlation matrix, we one set of moderately correlated features, which are: CNT_INSTALMENT and CNT_INSTALMENT_FUTURE.</li>
    <li>The correlation of features with Target is very low, this shows the absence of a linear relationship between the feature and the target variable.</li></ol>

#### Plotting Continuous Variables

Firstly we will group by the 'SK_ID_PREV' field and aggregate with mean, so that we get an averaged row for each of the previous loan that the client had.


```python
pos_cash_merged = pos_cash_merged.groupby('SK_ID_PREV').mean()
```

<b><u>Distribution of Continuous Vairable CNT_INSTALMENT_FUTURE</u></b>

This column describes the number of installments left to pay on the previous credit.


```python
plot_continuous_variables(pos_cash_merged, 'CNT_INSTALMENT_FUTURE', plots = ['box'], figsize = (8,8))
del pos_cash_merged
```


    
![png](output_216_0.png)
    


##### Observations and Conclusions

Looking at the above box-plot for CNT_INSTALMENT_FUTURE, we see that the percentile values>50% for Defaulters are usually higher than those of Non-Defaulters. Even the upper limit whisker for Defaulters is higher than that of Non-Defaulters. This suggests that the Defaulters tend to have more number of Installments remaining on their previous credits as compared to Non-Defaulters.

### credit_card_balance.csv

##### Description

This table consists of the monthly data related to any or multiple Credit Cards that the applicant had with the Home Credit Group. The table contains fields like balance, the credit limit, amount of drawings, etc. for each month of the credit card. 

<h4>Basic Stats</h4>


```python
print(f'The shape of credit_card_balance.csv is: {cc_balance.shape}')
print('-'*100)
print(f'Number of unique SK_ID_PREV in credit_card_balance.csv are: {len(cc_balance.SK_ID_PREV.unique())}')
print(f'Number of unique SK_ID_CURR in credit_card_balance.csv are: {len(cc_balance.SK_ID_CURR.unique())}')
print('-'*100)
print(f'Number of overlapping SK_ID_CURR in application_train.csv and credit_card_balance.csv are: {len(set(application_train.SK_ID_CURR.unique()).intersection(set(cc_balance.SK_ID_CURR.unique())))}')
print(f'Number of overlapping SK_ID_CURR in application_test.csv and credit_card_balance.csv are: {len(set(application_test.SK_ID_CURR.unique()).intersection(set(cc_balance.SK_ID_CURR.unique())))}')
print('-'*100)

print(f'Number of duplicate values in credit_card_balance: {cc_balance.shape[0] - cc_balance.duplicated().shape[0]}')
print('-'*100)
display(cc_balance.head(5))
```

    The shape of credit_card_balance.csv is: (3840312, 23)
    ----------------------------------------------------------------------------------------------------
    Number of unique SK_ID_PREV in credit_card_balance.csv are: 104307
    Number of unique SK_ID_CURR in credit_card_balance.csv are: 103558
    ----------------------------------------------------------------------------------------------------
    Number of overlapping SK_ID_CURR in application_train.csv and credit_card_balance.csv are: 86905
    Number of overlapping SK_ID_CURR in application_test.csv and credit_card_balance.csv are: 16653
    ----------------------------------------------------------------------------------------------------
    Number of duplicate values in credit_card_balance: 0
    ----------------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>AMT_BALANCE</th>
      <th>AMT_CREDIT_LIMIT_ACTUAL</th>
      <th>AMT_DRAWINGS_ATM_CURRENT</th>
      <th>AMT_DRAWINGS_CURRENT</th>
      <th>AMT_DRAWINGS_OTHER_CURRENT</th>
      <th>AMT_DRAWINGS_POS_CURRENT</th>
      <th>AMT_INST_MIN_REGULARITY</th>
      <th>AMT_PAYMENT_CURRENT</th>
      <th>AMT_PAYMENT_TOTAL_CURRENT</th>
      <th>AMT_RECEIVABLE_PRINCIPAL</th>
      <th>AMT_RECIVABLE</th>
      <th>AMT_TOTAL_RECEIVABLE</th>
      <th>CNT_DRAWINGS_ATM_CURRENT</th>
      <th>CNT_DRAWINGS_CURRENT</th>
      <th>CNT_DRAWINGS_OTHER_CURRENT</th>
      <th>CNT_DRAWINGS_POS_CURRENT</th>
      <th>CNT_INSTALMENT_MATURE_CUM</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2562384</td>
      <td>378907</td>
      <td>-6</td>
      <td>56.970</td>
      <td>135000</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>1700.325</td>
      <td>1800.0</td>
      <td>1800.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2582071</td>
      <td>363914</td>
      <td>-1</td>
      <td>63975.555</td>
      <td>45000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>60175.080</td>
      <td>64875.555</td>
      <td>64875.555</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1740877</td>
      <td>371185</td>
      <td>-7</td>
      <td>31815.225</td>
      <td>450000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>26926.425</td>
      <td>31460.085</td>
      <td>31460.085</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1389973</td>
      <td>337855</td>
      <td>-4</td>
      <td>236572.110</td>
      <td>225000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11795.760</td>
      <td>11925.0</td>
      <td>11925.0</td>
      <td>224949.285</td>
      <td>233048.970</td>
      <td>233048.970</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1891521</td>
      <td>126868</td>
      <td>-1</td>
      <td>453919.455</td>
      <td>450000</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>22924.890</td>
      <td>27000.0</td>
      <td>27000.0</td>
      <td>443044.395</td>
      <td>453919.455</td>
      <td>453919.455</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>101.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


##### Observations and Conclusions

<ol><li>There are around 3.84M rows in the table credit_card_balance.csv, each of which corresponds to the monthly status of the Credit Card which the applicant had with Home Credit Group. This table contains 23 features which contain the statistics about each month's Credit Card status, such as Balance amount, Amount of Drawings, Number of drawings, status, etc.</li>
    <li>There are 104.3k unique Credit Cards whose details are in this table.</li>
    <li>Out of these 104.3k there are 103.5k unique SK_ID_CURR. What this means is that most of the applicants had just 1 credit card with them, and only few of them had more than 1. These SK_ID_CURR are the ID of the applicants who have currently applied for loan.</li>
    <li>Out of the 103k unique SK_ID_CURR, 86.9k of these applicants belong to the training set, and 16.6k belong to test application set.</li>
    <li>Out of 307k applicants in application_train table, only 86.9k of those had a credit card previously with Home Credit Group.

#### NaN Columns and Percentages


```python
cc_balance_nan = nan_df_create(cc_balance)
print('-'*100)
plot_nan_percent(cc_balance_nan, 'credit_card_balance', tight_layout = False, rotation = 90, figsize = (14,5))
print('-'*100)
del cc_balance_nan
```

    ----------------------------------------------------------------------------------------------------
    Number of columns having NaN values: 9 columns
    


    
![png](output_224_1.png)
    


    ----------------------------------------------------------------------------------------------------
    

##### Observatiosn and Conclusions

<ol><li>Out of the 23 features, 9 of these features contain some NaN values.</li>
    <li>If we look at the percentages of NaN values, they are considerably lower than the rest of the tables we have seen so far.</li>
    <li>7 of these features have close to 20% NaN values. These features are mostly related to the Amounts of Drawing and Counts of Drawings. Other two of the features are related to the installments statistics.</li></ol>

<b>Merging the TARGETS from application_train to credit_card_balance table.</b>


```python
print("-"*100)
print("Merging TARGET with credit_card_balance Table")
cc_balance_merged = application_train.iloc[:,:2].merge(cc_balance, on = 'SK_ID_CURR', how = 'left')
print("-"*100)
```

    ----------------------------------------------------------------------------------------------------
    Merging TARGET with credit_card_balance Table
    ----------------------------------------------------------------------------------------------------
    

#### Correlation Matrix of Features


```python
corr_mat = correlation_matrix(cc_balance_merged, ['SK_ID_CURR','SK_ID_PREV'], figsize = (13,11))
corr_mat.plot_correlation_matrix()
```

    ----------------------------------------------------------------------------------------------------
    


    
![png](output_229_1.png)
    


    ----------------------------------------------------------------------------------------------------
    


```python
#Seeing the top columns with highest phik-correlation with the target variable in credit_card_balance table
top_corr_target_df = corr_mat.target_top_corr()
print("-" * 100)
print("Columns with highest values of Phik-correlation with Target Variable are:")
display(top_corr_target_df)
print("-" * 100)
```

    interval columns not set, guessing: ['TARGET', 'MONTHS_BALANCE']
    interval columns not set, guessing: ['TARGET', 'AMT_BALANCE']
    interval columns not set, guessing: ['TARGET', 'AMT_CREDIT_LIMIT_ACTUAL']
    interval columns not set, guessing: ['TARGET', 'AMT_DRAWINGS_ATM_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_DRAWINGS_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_DRAWINGS_OTHER_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_DRAWINGS_POS_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_INST_MIN_REGULARITY']
    interval columns not set, guessing: ['TARGET', 'AMT_PAYMENT_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_PAYMENT_TOTAL_CURRENT']
    interval columns not set, guessing: ['TARGET', 'AMT_RECEIVABLE_PRINCIPAL']
    interval columns not set, guessing: ['TARGET', 'AMT_RECIVABLE']
    interval columns not set, guessing: ['TARGET', 'AMT_TOTAL_RECEIVABLE']
    interval columns not set, guessing: ['TARGET', 'CNT_DRAWINGS_ATM_CURRENT']
    interval columns not set, guessing: ['TARGET', 'CNT_DRAWINGS_CURRENT']
    interval columns not set, guessing: ['TARGET', 'CNT_DRAWINGS_OTHER_CURRENT']
    interval columns not set, guessing: ['TARGET', 'CNT_DRAWINGS_POS_CURRENT']
    interval columns not set, guessing: ['TARGET', 'CNT_INSTALMENT_MATURE_CUM']
    interval columns not set, guessing: ['TARGET', 'SK_DPD']
    interval columns not set, guessing: ['TARGET', 'SK_DPD_DEF']
    ----------------------------------------------------------------------------------------------------
    Columns with highest values of Phik-correlation with Target Variable are:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Phik-Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>AMT_BALANCE</td>
      <td>0.059838</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AMT_RECIVABLE</td>
      <td>0.059311</td>
    </tr>
    <tr>
      <th>12</th>
      <td>AMT_TOTAL_RECEIVABLE</td>
      <td>0.059287</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AMT_RECEIVABLE_PRINCIPAL</td>
      <td>0.058895</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MONTHS_BALANCE</td>
      <td>0.050360</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AMT_INST_MIN_REGULARITY</td>
      <td>0.042174</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CNT_INSTALMENT_MATURE_CUM</td>
      <td>0.038261</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CNT_DRAWINGS_ATM_CURRENT</td>
      <td>0.030052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AMT_CREDIT_LIMIT_ACTUAL</td>
      <td>0.028752</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CNT_DRAWINGS_CURRENT</td>
      <td>0.027868</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    

##### Observations and Conclusions:

<ol>
    <li>The heatmap above shows visually the correlation among features in credit_card_balance table.</li>
    <li>From the heatmap of correlation matrix, we see a few couples of highly correlated features. These are:
        <ul><li>AMT_RECEIVABLE_PRINCIPLE, AMT_RECIVABLE, AMT_TOTAL_RECEIVABLE and AMT_BALANCE</li>
            <li>We also observe high correlation between these 3 AMT_RECEIVABLE columns</li>
            <li>AMT_PAYMENT_TOTAL_CURRENT and AMT_PAYMENT_CURRENT</li>
        </ul></li>
    <li>The sets of 2nd and 3rd correlating features are understandable because they more or less the same tale.</li>
    <li>The correlation of features with Target isn't noticeable, this shows the absence of a linear relationship between the feature and the target variable.</li></ol>

#### Plotting Continuous Variables

Firstly we will group by the 'SK_ID_PREV' field and aggregate with mean, so that we get an averaged row for each of the previous loan that the client had.


```python
cc_balance_merged = cc_balance_merged.groupby('SK_ID_PREV').mean()
```

<b><u>Distribution of Continuous Vairable AMT_BALANCE</u></b>

This column provided the average amount of balance that a person usually had on his credit card loan account for previous loan.


```python
plot_continuous_variables(cc_balance_merged, 'AMT_BALANCE', plots = ['box'], figsize = (8,8))
```


    
![png](output_236_0.png)
    


##### Observations and Conclusions

From the above plot, it can be seen that the Defaulters have a higher value of AMT_BALANCE as compared to Non-Defaulters. They show a higher values of all the quantiles and even the whiskers. This could imply that the Credit amount for Defaulters could also be relatively higher as compared to Non-Defaulters.

##### Observations and Conclusions:

We see that the Defaulters here too appeared to have a higher minimum installment each month as compared to Non-Defaulters. This usually tells about the spending and borrowing habbit of the people. The defaulters show a higher spending and borrowing habits as compared to Non-Defaulters.

<b><u>Distribution of Continuous Vairable AMT_TOTAL_RECEIVABLE</u></b>

This column describes the average of total amount receivable on the previous credit.


```python
plot_continuous_variables(cc_balance_merged, 'AMT_TOTAL_RECEIVABLE', plots = ['violin'], figsize = (8,8))
```


    
![png](output_240_0.png)
    


##### Observations and Conclusions

Looking at the box plot of AMT_TOTAL_RECEIVABLE, we see a similar behavious as seen with other amounts as well, which is that the Defaulters usually had higher Amount Receivable on their previous credit, which may imply the higher amounts of credits that they may have taken. The PDF also shows a very higher peak at lower amounts for Non-Defaulters as compared to Defaulters.

<b><u>Distribution of Continuous Vairable CNT_INSTALMENT_MATURE_CUM</u></b>

The column describes about the average number of installments paid on the previous credits.


```python
plot_continuous_variables(cc_balance_merged, 'CNT_INSTALMENT_MATURE_CUM', plots = ['box'], figsize = (8,8))
```


    
![png](output_243_0.png)
    


##### Observations and Conclusions

From the above plot, we see a very interesting behaviour. This plot shows that the Non-Defaulters usually had higher range of values for the number of installments paid as compared to Defaulters. This might show the defaulting behaviour, where in the defaulters usually would pay fewer number of installments on their previous credit.

## Conclusions From EDA

From the Exhaustive Exporatory Data Analysis that we performed, we can draw some high level conclusions of our given dataset.
<ol><li>Firstly, the whole dataset will need to be merged together with some ingenious way for the merged data to make sense.</li>
    <li>Some categories are very well discriminatory between the Defaulters and Non-Defaulters, which could be important for the purpose of classification.</li>
    <li>There are few Continuous Numerical Variables which contain Erroneous points, we would have to handle those points.</li>
    <li>We also noticed some correlated features, which would just be increasing the dimensionality of data, and not add much value. We would want to remove such features.</li>
    <li>Overall the dataset is Imbalanced, and we would need to come up with techniques to handle such imbalance.</li>
    <li>For Default Risk prediction, the Defaulters usually tend to have some behaviour which is not normal, and thus, we cannot remove outliers or far-off points, as they may suggest some important Defaulting tendency. </li>
    <li>With all these insights, we will move to Data Clearning and Feature Engineering task.</li></ol>


```python

```

# END
