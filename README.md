# COVID-19 Data Analysis

CMSC320 Final Project


Author: Kun Zhou, Qizhao Tang


## 1. Introduction
Coronavirus, also called COVID-19, is a group of related RNA viruses that cause diseases. This virus outbreaked a pandemic. This virus could cause respiratory tract infections that can range from mild to lethal. Mild illnesses in humans include some cases of having trouble breathing, persistent chest pain or pressure, and so on. According to CDC, this virus is deadly to humans. Additionally, COVID-19 is soon widely spread worldwide until WHO declared the outbreak a Public Health Emergency of International Conern on 30 January 2020. In this project, we will be collecting data from https://covidtracking.com. Besides, we will do data visualization to show how this virus spread in United States from April to November. Based on our collected data and prediction models, we will make predictions on death increase, hospitalization increase. 

![png](README_files/covid-19.png)

## 2. Install Packages


```python
pip install numpy pandas matplotlib sklearn stats graphviz pydot
```

    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (1.19.4)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (1.1.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (3.2.2)
    Requirement already satisfied: sklearn in /usr/local/lib/python3.6/dist-packages (0.0)
    Requirement already satisfied: stats in /usr/local/lib/python3.6/dist-packages (0.1.2a0)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.6/dist-packages (0.10.1)
    Requirement already satisfied: pydot in /usr/local/lib/python3.6/dist-packages (1.3.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from sklearn) (0.22.2.post1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sklearn) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sklearn) (1.0.0)


## 3. Data Tidying

Firstly, we load the data from "https://api.covidtracking.com/v1/us/daily.csv". The website that we use, https://covidtracking.com, is the authoritative website that was cited by John Hopkins and The White House. The dataset in this website updates twice in a day. Since the dataset that we download contains the missing data at the beginning of the outbreak of the COVID-19. Threfore, we will eliminate columns that have nothing to do with our analysis. Since Coronavirus data is updated twice every day, we will only be using data from the beginning of April to the end of November to do our data analysis. Due to COVID-19 has a very long incubation period, the death number in each day is relative to the positive case confirmed before. We shifts the death related data offset by **-17** days.




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_covid_data = pd.read_csv("https://api.covidtracking.com/v1/us/daily.csv")
raw_covid_data.drop(["states", "dateChecked", "lastModified", "posNeg", "total", "hash", "inIcuCurrently", "inIcuCumulative", 
                     "onVentilatorCurrently", "totalTestResultsIncrease", "onVentilatorCumulative", "recovered", "totalTestResults", 
                     "negativeIncrease", "pending", "hospitalizedCurrently", "hospitalizedCumulative"], axis=1, inplace=True)
covid_data = raw_covid_data[(raw_covid_data['date'] >= 20200401) & (raw_covid_data['date'] <= 20201130)]
covid_data.shift(periods=-17)
covid_data.reset_index()
covid_data.head(10)
```




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
      <th>date</th>
      <th>positive</th>
      <th>negative</th>
      <th>death</th>
      <th>hospitalized</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
      <th>positiveIncrease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>20201130</td>
      <td>13463395.0</td>
      <td>154112956.0</td>
      <td>259697.0</td>
      <td>559871.0</td>
      <td>1035</td>
      <td>3394</td>
      <td>148588</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20201129</td>
      <td>13314807.0</td>
      <td>152906226.0</td>
      <td>258662.0</td>
      <td>556477.0</td>
      <td>823</td>
      <td>2429</td>
      <td>136247</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20201128</td>
      <td>13178560.0</td>
      <td>152036674.0</td>
      <td>257839.0</td>
      <td>554048.0</td>
      <td>1251</td>
      <td>3404</td>
      <td>151184</td>
    </tr>
    <tr>
      <th>21</th>
      <td>20201127</td>
      <td>13027376.0</td>
      <td>150826280.0</td>
      <td>256588.0</td>
      <td>550644.0</td>
      <td>1392</td>
      <td>3499</td>
      <td>197180</td>
    </tr>
    <tr>
      <th>22</th>
      <td>20201126</td>
      <td>12830196.0</td>
      <td>149421572.0</td>
      <td>255196.0</td>
      <td>547145.0</td>
      <td>1387</td>
      <td>2247</td>
      <td>128709</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20201125</td>
      <td>12701487.0</td>
      <td>148522805.0</td>
      <td>253809.0</td>
      <td>544898.0</td>
      <td>2280</td>
      <td>4568</td>
      <td>187134</td>
    </tr>
    <tr>
      <th>24</th>
      <td>20201124</td>
      <td>12514353.0</td>
      <td>147311323.0</td>
      <td>251529.0</td>
      <td>540330.0</td>
      <td>2106</td>
      <td>4591</td>
      <td>164392</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20201123</td>
      <td>12349961.0</td>
      <td>145926485.0</td>
      <td>249423.0</td>
      <td>535739.0</td>
      <td>858</td>
      <td>2985</td>
      <td>153783</td>
    </tr>
    <tr>
      <th>26</th>
      <td>20201122</td>
      <td>12196178.0</td>
      <td>144656088.0</td>
      <td>248565.0</td>
      <td>532754.0</td>
      <td>917</td>
      <td>2291</td>
      <td>153372</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20201121</td>
      <td>12042806.0</td>
      <td>143409545.0</td>
      <td>247648.0</td>
      <td>530463.0</td>
      <td>1554</td>
      <td>3340</td>
      <td>182809</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_covid_data = pd.read_csv("https://api.covidtracking.com/v1/states/daily.csv")
state_covid_data.drop(["dateChecked", "pending", "posNeg", "total", "lastUpdateEt", "dateModified", "checkTimeEt", "hash", "grade", "score", "positiveScore", "negativeScore", 
                       "negativeRegularScore", "commercialScore", "totalTestEncountersViral", "positiveTestsPeopleAntibody", 
                       "negativeTestsPeopleAntibody", "totalTestsPeopleAntigen", "totalTestsPeopleAntibody", "negativeTestsAntibody", 
                       "positiveTestsAntibody", "totalTestsAntibody", "totalTestsPeopleViral", "deathProbable", "inIcuCumulative", "probableCases", 
                       "totalTestResultsSource", "totalTestResults","fips","totalTestsAntigen","positiveTestsAntigen","positiveTestsPeopleAntigen","dataQualityGrade","inIcuCurrently",
                       "onVentilatorCurrently","onVentilatorCumulative","recovered","positiveTestsViral", "negativeTestsViral","positiveCasesViral","totalTestsViral"], axis=1, inplace=True)
state_covid_data = state_covid_data[(state_covid_data['date'] >= 20200401) & (state_covid_data['date'] <= 20201130)]
state_covid_data.head(10)
```




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
      <th>date</th>
      <th>state</th>
      <th>positive</th>
      <th>negative</th>
      <th>hospitalizedCurrently</th>
      <th>hospitalizedCumulative</th>
      <th>death</th>
      <th>hospitalized</th>
      <th>deathConfirmed</th>
      <th>positiveIncrease</th>
      <th>negativeIncrease</th>
      <th>totalTestResultsIncrease</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1008</th>
      <td>20201130</td>
      <td>AK</td>
      <td>31323.0</td>
      <td>980073.0</td>
      <td>162.0</td>
      <td>725.0</td>
      <td>121.0</td>
      <td>725.0</td>
      <td>121.0</td>
      <td>507</td>
      <td>4709</td>
      <td>5216</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>20201130</td>
      <td>AL</td>
      <td>249524.0</td>
      <td>1376324.0</td>
      <td>1717.0</td>
      <td>25338.0</td>
      <td>3578.0</td>
      <td>25338.0</td>
      <td>3246.0</td>
      <td>2295</td>
      <td>2554</td>
      <td>4634</td>
      <td>1</td>
      <td>668</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>20201130</td>
      <td>AR</td>
      <td>157359.0</td>
      <td>1545401.0</td>
      <td>1063.0</td>
      <td>8937.0</td>
      <td>2502.0</td>
      <td>8937.0</td>
      <td>2295.0</td>
      <td>1112</td>
      <td>6663</td>
      <td>7629</td>
      <td>32</td>
      <td>94</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>20201130</td>
      <td>AS</td>
      <td>0.0</td>
      <td>1988.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1012</th>
      <td>20201130</td>
      <td>AZ</td>
      <td>326817.0</td>
      <td>1936949.0</td>
      <td>2513.0</td>
      <td>25786.0</td>
      <td>6639.0</td>
      <td>25786.0</td>
      <td>6152.0</td>
      <td>822</td>
      <td>16630</td>
      <td>22500</td>
      <td>5</td>
      <td>218</td>
    </tr>
    <tr>
      <th>1013</th>
      <td>20201130</td>
      <td>CA</td>
      <td>1212968.0</td>
      <td>22812203.0</td>
      <td>8578.0</td>
      <td>NaN</td>
      <td>19141.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14034</td>
      <td>203636</td>
      <td>217670</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>20201130</td>
      <td>CO</td>
      <td>232905.0</td>
      <td>1531344.0</td>
      <td>1940.0</td>
      <td>13488.0</td>
      <td>3037.0</td>
      <td>13488.0</td>
      <td>2561.0</td>
      <td>4133</td>
      <td>12038</td>
      <td>43725</td>
      <td>34</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>20201130</td>
      <td>CT</td>
      <td>117295.0</td>
      <td>3144159.0</td>
      <td>1098.0</td>
      <td>12257.0</td>
      <td>5020.0</td>
      <td>12257.0</td>
      <td>4025.0</td>
      <td>4714</td>
      <td>102123</td>
      <td>106837</td>
      <td>59</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>20201130</td>
      <td>DC</td>
      <td>21552.0</td>
      <td>673704.0</td>
      <td>158.0</td>
      <td>NaN</td>
      <td>680.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104</td>
      <td>4810</td>
      <td>4914</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>20201130</td>
      <td>DE</td>
      <td>35654.0</td>
      <td>388688.0</td>
      <td>243.0</td>
      <td>NaN</td>
      <td>772.0</td>
      <td>NaN</td>
      <td>679.0</td>
      <td>403</td>
      <td>1555</td>
      <td>10970</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now, we have two important dataset for our analysis, which contains the information about COVID-19 everyday positive cases increase, deaths increase, hospitalizations increase. Also, it has total positive, negative, death, and hospitalized cases.  Compared with the first datase, the second dataset has one more important column, states.

## 4. Data Analysis and Visualization


### 4.1 Hypothesis Testing
Firstly, we assume that there is positive relationship between confirmed cases increase and deaths increase. Then we will do Hypothesis Testing. We will use paired sampled t-test. 
- Null Hypothesis: there is no relationship between confirmed cases increase and deaths increase. 
- Alternative Hypothesis: there is relationship between confirmed cases increases and deaths increase. 



```python
from scipy import stats
covid_data = covid_data.iloc[::-1]
ttest,p_value = stats.ttest_rel(covid_data['positiveIncrease'], covid_data['deathIncrease'])
print(p_value)
```

    8.180684335968525e-57


From the paired-test, we get the p-value that is 6.306550299811238e-57, which is less than 0.05. Therefore, we reject the null hypothesis, which implies that there is relationship between positives cases increases and deaths increase. Therefore, this confirm our assumption that there is positive relationship between positive cases increase  and deaths increase. 

Now, in order to do prediction of death increase and hospitalizations increase. We will plot the monthly and daily average deaths increase, and monthly and daily average hospitalization increase to get the general picture. 

### 4.2 Graph of monthly hospitalization increase


```python
month_hos = dict()
month = 4
for i in range(4, 12):
  month_hos[i] = 0
for column,row in covid_data.iterrows():
  if int(str(row['date'])[4:6]) == month:
    month_hos[int(str(row['date'])[4:6])] += covid_data.at[column, 'hospitalizedIncrease']
  else:
    month += 1

hospermonth = pd.DataFrame(month_hos.items(), columns=['month', 'hospitalization'])
plt.barh(hospermonth['month'], hospermonth['hospitalization'])
plt.ylabel('Month')
plt.xlabel('Hospitalization')
plt.title('Hospitalization Over Month', fontweight = "bold")
plt.show()
```


    
![png](README_files/README_17_0.png)
    


### 4.3 Plot of showing daily hospitalization increase 


```python
month_avg_hos = dict()
month = 4
count = 0
for i in range(4, 12):
  month_avg_hos[i] = 0
for column,row in covid_data.iterrows():
  if int(str(row['date'])[4:6]) == month:
    month_avg_hos[int(str(row['date'])[4:6])] += covid_data.at[column, 'hospitalizedIncrease']
    count += 1
  else:
    month_avg_hos[month] = month_avg_hos[month] / count
    count = 0
    month += 1
month_avg_hos[11] = month_avg_hos[11] / 30
hos_avg_month = pd.DataFrame(month_avg_hos.items(), columns=['month', 'avg_hos'])
plt.plot(hos_avg_month['month'], hos_avg_month['avg_hos'])
plt.xlabel('Month')
plt.ylabel('Daily Average Hospitalized')
plt.title('Daily Average Hospitalization Over Month', fontweight = "bold")
plt.show()
```


    
![png](README_files/README_19_0.png)
    


### 4.4 Graph of Monthly Death Increase


```python
month_death = dict()
month = 4
for i in range(4, 12):
  month_death[i] = 0
for column,row in covid_data.iterrows():
  if int(str(row['date'])[4:6]) == month:
    month_death[int(str(row['date'])[4:6])] += covid_data.at[column, 'deathIncrease']
  else:
    month += 1

death_month = pd.DataFrame(month_death.items(), columns=['month', 'death'])
plt.barh(death_month['month'], death_month['death'])
plt.xlabel('Month')
plt.ylabel('Death')
plt.title('Deaths Over Month',fontweight = "bold")
plt.show()
```


    
![png](README_files/README_21_0.png)
    


### 4.5 Plot of Daily Death Increase


```python
month_avg_death = dict()
month = 4
count = 0
for i in range(4, 12):
  month_avg_death[i] = 0
for column,row in covid_data.iterrows():
  if int(str(row['date'])[4:6]) == month:
    month_avg_death[int(str(row['date'])[4:6])] += covid_data.at[column, 'deathIncrease']
    count += 1
  else:
    month_avg_death[month] = month_avg_death[month] / count
    count = 0
    month += 1
month_avg_death[11] = month_avg_death[11] / 30
death_avg_month = pd.DataFrame(month_avg_death.items(), columns=['month', 'avg_death'])
plt.plot(death_avg_month['month'], death_avg_month['avg_death'])
plt.xlabel('Month')
plt.ylabel('Daily Average Death')
plt.title('Daily Average Death Over Month', fontweight = "bold")
plt.show()
```


    
![png](README_files/README_23_0.png)
    


### 4.6 Plot of Positive Case Confirmed for Each State



```python
covid_group_by_state = state_covid_data.groupby('state')
plt.figure(figsize=(15, 10))
for state, data in covid_group_by_state:
  plt.plot(data['date'], data['positive'], label=state)
plt.legend(ncol=3)
plt.ticklabel_format(axis='y', style='plain')
month_starts = [20200401,20200501,20200601,20200701,20200801,20200901,20201001,20201101]
month_names = ['Apr','May','Jun', 'Jul','Aug','Sep','Oct','Nov'] 
plt.title('Death Increase Per Day for Each State in US')
plt.xlabel('Time')
plt.ylabel('Cumulative Death')
plt.xticks(month_starts,month_names)
plt.show()

```


    
![png](README_files/README_25_0.png)
    


### 4.7 Observation

We drew the graphs for total deaths and hospitalizations in a month and daily average deaths and hospitalizations from the beginning of the April to the end of November. 
1. The number of death decreases drastically from April to June. Additionally, daily death increase remains relatively low between June and October, which is less than 1000. 
2. The number of hospitalized people also reduces drastically from April to June. Additionally, daily hospitalization increases from June to July and decreases from August to September. However, the number of daily hospitalized individuals is relatively lower from June to October, which is less than 2000. 
3. Based on the graphs, it can be seen that the number of deaths and hospitalizations are lower in June, July, August, September than other months, which implied that temperature may be one factor that might affect the number of deaths and the number of hospitalized people. 

## 5. Data Prediction
Prediction the future COVID-19 growth rate and death rate is an extremely important.

### 5.1 Importing Packages
We will be using the following packages to train and illustrate our ML model.
1. Using `trian_test_split` to split our data to traning and testing.
2. Using `RandomForestRegressor` to train RF model.
3. Using `SVR` to train SVR model.
3. Using `export_graphviz` to format a dot format for displaying the tree.
4. Using `SVG` from `IPython Dislay` to display the SVG file.
5. Using `pydot` to generate tree graph.


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import export_graphviz
from IPython.display import SVG
import pydot
```

### 5.2 Preparing Data for ML
Since Random Forest is a supervised learning model, we have to provide a baseline for the model to calculate the baseline_error and compensate the loss function.

To provide the baseline, we used the research from [JHU Lab](https://coronavirus.jhu.edu/data/mortality) and select the mortality rate of **1.8%** for US.

Due to COVID-19 has a very long incubation period, the death number in each day is relative to the positive case confirmed before. We shifts the death related data offset by **-17** days. 

We will also drop few columns that is irrelevent with this ML model.


```python
p5_data = raw_covid_data.copy()
p5_data['deathIncreaseAvg'] = p5_data['positiveIncrease'] * 0.018 # Mortality Data from [3]

p5_data.deathIncreaseAvg.shift(periods=-17, fill_value=0)
p5_data.deathIncrease.shift(periods=-17, fill_value=0)
p5_data = p5_data[(p5_data['date'] >= 20200401) & (p5_data['date'] <= 20201130)]
p5_data.reset_index()

p5_labels = np.array(p5_data['deathIncrease'])
p5_data = p5_data.drop(['deathIncrease'], axis=1)
p5_f_list = list(p5_data.columns)
p5_f = np.array(p5_data)
```

### 5.3 Creating Training & Testing Set
We will seperate our data to a ratio 75% - 25%.

The random state will be set to 100 to ensure the Random Forest ML get enough random selections on our model.


```python
p5_rf_train_features, p5_rf_test_features, p5_rf_train_labels, p5_rf_test_labels = \
  train_test_split(p5_f, p5_labels, test_size = 0.25, random_state = 100)

p5_svr_train_features, p5_svr_test_features, p5_svr_train_labels, p5_svr_test_labels = \
  train_test_split(p5_f, p5_labels, test_size = 0.25)

p5_rf_baseline_preds = p5_rf_test_features[:, p5_f_list.index('deathIncreaseAvg')]
p5_rf_baseline_errors = abs(p5_rf_baseline_preds - p5_rf_test_labels)

p5_svr_baseline_preds = p5_svr_test_features[:, p5_f_list.index('deathIncreaseAvg')]
p5_svr_baseline_errors = abs(p5_svr_baseline_preds - p5_svr_test_labels)

print('RF Training Features Shape:', p5_rf_train_features.shape)
print('RF Training Labels Shape:', p5_rf_train_labels.shape)
print('RF Testing Features Shape:', p5_rf_test_features.shape)
print('RF Testing Labels Shape:', p5_rf_test_labels.shape)
print('RF Average baseline error: ', round(np.mean(p5_rf_baseline_errors), 2))

print('SVR Training Features Shape:', p5_svr_train_features.shape)
print('SVR Training Labels Shape:', p5_svr_train_labels.shape)
print('SVR Testing Features Shape:', p5_svr_test_features.shape)
print('SVR Testing Labels Shape:', p5_svr_test_labels.shape)
print('SVR Average baseline error: ', round(np.mean(p5_svr_baseline_errors), 2))
```

    RF Training Features Shape: (183, 8)
    RF Training Labels Shape: (183,)
    RF Testing Features Shape: (61, 8)
    RF Testing Labels Shape: (61,)
    RF Average baseline error:  639.81
    SVR Training Features Shape: (183, 8)
    SVR Training Labels Shape: (183,)
    SVR Testing Features Shape: (61, 8)
    SVR Testing Labels Shape: (61,)
    SVR Average baseline error:  626.24


### 5.4 Training Model







#### 5.4.1 Random Forest ML
In this prediction, we will be using Random Forest to conduct a supervised machine learning algorithem. Random Forests are an ensemble learning method for classificatipn, regression and other tasks that operate by constructing a multitude of decision tress at training time and outputting the class that is the mode of the classes or average prediction of the individual trees.

To find the suitable features for predicting deathIncrease, we follow the CDC's research on what factors are causing the death.  

We will run a model with 1000 estimators and 100 random state.

##### 5.4.1.1 Training Model & Accuracy


```python
p5_rf = RandomForestRegressor(n_estimators = 2000, random_state = 100)
p5_rf.fit(p5_rf_train_features, p5_rf_train_labels)

p5_rf_pred = p5_rf.predict(p5_rf_test_features)
p5_rf_errors = abs(p5_rf_pred - p5_rf_test_labels)

print('Mean Absolute Error:', round(np.mean(p5_rf_errors), 2))

p5_rf_mape = 100 * (p5_rf_errors / p5_rf_test_labels)

p5_rf_accuracy = 100 - np.mean(p5_rf_mape)
print('Accuracy:', round(p5_rf_accuracy, 3), '%.')
```

    Mean Absolute Error: 191.38
    Accuracy: 77.96 %.


##### 5.4.1.2 Random Forest Tree Graph
To save space, we will only display 1 tree from RF.


```python
p5_rf_graph = pydot.graph_from_dot_data(
    export_graphviz(p5_rf.estimators_[5], feature_names = p5_f_list, 
                    rounded = True, precision = 1))
# Write graph to a png file
SVG(p5_rf_graph[0].create_svg())
```




    
![svg](README_files/README_40_0.svg)
    



#### 5.4.2 SVR Learning

**Support Vector Regression(SVR)** gives us the flexibility to define error is acceptable in our model and will find an appropriate line to fit the data. 

##### 5.4.2.1 Training & Accuracy



```python
p5_svr = SVR()
p5_svr.fit(p5_svr_train_features, p5_svr_train_labels)


p5_svr_pred = p5_svr.predict(p5_svr_test_features)
p5_svr_errors = abs(p5_svr_pred - p5_svr_test_labels)

print('Mean Absolute Error:', round(np.mean(p5_svr_errors), 2))

p5_svr_mape = 100 * (p5_svr_errors / p5_svr_test_labels)

p5_svr_accuracy = 100 - np.mean(p5_svr_mape)
print('Accuracy:', round(p5_svr_accuracy, 3), '%.')
```

    Mean Absolute Error: 430.42
    Accuracy: 50.906 %.


### 5.5 Results
Obviously, Random Forest had a much better accuracy than Support Vector Regression. Therefore, we will stick with RF for the following steps.


```python
plt.bar(['SVR', 'RF'], [p5_svr_accuracy, p5_rf_accuracy])
plt.show
```




    <function matplotlib.pyplot.show>




    
![png](README_files/README_46_1.png)
    


### 5.6 Variable Importance Factor for RF Model
Displaying the variable importance could show us which factor is the most important one in decising the deathIncrease in our model.


```python
p5_importances = list(p5_rf.feature_importances_)
p5_feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(p5_f_list, p5_importances)]
p5_feature_importances = sorted(p5_feature_importances, key = lambda i: i[1], reverse = True)
plt.bar(list(range(len(p5_importances))), p5_importances)
plt.xticks(list(range(len(p5_importances))), p5_f_list, rotation="vertical")
plt.xlabel('Variables')
plt.ylabel('Importance Factor')
plt.title('Variable Importance for COVID-19 DeathIncrease')
plt.show()
```


    
![png](README_files/README_48_0.png)
    


## Reference and Resources

1. https://covidtracking.com/data
2. https://www.mayoclinic.org/diseases-conditions/coronavirus/symptoms-causes/syc-20479963
3. https://coronavirus.jhu.edu/data/mortality
4. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
5. https://en.wikipedia.org/wiki/Coronavirus
6. https://en.wikipedia.org/wiki/Random_forest#:~:text=Random%20forests%20or%20random%20decision,average%20prediction%20(regression)%20of%20the

