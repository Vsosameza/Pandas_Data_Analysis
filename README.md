

```python
import pandas as pd
import numpy as np
df = pd.read_json('purchase_data.json')
df.head(10)
#len(df.columns) # 6total columns 
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>Male</td>
      <td>10</td>
      <td>Sleepwalker</td>
      <td>1.73</td>
      <td>Tanimnya91</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20</td>
      <td>Male</td>
      <td>153</td>
      <td>Mercenary Sabre</td>
      <td>4.57</td>
      <td>Undjaskla97</td>
    </tr>
    <tr>
      <th>7</th>
      <td>29</td>
      <td>Female</td>
      <td>169</td>
      <td>Interrogator, Blood Blade of the Queen</td>
      <td>3.32</td>
      <td>Iathenudil29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>25</td>
      <td>Male</td>
      <td>118</td>
      <td>Ghost Reaver, Longsword of Magic</td>
      <td>2.77</td>
      <td>Sondenasta63</td>
    </tr>
    <tr>
      <th>9</th>
      <td>31</td>
      <td>Male</td>
      <td>99</td>
      <td>Expiration, Warscythe Of Lost Worlds</td>
      <td>4.53</td>
      <td>Hilaerin92</td>
    </tr>
  </tbody>
</table>
</div>




```python
#player count: total number of non unique screen names  ---number of unique screen names 780
df.SN.count()
```




    780




```python
#number of unique items 179
df["Item ID"].nunique()
```




    183




```python
#average purchase price is $2.93
df.Price.mean()
```




    2.931192307692303




```python
#total number of purchases is equal to total number of screennames  
#as eachscreenname represents one purchase. 
df["SN"].count()
```




    780




```python
#total revenue is $2286.33 
df.Price.sum()
```




    2286.33




```python
#total number of males is 633
#total number of females is 136
#total number of nondisclosed is 11
grouped =df.groupby("Gender").count().reset_index()
grouped.head()
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
      <th>Gender</th>
      <th>Age</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>136</td>
      <td>136</td>
      <td>136</td>
      <td>136</td>
      <td>136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>633</td>
      <td>633</td>
      <td>633</td>
      <td>633</td>
      <td>633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Other / Non-Disclosed</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
#percentage demographics
#males 81.15%
#females 17.43% 
#nondisclosed 1.41%
df["Gender"].value_counts(normalize=True) * 100
```




    Male                     81.153846
    Female                   17.435897
    Other / Non-Disclosed     1.410256
    Name: Gender, dtype: float64




```python
#Purchasing analysis by gender-----MALES
male_purch = df[df["Gender"] == "Male"]["Price"].count()
male_avgpurch = df[df["Gender"] == "Male"]['Price'].mean()
male_purchtotal = df[df["Gender"] == "Male"]['Price'].sum()
totalmales = df[df["Gender"] == "Male"]["SN"].nunique()
male_normalized = male_purchtotal/totalmales
male_purch, male_avgpurch, male_purchtotal, male_normalized

#male purchase count is 633, with an average purchase price of $2.95
#total male purchase value is $1867.68 with a normalized total of $4.02
```




    (633, 2.9505213270142154, 1867.68, 4.016516129032258)




```python
#Purchasing analysis-----FEMALES
female_purch = df[df["Gender"] == "Female"]["Price"].count()
female_avgpurch = df[df["Gender"] == "Female"]['Price'].mean()
female_purchtotal = df[df["Gender"] == "Female"]['Price'].sum()
totalfemales = df[df["Gender"] == "Female"]["SN"].nunique()
female_normalized = female_purchtotal/totalfemales
female_purch, female_avgpurch,female_purchtotal,female_normalized

#female purchase count is 136 with an everage purchase price of $2.81 
#total male purchase value is $382.91 with a normalized total of $3.83
```




    (136, 2.815514705882352, 382.90999999999997, 3.8290999999999995)




```python
#other/non disclosed purchase analysis
totalpurchases = df['Price'].count()
NDpurch = totalpurchases - male_purch - female_purch
ND_avgpurch = df[df["Gender"] == "Other / Non-Disclosed"]['Price'].mean()
ND_purchtotal = df[df["Gender"] == "Other / Non-Disclosed"]['Price'].sum()
totalpeople = df["SN"].nunique()
othercount = totalpeople - totalmales - totalfemales
other_normalized = ND_purchtotal/othercount

NDpurch, ND_avgpurch,ND_purchtotal,other_normalized
#Non disclosed purchase count is 11 with an everage purchase price of $3.25
#total nondisclosed purchase value is $35.74 with a normalized total of $4.47
```




    (11, 3.2490909090909086, 35.739999999999995, 4.467499999999999)




```python
#AGE DEMOGRAPHICS--------BROKEN DOWN BY 4 YEAR BINS 
bins = [0, 9, 14, 19, 24,29,34,39,100]
group_names = ['<10', '10-14', '15-19', '20-24','25-29','30-34','35-39','40+']
pd.cut(df["Age"], bins, labels=group_names)
df["Age Group"] = pd.cut(df["Age"], bins, labels=group_names)

pymoli_groupby_sn_agegrp = df.groupby(["SN","Age Group"])

                                                              
pymoli_unique_payers_df = pd.DataFrame(pymoli_groupby_sn_agegrp.size())
#print(pymoli_unique_payers_df)
pymoli_agegrp_df = pd.DataFrame(pymoli_unique_payers_df.groupby(["Age Group"]).count())
pymoli_agegrp_df.columns= ["Total Count"]
pymoli_agegrp_df["Percentage of Players"] =round(100 *pymoli_agegrp_df["Total Count"]/pymoli_agegrp_df["Total Count"].sum(),2)

pymoli_agegrp_df

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
      <th>Total Count</th>
      <th>Percentage of Players</th>
    </tr>
    <tr>
      <th>Age Group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;10</th>
      <td>19</td>
      <td>3.32</td>
    </tr>
    <tr>
      <th>10-14</th>
      <td>23</td>
      <td>4.01</td>
    </tr>
    <tr>
      <th>15-19</th>
      <td>100</td>
      <td>17.45</td>
    </tr>
    <tr>
      <th>20-24</th>
      <td>259</td>
      <td>45.20</td>
    </tr>
    <tr>
      <th>25-29</th>
      <td>87</td>
      <td>15.18</td>
    </tr>
    <tr>
      <th>30-34</th>
      <td>47</td>
      <td>8.20</td>
    </tr>
    <tr>
      <th>35-39</th>
      <td>27</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>40+</th>
      <td>11</td>
      <td>1.92</td>
    </tr>
  </tbody>
</table>
</div>




```python
#PURCHASING ANALYSIS 
pymoli_agegrp_not_sn_df = pd.DataFrame(df.groupby(["Age Group"]).count())
pymoli_agegrp_sum_not_sn_df = pd.DataFrame(df.groupby(["Age Group"]).sum())
pymoli_agegrp_df["Purchase Count"] =  pymoli_agegrp_not_sn_df["Item ID"]
pymoli_agegrp_df["Average Purchase Price"] = round(pymoli_agegrp_sum_not_sn_df["Price"]/pymoli_agegrp_df["Purchase Count"],2).map("${:,.2f}".format)
pymoli_agegrp_df["Total Purchase Value"] = pymoli_agegrp_sum_not_sn_df["Price"]
pymoli_agegrp_df["Normalized Totals"] =  round(pymoli_agegrp_df["Total Purchase Value"]/pymoli_agegrp_df["Total Count"],2).map("${:,.2f}".format)
pymoli_agegrp_df["Total Purchase Value"] = pymoli_agegrp_df["Total Purchase Value"].map("${:,.2f}".format)
pymoli_agegrp_df = pymoli_agegrp_df.drop(labels = ["Total Count","Percentage of Players"],axis = 1)
pymoli_agegrp_df
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
      <th>Purchase Count</th>
      <th>Average Purchase Price</th>
      <th>Total Purchase Value</th>
      <th>Normalized Totals</th>
    </tr>
    <tr>
      <th>Age Group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;10</th>
      <td>28</td>
      <td>$2.98</td>
      <td>$83.46</td>
      <td>$4.39</td>
    </tr>
    <tr>
      <th>10-14</th>
      <td>35</td>
      <td>$2.77</td>
      <td>$96.95</td>
      <td>$4.22</td>
    </tr>
    <tr>
      <th>15-19</th>
      <td>133</td>
      <td>$2.91</td>
      <td>$386.42</td>
      <td>$3.86</td>
    </tr>
    <tr>
      <th>20-24</th>
      <td>336</td>
      <td>$2.91</td>
      <td>$978.77</td>
      <td>$3.78</td>
    </tr>
    <tr>
      <th>25-29</th>
      <td>125</td>
      <td>$2.96</td>
      <td>$370.33</td>
      <td>$4.26</td>
    </tr>
    <tr>
      <th>30-34</th>
      <td>64</td>
      <td>$3.08</td>
      <td>$197.25</td>
      <td>$4.20</td>
    </tr>
    <tr>
      <th>35-39</th>
      <td>42</td>
      <td>$2.84</td>
      <td>$119.40</td>
      <td>$4.42</td>
    </tr>
    <tr>
      <th>40+</th>
      <td>17</td>
      <td>$3.16</td>
      <td>$53.75</td>
      <td>$4.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
#TOP SPENDERS IN THE GAME
pymoli_counts_df = pd.DataFrame(df.groupby("SN").count())
pymoli_sum_df = pd.DataFrame(df.groupby("SN").sum())
pymoli_spenders_df = pymoli_sum_df
pymoli_spenders_df = pymoli_spenders_df.sort_values(by=["Price"], ascending=False)
pymoli_spenders_df["Purchase Count"] = pymoli_counts_df["Item ID"]
pymoli_spenders_df["Average Purchase Price"] = round(pymoli_spenders_df["Price"]/pymoli_spenders_df["Purchase Count"],2).map("${:,.2f}".format)
pymoli_spenders_df["Total Purchase Value"] = pymoli_spenders_df["Price"].map("${:,.2f}".format)
pymoli_spenders_df = pymoli_spenders_df.drop(labels = ["Age","Item ID","Price"],axis = 1)
pymoli_spenders_df.head(10)
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
      <th>Purchase Count</th>
      <th>Average Purchase Price</th>
      <th>Total Purchase Value</th>
    </tr>
    <tr>
      <th>SN</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Undirrala66</th>
      <td>5</td>
      <td>$3.41</td>
      <td>$17.06</td>
    </tr>
    <tr>
      <th>Saedue76</th>
      <td>4</td>
      <td>$3.39</td>
      <td>$13.56</td>
    </tr>
    <tr>
      <th>Mindimnya67</th>
      <td>4</td>
      <td>$3.18</td>
      <td>$12.74</td>
    </tr>
    <tr>
      <th>Haellysu29</th>
      <td>3</td>
      <td>$4.24</td>
      <td>$12.73</td>
    </tr>
    <tr>
      <th>Eoda93</th>
      <td>3</td>
      <td>$3.86</td>
      <td>$11.58</td>
    </tr>
    <tr>
      <th>Isursti83</th>
      <td>3</td>
      <td>$3.68</td>
      <td>$11.05</td>
    </tr>
    <tr>
      <th>Isurria36</th>
      <td>3</td>
      <td>$3.67</td>
      <td>$11.01</td>
    </tr>
    <tr>
      <th>Eusri70</th>
      <td>3</td>
      <td>$3.52</td>
      <td>$10.55</td>
    </tr>
    <tr>
      <th>Aerithllora36</th>
      <td>3</td>
      <td>$3.48</td>
      <td>$10.45</td>
    </tr>
    <tr>
      <th>Yasriphos60</th>
      <td>3</td>
      <td>$3.47</td>
      <td>$10.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
#MOST POPULAR ITEMS IN THE GAME 
pymoli_items_count_df = pd.DataFrame(df.groupby(["Item ID","Item Name"]).count())
pymoli_items_sum_df = pd.DataFrame(df.groupby(["Item ID","Item Name"]).sum())
pymoli_items_count_df["Purchase Count"] = pymoli_items_count_df["Age"]
pymoli_items_count_df["Item Price"] = round(pymoli_items_sum_df["Price"]/pymoli_items_count_df["Age"],2)
pymoli_items_count_df["Total Purchase Value"] = pymoli_items_sum_df["Price"]
pymoli_items_count_df = pymoli_items_count_df.sort_values(by = ["Purchase Count","Total Purchase Value"],ascending = False)
pymoli_popular_items_df = pymoli_items_count_df.drop(labels = ["Age","Age Group","Gender","Price","SN"],axis = 1)
pymoli_popular_items_df["Item Price"] = pymoli_popular_items_df["Item Price"].map("${:,.2f}".format)
pymoli_popular_items_df["Total Purchase Value"] = pymoli_popular_items_df["Total Purchase Value"].map("${:,.2f}".format)
pymoli_popular_items_df.head()
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
      <th></th>
      <th>Age_Group</th>
      <th>Purchase Count</th>
      <th>Item Price</th>
      <th>Total Purchase Value</th>
    </tr>
    <tr>
      <th>Item ID</th>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <th>Betrayal, Whisper of Grieving Widows</th>
      <td>11</td>
      <td>11</td>
      <td>$2.35</td>
      <td>$25.85</td>
    </tr>
    <tr>
      <th>84</th>
      <th>Arcane Gem</th>
      <td>11</td>
      <td>11</td>
      <td>$2.23</td>
      <td>$24.53</td>
    </tr>
    <tr>
      <th>34</th>
      <th>Retribution Axe</th>
      <td>9</td>
      <td>9</td>
      <td>$4.14</td>
      <td>$37.26</td>
    </tr>
    <tr>
      <th>31</th>
      <th>Trickster</th>
      <td>9</td>
      <td>9</td>
      <td>$2.07</td>
      <td>$18.63</td>
    </tr>
    <tr>
      <th>13</th>
      <th>Serenity</th>
      <td>9</td>
      <td>9</td>
      <td>$1.49</td>
      <td>$13.41</td>
    </tr>
  </tbody>
</table>
</div>




```python
#MOST PROFITABLE ITEMS IN THE GAME 
pymoli_items_count_df = pymoli_items_count_df.sort_values(by = ["Total Purchase Value"],ascending = False)
pymoli_profitable_items_df = pymoli_items_count_df.drop(labels = ["Age","Age Group","Gender","Price","SN"],axis = 1)
pymoli_profitable_items_df["Item Price"] = pymoli_profitable_items_df["Item Price"].map("${:,.2f}".format)
pymoli_profitable_items_df["Total Purchase Value"] = pymoli_profitable_items_df["Total Purchase Value"].map("${:,.2f}".format)
pymoli_profitable_items_df.head()
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
      <th></th>
      <th>Age_Group</th>
      <th>Purchase Count</th>
      <th>Item Price</th>
      <th>Total Purchase Value</th>
    </tr>
    <tr>
      <th>Item ID</th>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <th>Retribution Axe</th>
      <td>9</td>
      <td>9</td>
      <td>$4.14</td>
      <td>$37.26</td>
    </tr>
    <tr>
      <th>115</th>
      <th>Spectral Diamond Doomblade</th>
      <td>7</td>
      <td>7</td>
      <td>$4.25</td>
      <td>$29.75</td>
    </tr>
    <tr>
      <th>32</th>
      <th>Orenmir</th>
      <td>6</td>
      <td>6</td>
      <td>$4.95</td>
      <td>$29.70</td>
    </tr>
    <tr>
      <th>103</th>
      <th>Singed Scalpel</th>
      <td>6</td>
      <td>6</td>
      <td>$4.87</td>
      <td>$29.22</td>
    </tr>
    <tr>
      <th>107</th>
      <th>Splitter, Foe Of Subtlety</th>
      <td>8</td>
      <td>8</td>
      <td>$3.61</td>
      <td>$28.88</td>
    </tr>
  </tbody>
</table>
</div>


