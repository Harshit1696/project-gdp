import pandas as pd ##data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #built on matplotlib provides interactive visualization
import matplotlib.pyplot as plt


data = pd.read_csv('countries of the world.csv', decimal=',')
 #reads thee file into the python program
#Character to recognize as decimal point
print('number of missing data in each coloumn of the datset:')
print(data.isnull().sum()) #gives coloumn wise sum of missing values



for col in data.columns.values:    # for each value in each coloumn in the dataset
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate': 
        guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().mean())
        # we grouped by region as countries in the same region are similar 
        # then we apply mode because climate is a categorical value
    else:
        guess_values = data.groupby('Region')[col].mean()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())] = guess_values[region]
        # will fill all the null values of the coloumns of the dataset

print(data.isnull().sum()) #check if we filled all missing values

plt.figure(figsize=(15,12))    # setting the size of the heatmap
ax = plt.axes()
sns.heatmap(data=data.iloc[2:18].corr(),annot=True,fmt='.1f',cmap='coolwarm')
ax.set_title('CORRELATION MATRIX')
 #puts data of all 20 coloumns in the heatmap
 #annot assigns correlation value to each cell of the heatmap 
 #fmt sets floating point precision i.e 2 in this case
plt.show()   #display the heatmap

fig = plt.subplots(figsize=(10,6))   #sets the size of the barplot
top_gdp_countries = data.sort_values('GDP ($ per capita)',ascending=False).head(20) 
# sorts the top 20 countries in descending order according to their gdp
mean = pd.DataFrame({'Country':['World mean'], 'GDP ($ per capita)':[data['GDP ($ per capita)'].mean()]})
gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean])

sns.barplot(x='Country', y='GDP ($ per capita)', data=gdps, palette='Set2')
#data is feeded through top_gdp_countries into barplot

plt.xticks(rotation=90)  #rotates the text of x axis by 90degrees
plt.show()  #displays the barplot


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
#number of rows and coloumns of the subplot grid
plt.subplots_adjust(hspace=0.4)  #amount of space between subplots

corr_to_gdp = pd.Series() #puts the values into form of 1D array
for col in data.columns.values[2:]: #taking coloumns from 2 to last
    if ((col!='GDP ($ per capita)')&(col!='Climate')):
        corr_to_gdp[col] = data['GDP ($ per capita)'].corr(data[col])
        #correlating between any given coloumn and gdp per capita
        #stores a float correlaation value in corr_to_gdp coloumn
        
corr_to_graph = corr_to_gdp.sort_values(ascending=False)
#sorts the correlated values in descending order
corr_to_gdp = corr_to_gdp.loc[corr_to_graph.index]
#selects the index labels of those who have the maximum correlation with gdp/capita

for i in range(2):
    for j in range(2):  #for each subplots 
        sns.regplot(x=corr_to_gdp.index.values[i*2+j], y='GDP ($ per capita)', data=data,
                   ax=axes[i,j], fit_reg=False, marker='.')
        #in x we assign correlation values at a given index (in corr_to_gdp)
        #in y we give value gdp_per_capita
        #regplot used to visualize linear correlation
        #[i*2+j] assigns the index value in the corr_to_gdp
        # fit_reg=False to disable fitting linear model and plotting a line
        title = 'correlation='+str(corr_to_gdp[i*2+j])
        #set the title of the subplots
        axes[i,j].set_title(title)
        #sets the title of the subplots

plt.show()
#displays the subplots

#Analysis of the dataset
print("\nCOUNTRY WITH MAXIMUM GDP/CAPITA IS:",)
print(data.loc[data['GDP ($ per capita)'].idxmax()].head(1))
print("GDP:",data['GDP ($ per capita)'].max())
print("\nCOUNTRY WITH MINIMUM GDP/CAPITA IS:")
print(data.loc[data['GDP ($ per capita)'].idxmin()].head(1))
print("gdp:",data['GDP ($ per capita)'].min())
print("\nTHE WORLD MEAN GDP/CAPITA IS:",data['GDP ($ per capita)'].mean())
print("\nTHE GDP/CAPITA FOR INDIA IS:",data.iloc[94][8])
print("\nFOLLOWING SHOWS THE GDP OF THE WORLD REGION WISE:")
world_gdp_shift = data.groupby('Region')['GDP ($ per capita)'].apply(lambda x: x.mode().max())
print(world_gdp_shift)

