#%%
import matplotlib.pyplot as plt
year = [1950, 1970, 1980, 2010]
pop = [2.519, 3.692, 5.263 , 6.972]
# %%
#plt.plot(year,pop)
plt.scatter(year,pop)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projektions')
plt.yticks([0,2,4,6,8,10],['0B','2B','4B','6B','8B','10B'])
plt.show()

# %%
plt.scatter(year,pop)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()
# %%
import numpy as np
random_list =  np.random.rand(10)

plt.hist(random_list, bins=20)
# %%
# Histogram of life_exp, 15 bins
plt.hist(random_list, bins = 5)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins
plt.hist(random_list, bins = 15)

# Show and clear plot again
plt.show()
plt.clf()
# %%
# Specify c and alpha inside plt.scatter()
col = ['red','blue','yellow','green']
plt.scatter(x = year, y = pop, s = np.array(pop) * 2,c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([0,2,4,6,8,10],['0B','2B','4B','6B','8B','10B'])
# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# %%
