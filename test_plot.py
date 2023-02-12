
#%%
kernel = [3,4,5,6,7]

loss3 =  0.0539 
loss4 = 0.0595
loss5 = 0.0477
loss6 = 0.0518
loss7 = 0.0589


loss =[loss3,loss4,loss5,loss5,loss7]

accuracy3 = 0.9826
accuracy4 = 0.9807
accuracy5 = 0.9863
accuracy6 = 0.9830
accuracy7 = 0.9823

accuracy = [accuracy3,accuracy4, accuracy5,accuracy6,accuracy7]




# %%
import matplotlib.pyplot as plt
#plt.scatter(loss,accuracy)
plt.plot(kernel,loss,'r--')
plt.xlabel('Kernel')
plt.ylabel('Loss')
plt.title('Loss vs Kernel size')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
#plt.scatter(loss,accuracy)
plt.plot(kernel,accuracy,'r--')
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Kernel size')
plt.grid(True)
plt.show()

# %%


# Test set: Avg. loss: 0.0518, Accuracy: 9830/10000 (98%)


#%%
filter_size = 7
50*(  (((28-filter_size+1)//2)-filter_size+1)//2)**2

# %%
filter_size = 7
f = open(r'C:/Users/Vardan/Desktop/Train_Data/Data.txt', 'a',newline = '\n')
data = 50*(  (((28-filter_size+1)//2)-filter_size+1)//2)**2
f.write('{:.2f}\n'.format(data))

f.close()
#%%
f = open(r'C:/Users/Vardan/Desktop/Train_Data/Data.txt', 'r')
print("Data",f.read())
f.close()
# %%
import pandas as pd
my_df = pd.DataFrame({'AAA': [1, 2, 3, 4, 5], 'BBB': [6, 7, 8, 9, 10], 'CCC': [11, 12, 13, 14, 15]})
my_df.to_csv(r'C:/Users/Vardan/Desktop/Train_Data/my_array.csv')
 
# %%
df = pd.read_csv(r'C:/Users/Vardan/Desktop/Train_Data/my_array.csv')
print (df)

# %%
