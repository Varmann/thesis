#%%
import matplotlib.pyplot as plt
import pandas as pd

#######################################################################################################
#%% Lenet with Maxpool Kernel Size
filename = r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_MaxPool_Kernel_Size.csv"
df = pd.read_csv(filename, sep=";")
#plt.scatter(loss,accuracy)
plt.plot(df["Kernel_Size"],df["Loss"],'r--')
plt.xlabel('Kernel')
plt.ylabel('Loss')
plt.title('Loss vs Kernel size')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
#plt.scatter(loss,accuracy)
plt.plot(df["Kernel_Size"],df["Accuracy"],'r--')
plt.xlabel('Kernel')
plt.ylabel('Accuracy in Prozent')
plt.title('Accuracy vs Kernel size')
plt.grid(True)
plt.show()


#######################################################################################################

#%% Lenet with Maxpool Kernel Number
filename = r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_MaxPool_Kernel_Number.csv"
df = pd.read_csv(filename, sep=";")
#plt.scatter(loss,accuracy)
plt.plot(df["Kernel_Number"],df["Loss"],'r--')
plt.xlabel('Kernel')
plt.ylabel('Loss')
plt.title('Loss vs Kernel size')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
#plt.scatter(loss,accuracy)
plt.plot(df["Kernel_Number"],df["Accuracy"],'r--')
plt.xlabel('Kernel')
plt.ylabel('Accuracy in Prozent')
plt.title('Accuracy vs Kernel size')
plt.grid(True)
plt.show()


#######################################################################################################
# %%

