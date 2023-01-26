#%%
# Create a variable savings
savings = 100

# Create a variable growth_multiplier
growth_multiplier = 1.1

# Calculate result
result = savings * growth_multiplier

# Print out result
print("Result : " ,result)

# %%
print(4+5)
# %%
# How much is your $100 worth after 7 years?
print(100* (1.1**7))

# %% 
desc  = "desc"
doubledesc = desc*2
print(doubledesc)
# %%
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]

# Print areas
print(areas)
print(type(areas))
#%%
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))

# %%
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])
print(areas[len(areas) - 1])

# Print out the area of the living room
print(areas[5])

# %%
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
#upstairs = areas[(-4):]
upstairs = areas[-4:]
print(downstairs)
print(upstairs)
# %%
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
#areas_2 = areas_1 + ["garage", 15.45]
areas_2 = list(areas_1)
areas_2.append("garage")
areas_2.append(15.45)


# %%
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50, "pool", 24.5]
# Deletes      "bathroom", 10.50   
del(areas[-4:-2])
print(areas)
# %%
student_tuples = [

    ('john', 'A', 15),

    ('jane', 'B', 12),

    ('dave', 'B', 10),

]
a = sorted(student_tuples, key=lambda student: student[2])
print("a = ",a)
# %%
# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place) ; print(place_up)

# Print out the number of o's in place
print(place.count("o"))
#%%
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 9.50 appears in areas
print(areas.count(9.50))
#%%
import numpy as np
np.array([1,2,3])

# %%
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))
# %%
height_in = [50, 73, 67, 67, 76, 74, 73, 70, 75, 70 ]
weight_lb = [ 195, 215, 215, 220, 220, 230, 195, 190, 195,198]
# height_in and weight_lb are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light =  np.array(bmi < 27)

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(np.array(bmi[bmi < 27]))
#%%
print(np.array([4, 3, 0]) + np.array([0, 2, 2]))
print(np.array([3, 3, False]) + np.array([True, 2, 2]))

 #%%

#################### Mehrdimensionale arrays##################

# %%
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)
# %%
baseball = [ [76, 220],
 [74, 207],
 [74, 225],
 [74, 207],
 [75, 212],
 [75, 225],
 [71, 170],
 [71, 190],
 [74, 210],
 [77, 230],
 [71, 210],
 [74, 200],
 [75, 238],
 [77, 234],
 [76, 222],
 [74, 200],
 [76, 190],
 [72, 170],
 [71, 220],
 [72, 223],
 [75, 210],
 [73, 215],
 [68, 196],
 [75, 220],
 [74, 228],
 [74, 190],
 [73, 204],
 [74, 165],
 [75, 216],
 [77, 220],
 [73, 208],
 [74, 210],
 [76, 215],
 [74, 195],
 [180, 78.4],
 [215, 102.7],
 [210, 98.5],
 [188, 75.2]]

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 10th row of np_baseball
print(np_baseball[9])

# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:,1]
print("Weight = ",np_weight_lb)

# Print out height of 124th player
np_height_m  = np_baseball[:,0]
print("Height = ",np_height_m)
# %%

############################# Numpy random 
############################# 
# random.normal(loc=0.0, scale=1.0, size=None)
# Draw random samples from a normal (Gaussian) distribution.
# loc : float or array_like of floats : Mean (“centre”) of the distribution.
# scale : float or array_like of floats : Standard deviation (spread or “width”) of the distribution. Must be non-negative.
# size : int or tuple of ints, optional :  Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

# %%
import numpy as np

height =  np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight =  np.round(np.random.normal(60.32, 15, 5000), 2)
np_city  = np.column_stack((height,weight))
print(" Height:",height.shape ,"  Mean = ", np.mean(height) , "    Median = ",np.median(height))
print(" Weight:",weight.shape ,"  Mean = ", np.mean(weight) , "    Median = ",np.median(weight))

# %%
# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))
# %%
# heights and positions are available as lists

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
#np_positions = np.array(positions)
np_positions  = np.array(['GK','A','GK','B','C'])
np_heights = np.array([188,170,190,175,181])

# Heights of the goalkeepers: gk_heights
gk_heights  = np_heights[np_positions == 'GK']

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)) )

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)) )
# %%
