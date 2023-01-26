#%%

numbers = list(range(10))
print(numbers)

iterator = iter(numbers)

print(next(iterator))
print(next(iterator))
print(next(iterator))


numbers = [(x, 2*x) for x in range(5)]
print(numbers)

iterator = iter(numbers)

print(next(iterator))
print(next(iterator))
print(next(iterator))
