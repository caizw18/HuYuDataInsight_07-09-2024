import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example data
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 6, 5]}
df = pd.DataFrame(data)

# Prepare the data
X = df[['x']]
y = df['y']

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plot the data and the regression line
plt.scatter(df['x'], df['y'], color='blue')
plt.plot(df['x'], predictions, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.show()

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')