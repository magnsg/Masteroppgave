import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.special as sp


# Load the .npz file
data = np.load('result1.npz')

# Access the arrays
S = data['arr_0']  # This is the first array saved (S)
pi = data['arr_1']  # This is the second array saved (pi)
lambdaRate = data['arr_2']  # This is the third array saved (lambdaRate)
numstates = data['arr_3']  # This is the fourth array saved (numstates)

# If you named the arrays when saving, you can access them by their names:
# S = data['S']
# pi = data['pi']
# lambdaRate = data['lambdaRate']

# Close the file
data.close()



data = sio.loadmat('MouseData.mat')
headdata = data['resampledAwakeHeadAngleData'].flatten()

# Plot the state sequence
plt.plot(S)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('State sequence')
plt.show()

# Plot the transition matrix
plt.imshow(pi, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Transition matrix')
plt.show()

# Plot the head data colored by state
for i in range(numstates):
    plt.plot(np.arange(len(headdata))[S == i], headdata[S == i], label='State ' + str(i))
plt.xlabel('Time')
plt.ylabel('Head angle')
plt.title('Head data colored by state')
plt.legend()
plt.show()

# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Plot the head data colored by state
for i in range(numstates):
    time = np.arange(len(headdata))[S == i]
    angles = headdata[S == i]
    ax.plot(angles, time, 'o', label='State ' + str(i))

# Set the direction of the theta-axis to increase clockwise
ax.set_theta_direction(-1)

# Set the zero location of the theta-axis to the top
ax.set_theta_offset(np.pi / 2.0)

# Set the labels
ax.set_xlabel('Head angle (degrees)')
ax.set_ylabel('Time')
ax.set_title('Head data colored by state (radial plot)')

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Show the plot
plt.show()


"""""
# Plot the rate matrix
plt.imshow(lambdaRate, cmap='hot', interpolation='nearest') # Change cmap to 'hot' for better visualization of the rate matrix
plt.colorbar()  # Add a colorbar
plt.title('Rate matrix')    # Add a title
plt.show()  # Display the plot
"""""