import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tensorflow as tf

# Function to plot the results
def plot(X, F, S, P):
    plt.subplot(2, 2, 1)
    plt.plot(times, X)
    plt.xlabel('time(hr)')
    plt.ylabel('Total biomass concentration(g/L)')

    plt.subplot(2, 2, 2)
    plt.plot(times, F)
    plt.xlabel('time(hr)')
    plt.ylabel('fat-free biomass concentration(g/L)')

    plt.subplot(2, 2, 3)
    plt.plot(times, S)
    plt.xlabel('time(hr)')
    plt.ylabel('glucose(g/L)')

    plt.subplot(2, 2, 4)
    plt.plot(times, P)
    plt.xlabel('time(hr)')
    plt.ylabel('GLA(g/L)')

    plt.show()

kf = 0.00135
kx = 0.0017
ks = 0.0031
kp = 0.0000323
Ks = 48.64
# ODE system
def ode_system(t, y, theta):
    x, f, s, p = y
    
    yx, yf, ys, yp = theta[:,0], theta[:,1], theta[:,2], theta[:,3]
   
    
    dx_dt = yx * (s / (s + Ks * x)) * x - (kx * x)
    df_dt = yf * (s / (s + Ks * x)) * x - (kf * x)
    ds_dt = -ys * (s / (s + Ks * x)) * x - (ks * x * (s / (s + 0.1)))
    dp_dt = yp * (s / (s + Ks * x)) * x - (kp * x)

    return [dx_dt, df_dt, ds_dt, dp_dt]

# Combine ODE and Neural Network
def combined_system(t, y, nn_model):
    # Extract the parameters from the neural network
    theta = nn_model(tf.constant([[t]], dtype=tf.float32))

    # Use the ODE system with time-varying parameters
    dydt = ode_system(t, y, theta)
    return dydt

# Solve the Coupled System
def solve_coupled_system(initial_state, times, nn_model):
    # Create a function that wraps the combined system for solve_ivp
    def wrapper(t, y):
        return np.array(combined_system(t, y, nn_model)).ravel()
    # Use solve_ivp to solve the system
    solution = solve_ivp(wrapper, [times[0], times[-1]], initial_state, t_eval=times)
    return solution

# Polynomial equations for ys, yp, yf, yx
def polynomial_equations(t):
    ys = (0.131521964231799) + ((0.000288810956006872) * t) + ((-4.36218325082568E-07) * (t * t)) + (
        (1.78626182553539E-07) * (t * t * t)) + ((-1.20517173730103E-09) * (t * t * t * t)) + (
             (2.9245613225185E-12) * (t * t * t * t * t)) + (
                 (-2.49061203018756E-15) * (t * t * t * t * t * t))
    yp = (0.00176302960718107) + ((-0.0000109239822661364) * t) + ((3.80821065308153E-07) * (t * t)) + (
        (-3.09143445331959E-10) * (t * t * t)) + ((-7.5041030481648E-12) * (t * t * t * t)) + (
              (2.60309484672425E-14) * (t * t * t * t * t)) + ((-2.56955399004401E-17) * (t * t * t * t * t * t))
    yf = (0.187299951265649) + ((0.000197389335883713) * t) + ((-0.0000165341666570098) * (t * t)) + (
        (1.9871373010332E-07) * ((t * t * t))) + ((-8.30399028489208E-10) * (t * t * t * t)) + (
              (1.46497968261375E-12) * (t * t * t * t * t)) + ((-9.32012803097304E-16) * (t * t * t * t * t * t))
    yx = (0.2274805609435) + ((0.000585954306142759) * t) + ((-0.0000274584467252314) * (t * t)) + (
        (2.87345758721163E-07) * ((t * t * t))) + ((-1.15825748856662E-09) * (t * t * t * t)) + (
              (2.03068133063216E-12) * (t * t * t * t * t)) + ((-1.29861805832893E-15) * (t * t * t * t * t * t))

    return ys, yp, yf, yx

# Create data for training
times = np.linspace(0, 405, 1000)
ys, yp, yf, yx = polynomial_equations(times)

# Function to create and train the neural network
def create_and_train_nn_model(times, actual_values):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=40, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(units=4, activation='linear')  # Output layer with 4 neurons for yx, yf, ys, yp
    ])

    def composite_loss(y_true, y_pred):
        mse_loss = tf.keras.losses.MAPE(y_true, y_pred)
        return mse_loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=optimizer, loss=composite_loss)

    model.fit(times.reshape(-1, 1), actual_values, epochs=6000, batch_size=24)

    return model

# Train the neural network
nn_model = create_and_train_nn_model(times, np.array([yx, yf, ys, yp]).T)

# Solve the coupled system using the trained neural network
initial_state = np.array([0.477527722, 0.165917233, 58.91592385, 0], dtype=np.float32)
solution = solve_coupled_system(initial_state, times, nn_model)
x, f, s, p = solution.y

# Plotting
plot(x, f, s, p)


