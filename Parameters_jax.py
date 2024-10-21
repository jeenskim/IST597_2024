import jax.numpy as jnp

# Parameters
Rnum = 1000.0
spatial_resolution = 128 # Grid points
temporal_resolution = 300 # Snapshots
final_time = 2.0 # Keep fixed

# Independent variables
x = jnp.linspace(0.0,1.0,num=spatial_resolution)
dx = 1.0/jnp.shape(x)[0]
tsteps = jnp.linspace(0.0,2.0,num=temporal_resolution)
dt = final_time/jnp.shape(tsteps)[0]

# Compression metrics - POD field
K = 12  # num_modes field
# Compression metrics - DEIM nonlinear
M = 24 # num_modes nonlinear
fac = 2 # Gappy DEIM parameter
gappy = False # True or False for gappy DEIM

# ML hyperparameters
num_epochs = 100
num_neurons = 50
seq_num = 10

if __name__ == "__main__":
    print('Parameter file')