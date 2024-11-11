import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from jax import config
import optax
from tqdm import tqdm
from Parameters_jax import tsteps, Rnum, dx, dt, seq_num
import pickle
import csv

config.update("jax_enable_x64", True)



#jnp.random.seed(10)

# Import relevant modules
from Parameters import K,M, spatial_resolution
from Compression import field_compression, nonlinear_compression
from Problem import collect_snapshots_field, collect_snapshots_nonlinear
from Plotting import plot_coefficients, plot_gp, plot_comparison
from GP_jax import gp_evolution, MLP, gp_evolution_ML




    
    
def Dataloader(data,batch_size,batch_time):
    time_chunks = []
    for i in range(data.shape[1] - batch_time):
        time_chunks.append(data[:,i:i+batch_time])

    extra = len(time_chunks) % batch_size
    time_chunks = np.array(time_chunks[:-extra])
    split = np.random.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split


def Train(model, gp_rom_train, train_data, test_data, lr, epochs, name='try'):
    
    params = gp_rom_train.init_params()

    optimizer = optax.adam(learning_rate = lr)
    opt_state = optimizer.init(params)


    def loss(params,batch):
            preds, _=gp_rom_train.pod_deim_ML_evolve(params, batch[:, 0])

            L = jnp.mean((jnp.abs(preds[:,1:] - batch[:,1:])))
            #L = jnp.mean((jnp.square(preds[:,1:] - batch[:,1:])))
            
            return L
        

    vloss =  lambda params,batch :jnp.mean(jax.vmap(loss,in_axes=(None,0))(params,batch))
    
    
    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(vloss,argnums=0)(params,batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads
    
    
    with open(f"{name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss"]) 
        for i in range(1,epochs+1): #tqdm(range(epochs)):#
            losses = []
            for batch in train_data: # train_data:#
                params, opt_state, loss_value, grads = step(params,opt_state,batch)
                losses.append(loss_value)
            net_loss = np.mean(np.array(losses))
            pickle.dump(params,open(f'params/{name}','wb'))
            print(f'epochs={i}, net_loss={net_loss}')
            writer.writerow([i, net_loss])
            #test_loss = vloss(params,test_data[np.random.randint(len(test_data))])
            #print(f'net_loss: {net_loss}, test_loss: {test_loss}')
            #if test_loss < best_loss:
            #    best_loss = test_loss
            #    pickle.dump(params,open(f'params/{name}','wb'))


        


# This is the ROM assessment

if __name__ == "__main__":

        # Snapshot collection for field
    # Note that columns of a snapshot/state are time always and a state vector is a column vector
    Y, Y_mean, Ytot = collect_snapshots_field() 
    F, F_mean, Ftot = collect_snapshots_nonlinear()


    # Field compression
    V, Ytilde = field_compression(Ytot,K) # K is the number of retained POD bases for field (and also dimension of reduced space)
    U, Ftilde_exact, P = nonlinear_compression(V,Ftot,M) # M is the number of retained POD basis for nonlinear term

    print(U.shape)
    
    indices = np.zeros((P.shape[1]), dtype=int)

    for col in range(P.shape[1]):
        indices[col] = np.where(P[:, col] == 1)[0][0] 
    
    np.save('sampling_index_DEIM.npy',indices)
    
    
    # Initialize ROM class
    ytilde_init = Ytilde[:,0].reshape(np.shape(Ytilde)[0],1)
    gp_rom = gp_evolution(V,U,P,ytilde_init)
    
    
    # ROM assessments - DEIM
    gp_rom.pod_deim_evolve()
    Ytilde_pod_deim = np.copy(gp_rom.state_tracker)

    np.save('POD_DEIM_test.npy',np.matmul(V, Ytilde_pod_deim))
    
    
    # ROM assessments - DEIM-ML
    model = MLP(3072)
    sampling_factor = U.shape[1]

    
    
    batch_size = 16
    batch_time = 2
    lr = 1e-5
    train_data = Dataloader(Ytilde, batch_size, batch_time)
    
    num_steps = batch_time    
    state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
    state_tracker = state_tracker.at[:, 0].set(ytilde_init[:, 0])
    nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
      
      
    name = f'batchtime_{batch_time}_lr_{lr:.1e}'
    
    gp_rom_ml_train = gp_evolution_ML(V,U,P,batch_time, model, sampling_factor, train=True)
    Train(model, gp_rom_ml_train, train_data, train_data, lr=lr, epochs=2500, name=name)
    
    
    # model = MLP(3072)
    # params = pickle.load(open(f'params/{name}','rb'))
    # gp_rom_ml_test = gp_evolution_ML(V,U,P,1000, model, sampling_factor, train=False)
    # Ytilde_pod_deim_ML, sampling_index   = gp_rom_ml_test.pod_deim_ML_evolve(params, ytilde_init[:, 0])
    

    # np.save(f'sampling_index_DEIM_ML_{name}.npy', sampling_index)
    # np.save(f'POD_DEIM_ML_{name}.npy',np.matmul(V, Ytilde_pod_deim_ML))
    
    


    print('Saving data')
    
    
    

