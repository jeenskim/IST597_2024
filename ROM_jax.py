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

config.update("jax_enable_x64", True)



#jnp.random.seed(10)

# Import relevant modules
from Parameters import K,M, spatial_resolution
from Compression import field_compression, nonlinear_compression
from Problem import collect_snapshots_field, collect_snapshots_nonlinear
from Plotting import plot_coefficients, plot_gp, plot_comparison
from GP_jax import gp_evolution, MLP




    
    
def Dataloader(data,batch_size,batch_time):
    time_chunks = []
    for i in range(data.shape[1] - batch_time):
        #time_chunks.append(data[i:i+batch_time])
        time_chunks.append(data[:,i:i+batch_time])
        #print(data[:,i:i+batch_time].shape)
    # print(len(time_chunks))
    extra = len(time_chunks) % batch_size
    time_chunks = np.array(time_chunks[:-extra])
    split = np.random.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    # print(split.shape)
    return split


def Train(model, gp_rom_train, train_data, test_data, lr, epochs, name='try'):
    
    params = gp_rom_train.init_params()

    optimizer = optax.adam(learning_rate = lr)
    opt_state = optimizer.init(params)


    def loss(params,batch):
            preds=gp_rom_train.pod_deim_ML_evolve(params, batch[:, 0])

            #print (preds.shape)
            #exit()
            #preds = model(params,batch[0])
            L = jnp.mean((jnp.abs(preds[:,1:] - batch[:,1:])))
            return jnp.linalg.norm(preds[:,1:])
        

    vloss =  lambda params,batch :jnp.mean(jax.vmap(loss,in_axes=(None,0))(params,batch))
    
    
    #@jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(vloss,argnums=0)(params,batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads
    
    

    for i in range(1,epochs+1): #tqdm(range(epochs)):#
        losses = []
        for batch in tqdm(train_data): # train_data:#
            params, opt_state, loss_value, grads = step(params,opt_state,batch)
            print(grads)
            losses.append(loss_value)
        net_loss = np.mean(np.array(losses))
        print(net_loss)
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

    
    indices = np.zeros((P.shape[1]), dtype=int)

    for col in range(P.shape[1]):
        indices[col] = np.where(P[:, col] == 1)[0][0] 
    
    np.save('sampling_index_DEIM.npy',indices)
    
    # Initialize ROM class
    ytilde_init = Ytilde[:,0].reshape(np.shape(Ytilde)[0],1)
    # gp_rom = gp_evolution(V,U,P,ytilde_init)

    # #Filter exact for stability
    # Ftilde_filtered = np.copy(Ftilde_exact)
    # for i in range(K):
    #     Ftilde_filtered[i,:] = savgol_filter(Ftilde_exact[i,:],65,2)
    #     #Ftilde_filtered=Ftilde_filtered.at[i,:].set(savgol_filter(Ftilde_exact[i,:],65,2))
    # # Plot comparison of filtered and exact nonlinear term modes
    # plot_gp(Ftilde_exact,'Exact',Ftilde_filtered,'Filtered')

    #  # ROM assessments - GP
    # gp_rom.pod_gp_evolve()
    # Ytilde_pod_gp = np.copy(gp_rom.state_tracker)
    # # Plot comparison of POD-GP and truth
    # plot_gp(Ytilde,'True',Ytilde_pod_gp,'POD-GP')
    
    # ROM assessments - DEIM-ML
    
    layer_sizes = [spatial_resolution, 50, 50, 50, spatial_resolution]
    model = MLP(1536)
    sampling_factor = 10
    #gp_rom.set_ML_sampling(10)
    
    
    batch_size = 60
    batch_time = 10
    train_data = Dataloader(Ytilde, batch_size, batch_time)
    
    #num_steps = int(jnp.shape(tsteps)[0]-1)
    num_steps = batch_time    
    state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
    state_tracker = state_tracker.at[:, 0].set(ytilde_init[:, 0])
    nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
      
      
    gp_rom_train = gp_evolution(V,U,P,batch_time, model, sampling_factor)
    # gp_rom_train.num_steps = batch_time
    # gp_rom_train.state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],gp_rom_train.num_steps),dtype='double')
    # gp_rom_train.nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],gp_rom_train.num_steps),dtype='double')
    # gp_rom_train.set_ML_sampling(10)
    # gp_rom_train.k_index_tracker = jnp.zeros(shape=(int(jnp.shape(gp_rom_train.V)[0]/gp_rom_train.sampling_factor),gp_rom_train.num_steps+1),dtype='int')


    Train(model, gp_rom_train, train_data, train_data, lr=1e-4, epochs=10, name='try')
    
    
    # sampling_index = jnp.copy(gp_rom_train.k_index_tracker)
    # np.save('sampling_index_DEIM_ML_train.npy', sampling_index)
        
    

    

    # gp_rom.pod_deim_ML_evolve(model)
    # Ytilde_pod_deim_ML = jnp.copy(gp_rom.state_tracker)
    
    
    
    
    
    # plot_gp(Ytilde,'True',Ytilde_pod_deim_ML,'POD-DEIM_ML')
    
    # sampling_index = jnp.copy(gp_rom.k_index_tracker)
    # print(sampling_index)
    # #print(sampling_index.shape)

    # np.save('sampling_index_DEIM_ML.npy', sampling_index)
    # np.save('POD_DEIM_ML.npy',np.matmul(V, Ytilde_pod_deim_ML))
    
    


    print('Saving data')
    
    
    





