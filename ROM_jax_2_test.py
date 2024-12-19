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
from GP_jax_2 import gp_evolution, MLP, gp_evolution_ML, MLP2



def Dataloader(data,batch_size,batch_time,seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    time_chunks = []
    for i in range(data.shape[1] - batch_time):
        time_chunks.append(data[:,i:i+batch_time])

    extra = len(time_chunks) % batch_size
    time_chunks = np.array(time_chunks[:-extra])
    split = np.random.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split


def Train(model, gp_rom_train, train_data, test_data, lr, epochs, name='try'):
    
    params = gp_rom_train.init_params()

    # optimizer = optax.adam(learning_rate = lr)
    # opt_state = optimizer.init(params)

    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # Add gradient clipping
    #     optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    #     optax.scale(-lr)
    # )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=optax.exponential_decay(
                init_value=lr,
                transition_steps=100,
                decay_rate=0.99
            ),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-4
        )
    )

    opt_state = optimizer.init(params)


    def loss(params,batch):
            preds, _=gp_rom_train.pod_deim_ML_evolve(params, batch[:, 0], train=True, num_steps=batch.shape[1])

            # print("Train:", jnp.unique(preds, return_counts=True))

            L = jnp.mean((jnp.abs(preds[:,1:] - batch[:,1:]))) # * 10
            # L = jnp.mean((jnp.square(preds[:,1:] - batch[:,1:])))

            return L

    vloss =  lambda params,batch :jnp.mean(jax.vmap(loss,in_axes=(None,0))(params,batch))
    
    # def validation_loss(params, test_data):
    #     test_data = test_data[:,:int(test_data.shape[1]/2)]
    #     preds, _ = gp_rom_train.pod_deim_ML_evolve(params, test_data[:, 0], train=False, num_steps=test_data.shape[1])
    #     # print("Val:", jnp.unique(preds, return_counts=True))
    #     L = jnp.mean((jnp.abs(preds[:,1:] - test_data[:,1:])))
    #
    #     # L = jnp.mean((jnp.square(preds[:, 1:] - test_data[:, 1:])))
    #
    #     return L


    def validation_loss(params, test_data):
        preds, _ = gp_rom_train.pod_deim_ML_evolve(params, test_data[:, 0], train=False, num_steps=test_data.shape[1])
        L = jnp.mean((jnp.abs(preds[:, 1:] - test_data[:, 1:])))

        return L

    
    # @jax.jit
    # def step(params, opt_state, batch):
    #     loss_value, grads = jax.value_and_grad(vloss,argnums=0)(params,batch)
    #     updates, opt_state = optimizer.update(grads, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     return params, opt_state, loss_value, grads
    
    
    # with open(f"{name}.csv", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Epoch", "Loss", "Val_Loss"])
    #     for i in range(1,epochs+1): #tqdm(range(epochs)):#
    #         losses = []
    #         for batch in train_data: # train_data:#
    #             params, opt_state, loss_value, grads = step(params,opt_state,batch)
    #             print(loss_value)
    #             losses.append(loss_value)
    #         net_loss = np.mean(np.array(losses))
    #         pickle.dump(params,open(f'params/{name}','wb'))
    #         val_loss = validation_loss(params, test_data)
    #         print(f'epochs={i}, net_loss={net_loss}, val_loss={val_loss}')
    #         writer.writerow([i, net_loss, val_loss])
    #         #test_loss = vloss(params,test_data[np.random.randint(len(test_data))])
    #         #print(f'net_loss: {net_loss}, test_loss: {test_loss}')
    #         #if test_loss < best_loss:
    #         #    best_loss = test_loss
    #         #    pickle.dump(params,open(f'params/{name}','wb'))
    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(vloss, argnums=0)(params, batch)

        grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)

        # grads = jnp.clip(grads, -1.0, 1.0)

        # Calculate gradient norm before clipping
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(x ** 2),
            grads,
            initializer=0.0
        ))

        has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | jnp.any(jnp.isnan(x)),
            grads,
            initializer=False
        )

        # Clip gradients
        # grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads, grad_norm, has_nan

    # Modified training loop
    with open(f"{name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss", "Val_Loss", "Grad_Norm"])
        best_loss = 0.0027  # np.inf
        for i in range(1, epochs + 1):
            losses = []
            grad_norms = []
            for batch in train_data:
                params, opt_state, loss_value, grads, grad_norm, has_nan = step(params, opt_state, batch)

                # print(loss_value)
                # if has_nan:
                #     print("nan in grad")

                losses.append(loss_value)
                grad_norms.append(grad_norm)
            net_loss = np.mean(np.array(losses))
            mean_grad_norm = np.mean(np.array(grad_norms))
            val_loss = validation_loss(params, test_data)

            if val_loss < best_loss:
                best_loss = val_loss
                print('Saving model...')
                pickle.dump(params, open(f'params/{name}_epoch_{i}_{best_loss:.4f}', 'wb'))

            print(f'epochs={i}, net_loss={net_loss}, val_loss={val_loss}, grad_norm={mean_grad_norm}')
            writer.writerow([i, net_loss, val_loss, mean_grad_norm])

def calc_loss(V,U,P,gp_rom_train, params, test_data):
    # test_data_half = test_data[:, :int(test_data.shape[1] / 2)]
    preds, _, _ = gp_rom_train.pod_deim_ML_evolve(params, test_data[:, 0], train=False, num_steps=test_data.shape[1])
    # print(preds.shape, test_data.shape)
    # print("Val:", jnp.unique(preds, return_counts=True))
    L_ml = jnp.mean((jnp.abs(preds[:, 1:] - test_data[:, 1:])))

    # L = jnp.mean((jnp.square(preds[:, 1:] - test_data[:, 1:])))
    gp_rom_ = gp_evolution(V, U, P, test_data[:, 0].reshape(np.shape(Ytilde)[0], 1))

    # ROM assessments - DEIM
    gp_rom.pod_deim_evolve()
    Ytilde_pod_deim_ = np.copy(gp_rom_.state_tracker)
    L_deim = jnp.mean((jnp.abs(Ytilde_pod_deim_[:, 1:] - test_data[:, 1:])))

    return L_ml, L_deim

# This is the ROM assessment

if __name__ == "__main__":

        # Snapshot collection for field
    # Note that columns of a snapshot/state are time always and a state vector is a column vector
    Y, Y_mean, Ytot = collect_snapshots_field() 
    F, F_mean, Ftot = collect_snapshots_nonlinear()


    # Field compression
    V, Ytilde = field_compression(Ytot,K) # K is the number of retained POD bases for field (and also dimension of reduced space)
    U, Ftilde_exact, P = nonlinear_compression(V,Ftot,M) # M is the number of retained POD basis for nonlinear term

    # print(U.shape)
    
    indices = np.zeros((P.shape[1]), dtype=int)

    for col in range(P.shape[1]):
        indices[col] = np.where(P[:, col] == 1)[0][0] 
    
    np.save('sampling_index_DEIM.npy',indices)

    # Initialize ROM class
    ytilde_init = Ytilde[:,0].reshape(np.shape(Ytilde)[0],1)
    # print(ytilde_init.shape)
    gp_rom = gp_evolution(V,U,P,ytilde_init)
    
    
    # ROM assessments - DEIM
    gp_rom.pod_deim_evolve()
    Ytilde_pod_deim = np.copy(gp_rom.state_tracker)

    np.save('POD_DEIM_test.npy',np.matmul(V, Ytilde_pod_deim))

    # ROM assessments - DEIM-ML
    model = MLP2(3072)

    sampling_factor = U.shape[1]

    batch_size = 16
    batch_time = 4
    lr = 1e-5

    # train_data = Dataloader(Ytilde, batch_size, batch_time, seed = 42)
    # print(train_data.shape)
    
    # num_steps = batch_time    
    # state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
    # state_tracker = state_tracker.at[:, 0].set(ytilde_init[:, 0])
    # nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps+1),dtype='double')
      
    name = f'batchtime_{batch_time}_lr_{lr:.1e}'
    # print(jnp.linalg.norm(P))
    # gp_rom_ml_train = gp_evolution_ML(V,U,P, model, sampling_factor)
    # Train(model, gp_rom_ml_train, train_data, Ytilde, lr=lr, epochs=3000, name=name)
        # pod_deim_ML_evolve(params, test_data[:, 0], train=False, num_steps=test_data.shape[1])

    name = 'batchtime_4_lr_1.0e-03_re_1000_epoch_28_0.0190'

    model = MLP2(3072)
    params = pickle.load(open(f'params/{name}','rb'))
    gp_rom_ml_test = gp_evolution_ML(V,U,P, model, sampling_factor)

    loss_ml, loss_deim = calc_loss(V, U, P, gp_rom_ml_test, params, Ytilde)

    print(f"ML loss: {loss_ml}, DEIM loss: {loss_deim}")

    Ytilde_pod_deim_ML, sampling_index, _ = gp_rom_ml_test.pod_deim_ML_evolve(params, ytilde_init[:, 0], train=False,
                                                                           num_steps=1000)

    # ml_loss = jnp.mean((jnp.abs(Ytilde_pod_deim_ML[:, 1:] - ytilde_init[:, 1:])))
    # deim_loss = jnp.mean((jnp.abs(Ytilde_pod_deim[:, 1:] - ytilde_init[:, 1:])))
    #
    # print(f"ML loss (alt): {ml_loss}, DEIM loss (alt): {deim_loss}")


    np.save(f'sampling_index_DEIM_ML_{name}.npy', sampling_index)
    np.save(f'POD_DEIM_ML_{name}.npy',np.matmul(V, Ytilde_pod_deim_ML))


    print('Saving data')
    
    
    





