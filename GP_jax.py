import jax.numpy as jnp
import numpy as np
import jax
import jax.nn as jnn
from jax import random
import flax
import flax.linen as nn
from Parameters_jax import tsteps, Rnum, dx, dt, seq_num
from Problem_jax import nl_calc
import optax
import tqdm
import pickle
from jax.nn import sigmoid

class gp_evolution(flax.struct.PyTreeNode):
    V : jnp.array
    U : jnp.array
    P : jnp.array
    num_steps : int
    model : nn.Module
    sampling_factor : int
    
    # '''
    # Does a POD/POD-DEIM GP evolution
    # '''
    # #def __init__(self,V,U,P,ytilde_init,model):
    #     '''
    #     ytilde - initial condition in POD space - shape: Kx1
    #     V - Truncated POD bases (precomputed) - shape: NxK
    #     V - Truncated POD bases for nonlinear snapshots (precomputed) - shape: NxM
    #     P - DEIM matrix - shape: NxM
    #     '''
        
    #     # self.V = V
    #     # self.U = U
    #     # self.P = P
    #     # self.num_steps = int(jnp.shape(tsteps)[0]-1)
    #     # self.state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps+1),dtype='double')
    #     # self.state_tracker = self.state_tracker.at[:, 0].set(ytilde_init[:, 0])
    #     # self.nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps+1),dtype='double')
    #     # self.model = model
    
    def init_params(self):
        return self.model.init(jax.random.PRNGKey(0),jnp.zeros((128)))
        
        
    def set_ML_sampling(self, sampling_factor):
        k_index_tracker = jnp.zeros(shape=(int(jnp.shape(self.V)[0]/sampling_factor),self.num_steps+1),dtype='int')

        

    def linear_operator_fixed(self):
        '''
        This method fixes the laplacian based linear operator
        '''
        N_ = jnp.shape(self.V)[0]
        laplacian_matrix = -2.0*jnp.identity(N_)

        # Setting upper diagonal
        i_range = jnp.arange(0,N_-1,dtype='int')
        j_range = jnp.arange(1,N_,dtype='int')

        laplacian_matrix = laplacian_matrix.at[i_range,j_range].set(1.0)

        # Setting lower diagonal
        i_range = jnp.arange(1,N_,dtype='int')
        j_range = jnp.arange(0,N_-1,dtype='int')

        laplacian_matrix = laplacian_matrix.at[i_range,j_range].set(1.0)


        # Periodicity
        laplacian_matrix=laplacian_matrix.at[0,N_-1].set(1.0)
        laplacian_matrix=laplacian_matrix.at[N_-1,0].set(1.0)
        laplacian_matrix = 1.0/(Rnum*dx*dx)*laplacian_matrix
        
        # Final Linear operator
        linear_matrix = jnp.matmul(jnp.matmul(jnp.transpose(self.V),laplacian_matrix),self.V)
        
        return linear_matrix

    def linear_operator(self,Ytilde, linear_matrix):
        '''
        Calculates the linear term on the RHS
        '''
        M_ = jnp.shape(self.V)[1]
        return jnp.matmul(linear_matrix,Ytilde).reshape(M_,1)

    def nonlinear_operator_pod(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the standard POD-GP technique
        '''
        N_ = jnp.shape(self.V)[0]
        field = jnp.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return jnp.matmul(jnp.transpose(self.V),nonlinear_field_approx)

    def pod_gp_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod(state)
    
        return jnp.add(linear_term,-non_linear_term)

    def pod_gp_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_gp
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        state = jnp.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod(state)
        #self.nl_state_tracker[:,0] = non_linear_term[:,0]
        self.nl_state_tracker=self.nl_state_tracker.at[:,0].set(non_linear_term[:,0])

        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_gp_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l2)
            #state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
            state = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            #self.state_tracker[:,t] = state[:]
            self.state_tracker=self.state_tracker.at[:,t].set(state[:])
            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod(state)
            #self.nl_state_tracker[:,t] = non_linear_term[:,0]
            self.nl_state_tracker=self.nl_state_tracker.at[:,t].set(non_linear_term[:,0])

    def deim_matrices_precompute(self):
        '''
        Precompute the DEIM matrices for nonlinear term in ODE
        '''
        mtemp = jnp.linalg.pinv(jnp.matmul(jnp.transpose(self.P),self.U)) #ok MxM
        mtemp = jnp.matmul(self.U,mtemp) # NxM
        mtemp = jnp.matmul(mtemp,jnp.transpose(self.P)) # NxN
        
        self.varP = jnp.matmul(jnp.transpose(self.V),mtemp) # KxN
        
    def deim_matrices_MLsampling(self, ytilde, params):
        '''
        Select sampling points for non-linear term calculations: two options
        
        1. Global Operation(using MLP)
        * Input: state variable at all points - shape: Nx1 / Output: Characteristic variable field - shape: Nx1
        * Sort by descending order to truncate by Kx1 : sampling points
        
        2. Node-wise Operation (using projection vector-Shivam's Top-K)
        * X: state variable - shape: NxNf / p: projection vector - shape: Nfx1 / y: projected field -shape: Nx1
        * Multiply p to each point to get y -> sort by descending order to truncate by Kx1 : sampling points
        '''
        N_ = jnp.shape(self.V)[0]
        field = jnp.matmul(self.V, ytilde).reshape(N_,1)
        
        # Numer of sampling point    
        k_sampling=int(N_/self.sampling_factor)
        
        #calculate characteristic variable field using MLP
        #print(field.shape)
        #y=self.model.apply(params, field[:,0])
        # k_index=self.model.apply(params, field[:,0])
        # print(k_index.shape)
        
        P_ML = self.model.apply(params, field[:,0])
        
        # k_index = jnp.argsort(y, descending=True)[:k_sampling]
        
        k_index = jnp.ones((12))
        #y_minus=jnp.array(-y, dtype=np.float64)
        #print(y)
        # print(k_index.shape)
        # print(k_sampling)
        
        #sorting y by descending order to truncate by Kx1 & making k_index vector - shape: Kx1
        #k_index=jnp.argsort(y_minus)[:k_sampling]
        
        #build P matrix using k_index
        # P_ML=jnp.ones((N_, k_sampling)) * k_index[0]
        
        # for j in range(k_sampling):
        #     i=k_index[j]
        #     #self.P[i][j]=1
        #     P_ML=P_ML.at[i,j].set(1)
            
        # print(P_ML.shape)
            
        #P_ML=jnp.ones((N_, k_sampling)) *y[0]
        # P_ML=jnp.ones((N_, k_sampling)) *y[k_index][3]
        # P_ML=jnp.ones((N_, k_sampling)) * k_index[1]
        #print(P_ML)
        
        mtemp = jnp.linalg.pinv(jnp.matmul(jnp.transpose(P_ML),self.U)) #ok MxM
        mtemp = jnp.matmul(self.U,mtemp) # NxM
        mtemp = jnp.matmul(mtemp,jnp.transpose(P_ML)) # NxN
        
        varP = jnp.matmul(jnp.transpose(self.V),mtemp) # KxN
        
 
        
        return k_index, varP
          

    def nonlinear_operator_pod_deim(self, ytilde, varP):
        '''
        Defines the nonlinear reconstruction using the POD-DEIM technique
        '''
        N_ = jnp.shape(self.V)[0]
        field = jnp.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return jnp.matmul(varP,nonlinear_field_approx)

    def pod_deim_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod_deim(state)
    
        return jnp.add(linear_term,-non_linear_term)
    
    
    def pod_deim_rhs_MLsampling(self,state, linear_matrix, varP):
        '''
        Calculate the rhs of the POD GP implementation using ML-sampling
        '''
        linear_term = self.linear_operator(state, linear_matrix)
        non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    
        return jnp.add(linear_term,-non_linear_term)
        

    def pod_deim_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_deim
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        self.deim_matrices_precompute()
        state = jnp.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod_deim(state)
        #self.nl_state_tracker[:,0] = non_linear_term[:,0]
        nl_state_tracker=nl_state_tracker.at[:,0].set(non_linear_term[:,0])
        
        
        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_deim_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l2)
            #state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
            state = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
            
            #self.state_tracker[:,t] = state[:]
            state_tracker=state_tracker.at[:,t].set(state[:])
            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod_deim(state)
            #self.nl_state_tracker[:,t] = non_linear_term[:,0]
            nl_state_tracker=nl_state_tracker.at[:,t].set(non_linear_term[:,0])

    # def slfn_gp_evolve(self,model):
    #     '''
    #     Use Euler forward to do a system evolution for slfn_gp
    #     '''
    #     # Setup fixed operations
    #     self.linear_operator_fixed()
    #     self.deim_matrices_precompute()
    #     state = jnp.copy(self.state_tracker[:,0])
    #     N_ = jnp.shape(state)[0]
    #     non_linear_term = self.nonlinear_operator_pod_deim(state)
    #     for t in range(1,self.num_steps+1):
            
    #         linear_term = self.linear_operator(state)
    #         non_linear_term = jnp.transpose(model.predict(non_linear_term.reshape(1,N_)))
    #         rhs = jnp.add(linear_term,-non_linear_term.reshape(N_,1))
    #         state[:] = state[:] + dt*rhs[:,0]

    #         self.state_tracker[:,t] = state[:]

    # def lstm_gp_evolve(self,model):
    #     '''
    #     Use Euler forward to do a system evolution for slfn_gp
    #     '''
    #     # Setup fixed operations
    #     state = jnp.copy(self.state_tracker[:,0])
    #     self.state_tracker[:,1:] = 0.0
    #     self.linear_operator_fixed()
    #     self.deim_matrices_precompute()
    #     N_ = jnp.shape(state)[0]
    #     non_linear_term_input = jnp.zeros(shape=(1,seq_num,N_))

    #     for t in range(0,seq_num):
    #         rhs = self.pod_deim_rhs(state)
    #         l1 = state + dt*rhs[:,0]

    #         rhs = self.pod_deim_rhs(l1)
    #         l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

    #         rhs = self.pod_deim_rhs(l2)
    #         state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
    #         self.state_tracker[:,t] = state[:]

    #         non_linear_term_input[0,t,:] = self.nonlinear_operator_pod_deim(state)[:,0]
       
    #     for t in range(seq_num,self.num_steps+1):
    #         linear_term = self.linear_operator(state)
    #         non_linear_term_next = jnp.transpose(model.predict(non_linear_term_input))

    #         rhs = jnp.add(linear_term,-non_linear_term_next.reshape(N_,1))
            
    #         state[:] = state[:] + dt*rhs[:,0]
    #         self.state_tracker[:,t] = state[:]

    #         non_linear_term_input[0,:-1,:] = non_linear_term_input[0,1:,:]
    #         non_linear_term_input[0,-1,:] = non_linear_term_next[:,0]

    def pod_deim_ML_evolve(self, params, ytilde_init):
        '''
        Use RK3 to do a system evolution for pod_deim
        '''
        
        state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps),dtype='double')
        state_tracker = state_tracker.at[:, 0].set(ytilde_init)
        nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps),dtype='double')
        k_index_tracker = jnp.zeros(shape=(int(jnp.shape(self.V)[0]/self.sampling_factor),self.num_steps),dtype='int')
        #k_index_tracker = jnp.zeros(shape=(128,self.num_steps),dtype='int')
        

        # Setup fixed operations
        linear_matrix = self.linear_operator_fixed()
        state = state_tracker[:,0]
        
        k_index, varP = self.deim_matrices_MLsampling(state, params)
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
        #self.nl_state_tracker[:,0] = non_linear_term[:,0]
        nl_state_tracker=nl_state_tracker.at[:,0].set(non_linear_term[:,0])
        k_index_tracker=k_index_tracker.at[:,0].set(k_index)
        
        
        for t in range(1,self.num_steps+1):
            rhs = self.pod_deim_rhs_MLsampling(state, linear_matrix, varP)
            l1 = state + dt*rhs[:,0]

            k_index, varP = self.deim_matrices_MLsampling(l1, params)
            rhs = self.pod_deim_rhs_MLsampling(l1, linear_matrix, varP)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            k_index, varP = self.deim_matrices_MLsampling(l2, params)
            rhs = self.pod_deim_rhs_MLsampling(l2, linear_matrix, varP)
            #state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
            state = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            state_tracker=state_tracker.at[:,t].set(state[:])
            

            # Recording the nonlinear term
            k_index, varP = self.deim_matrices_MLsampling(state, params)
            #print(k_index)
            
            non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
            nl_state_tracker=nl_state_tracker.at[:,t].set(non_linear_term[:,0])
            
            k_index_tracker=k_index_tracker.at[:,t].set(k_index)
            
        return state_tracker
            
            
        



# # A helper function to randomly initialize weights and biases
# # for a dense neural network layer
# def random_layer_params(m, n, key, scale=1e-2):
#     w_key, b_key = random.split(key)
#     return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# # Initialize all layers for a fully-connected neural network with sizes "sizes"
# def init_network_params(sizes, key):
#     keys = random.split(key, len(sizes))
#     return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# class MLP:
#     def __init__(self, layer_sizes, key):
#         self.layer_sizes = layer_sizes
#         #self.params = self.init_network_params(layer_sizes, key)

#     def init_network_params(self, sizes, key):
#         keys = jax.random.split(key, len(sizes) - 1)
#         params = []
#         for in_size, out_size, k in zip(sizes[:-1], sizes[1:], keys):
#             w_key, b_key = jax.random.split(k)
#             weight = jax.random.normal(w_key, (out_size, in_size))
#             bias = jax.random.normal(b_key, (out_size,))
#             params.append((weight, bias))
#         return params

#     def __call__(self, params, y):
#         activations = y
#         for w, b in params[:-1]:
#             outputs = jnp.dot(w, activations) + b
#             activations = jax.nn.sigmoid(outputs)
        
#         final_w, final_b = params[-1]
#         logits = jnp.dot(final_w, activations) + final_b
#         return logits


class MLP(nn.Module):
    # spatial_resolution : 128
    spatial_resolution : 1536
    
    @nn.compact
    def __call__(self, x):
        ###---------------CNN --------------------
        # x = nn.Dense(features = 200)(x)
        # x = nn.gelu(x)
        # x = nn.Dense(features = 200)(x)
        # x = nn.gelu(x)
        x = nn.Dense(self.spatial_resolution)(x)
        x = x.reshape((128,12))
        # x = nn.sigmoid(x) * 128
        # x = jnp.array(x, dtype=jnp.int32)
        # sort = jax.tree_map(lambda x:jnp.argsort(x, descending=True)[:k_sampling])
        # sort = jnp.argsort(x, descending=True)
        # k_index = sort[:k_sampling]
        # x = nn.gelu(x) # relu(x) #
        # x = nn.Conv(features=self.latent_dim, padding='CIRCULAR', kernel_size=(3, 3, 3))(x) #4, 4, 32 ---> 4, 4, 40
        return x
    

