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

class gp_evolution_ML(flax.struct.PyTreeNode):
    V : jnp.array
    U : jnp.array
    P : jnp.array
    # num_steps : int
    model : nn.Module
    sampling_factor : int
    # train : bool
    
    def init_params(self):
        return self.model.init(jax.random.PRNGKey(0),jnp.zeros((128)), training=False)
        

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

    def deim_matrices_precompute(self):
        '''
        Precompute the DEIM matrices for nonlinear term in ODE
        '''
        # np.save('P.npy',self.P)
        
        k_index = jnp.argmax(self.P, axis=0)
        
        mtemp = np.linalg.pinv(np.matmul(np.transpose(self.P),self.U)) #ok MxM
        mtemp = np.matmul(self.U,mtemp) # NxM
        mtemp = np.matmul(mtemp,np.transpose(self.P)) # NxN
        
        varP = np.matmul(np.transpose(self.V),mtemp) # KxN
        
        return k_index, varP

    def straight_through_sampling(self, logits, train=True, hard_threshold=0.9):
        probs = jnn.softmax(logits, axis=0)

        if not train:
            # print("validating...")
            result = jnp.zeros_like(probs)
            max_indices = jnp.argmax(probs, axis=0)
            return result.at[max_indices, jnp.arange(probs.shape[1])].set(1), 0.0

        sample = probs

        l1_penalty = 0.01 * jnp.sum(jnp.abs(probs))

        return sample, l1_penalty

    def deim_matrices_MLsampling(self, ytilde, params, train):
        N_ = jnp.shape(self.V)[0]
        field = jnp.matmul(self.V, ytilde).reshape(N_, 1)

        key = jax.random.PRNGKey(seed=42)
        logits = self.model.apply(params, field[:, 0], training=train, rngs={'dropout': key})

        P_ML, l1_penalty = self.straight_through_sampling(logits, train=train)
        k_index = jnp.argmax(P_ML, axis=0)

        mtemp = jnp.linalg.pinv(jnp.matmul(jnp.transpose(P_ML), self.U))
        mtemp = jnp.matmul(self.U, mtemp)
        mtemp = jnp.matmul(mtemp, jnp.transpose(P_ML))
        varP = jnp.matmul(jnp.transpose(self.V), mtemp)

        return k_index, varP, l1_penalty
        
    # def deim_matrices_MLsampling(self, ytilde, params, train):
    #     '''
    #     Select sampling points for non-linear term calculations: two options
    #
    #     1. Global Operation(using MLP)
    #     * Input: state variable at all points - shape: Nx1 / Output: Characteristic variable field - shape: Nx1
    #     * Sort by descending order to truncate by Kx1 : sampling points
    #
    #     2. Node-wise Operation (using projection vector-Shivam's Top-K)
    #     * X: state variable - shape: NxNf / p: projection vector - shape: Nfx1 / y: projected field -shape: Nx1
    #     * Multiply p to each point to get y -> sort by descending order to truncate by Kx1 : sampling points
    #     '''
    #     N_ = jnp.shape(self.V)[0]
    #     field = jnp.matmul(self.V, ytilde).reshape(N_,1)
    #
    #     # Numer of sampling point
    #     #k_sampling=int(N_/self.sampling_factor)
    #
    #     #calculate characteristic variable field using MLP
    #     #print(field.shape)
    #     #y=self.model.apply(params, field[:,0])
    #     # k_index=self.model.apply(params, field[:,0])
    #     # print(k_index.shape)
    #
    #     key = jax.random.PRNGKey(seed=42)
    #     P_ML = self.model.apply(params, field[:,0], training=True, rngs={'dropout': key})
    #     # jax.debug.print("{}, {}",jnp.max(P_ML), jnp.min(P_ML))
    #
    #     k_index = jnp.argmax(P_ML, axis=0)
    #
    #     if train == False:
    #         result = jnp.zeros_like(P_ML)
    #         max_indices = jnp.argmax(P_ML, axis=0)
    #         result = result.at[max_indices, jnp.arange(P_ML.shape[1])].set(1)
    #         P_ML = result
    #         k_index = jnp.argmax(P_ML, axis=0)
    #         # jax.debug.print("{}, {}", jnp.max(P_ML), jnp.min(P_ML))
    #         # P_ML = self.P
    #         # k_index = jnp.argmax(P_ML, axis=0)
    #         # print(k_index)
    #
    #     # jax.debug.print("{}, {}",jnp.max(k_index), jnp.min(k_index))
    #
    #     # P_ML=self.P
    #
    #     # np.save('P_ML.npy',P_ML)
    #     #exit()
    #
    #
    #     #print(jnp.average(P_ML, axis=0))
    #
    #     #jax.debug.print("{}",P_ML[:,0])
    #     #P_ML = jax.where()
    #
    #     # print(k_index)
    #     #y_minus=jnp.array(-y, dtype=np.float64)
    #     #print(y)
    #     # print(k_index.shape)
    #     # print(k_sampling)
    #
    #     #sorting y by descending order to truncate by Kx1 & making k_index vector - shape: Kx1
    #     #k_index=jnp.argsort(y_minus)[:k_sampling]
    #
    #     #build P matrix using k_index
    #     # P_ML=jnp.ones((N_, k_sampling)) * k_index[0]
    #
    #     # for j in range(k_sampling):
    #     #     i=k_index[j]
    #     #     #self.P[i][j]=1
    #     #     P_ML=P_ML.at[i,j].set(1)
    #
    #     # print(P_ML.shape)
    #
    #     #P_ML=jnp.ones((N_, k_sampling)) *y[0]
    #     # P_ML=jnp.ones((N_, k_sampling)) *y[k_index][3]
    #     # P_ML=jnp.ones((N_, k_sampling)) * k_index[1]
    #     #print(P_ML)
    #
    #     mtemp = jnp.linalg.pinv(jnp.matmul(jnp.transpose(P_ML),self.U)) #ok MxM
    #     mtemp = jnp.matmul(self.U,mtemp) # NxM
    #     mtemp = jnp.matmul(mtemp,jnp.transpose(P_ML)) # NxN
    #
    #     varP = jnp.matmul(jnp.transpose(self.V),mtemp) # KxN
    #
    #     return k_index, varP
          

    def nonlinear_operator_pod_deim(self, ytilde, varP):
        '''
        Defines the nonlinear reconstruction using the POD-DEIM technique
        '''
        N_ = jnp.shape(self.V)[0]
        field = jnp.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return jnp.matmul(varP,nonlinear_field_approx)

    
    def pod_deim_rhs_MLsampling(self,state, linear_matrix, varP):
        '''
        Calculate the rhs of the POD GP implementation using ML-sampling
        '''
        linear_term = self.linear_operator(state, linear_matrix)
        non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    
        return jnp.add(linear_term,-non_linear_term)

    def pod_deim_ML_evolve(self, params, ytilde_init, train, num_steps):
        '''
        Use RK3 to do a system evolution for pod_deim
        '''
        state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0], num_steps), dtype='double')
        state_tracker = state_tracker.at[:, 0].set(ytilde_init)
        nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0], num_steps), dtype='double')
        k_index_tracker = jnp.zeros(shape=(int(self.sampling_factor), num_steps), dtype='int')

        total_l1_penalty = 0.0

        linear_matrix = self.linear_operator_fixed()
        state = state_tracker[:, 0]

        k_index, varP, l1_penalty = self.deim_matrices_MLsampling(state, params, train)
        total_l1_penalty += l1_penalty

        non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
        nl_state_tracker = nl_state_tracker.at[:, 0].set(non_linear_term[:, 0])
        k_index_tracker = k_index_tracker.at[:, 0].set(k_index)

        for t in range(1, num_steps):
            # Strong Stability Preserving Runge-Kutta
            rhs = self.pod_deim_rhs_MLsampling(state, linear_matrix, varP)
            l1 = state + dt * rhs[:, 0]

            k_index, varP, l1_penalty = self.deim_matrices_MLsampling(l1, params, train)
            total_l1_penalty += l1_penalty

            rhs = self.pod_deim_rhs_MLsampling(l1, linear_matrix, varP)
            l2 = 0.75 * state + 0.25 * l1 + 0.25 * dt * rhs[:, 0]

            k_index, varP, l1_penalty = self.deim_matrices_MLsampling(l2, params, train)
            total_l1_penalty += l1_penalty

            rhs = self.pod_deim_rhs_MLsampling(l2, linear_matrix, varP)
            state = 1.0 / 3.0 * state[:] + 2.0 / 3.0 * l2[:] + 2.0 / 3.0 * dt * rhs[:, 0]

            state_tracker = state_tracker.at[:, t].set(state[:])

            k_index, varP, l1_penalty = self.deim_matrices_MLsampling(state, params, train)
            total_l1_penalty += l1_penalty

            non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
            nl_state_tracker = nl_state_tracker.at[:, t].set(non_linear_term[:, 0])
            k_index_tracker = k_index_tracker.at[:, t].set(k_index)

        avg_l1_penalty = total_l1_penalty / (num_steps * 4)

        return state_tracker, k_index_tracker, avg_l1_penalty
        

    # def pod_deim_ML_evolve(self, params, ytilde_init, train, num_steps):
    #     '''
    #     Use RK3 to do a system evolution for pod_deim
    #     '''
    #     # self.train = train
    #     # if num_steps is not None:
    #     #     self.num_steps = num_steps
    #
    #     state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps),dtype='double')
    #     state_tracker = state_tracker.at[:, 0].set(ytilde_init)
    #     nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],num_steps),dtype='double')
    #     k_index_tracker = jnp.zeros(shape=(int(self.sampling_factor),num_steps),dtype='int')
    #
    #
    #
    #     # Setup fixed operations
    #     linear_matrix = self.linear_operator_fixed()
    #     state = state_tracker[:,0]
    #
    #     k_index, varP = self.deim_matrices_MLsampling(state, params, train)
    #     # k_index, varP = self.deim_matrices_precompute()
    #     # Recording the nonlinear term
    #     non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    #     nl_state_tracker=nl_state_tracker.at[:,0].set(non_linear_term[:,0])
    #     k_index_tracker=k_index_tracker.at[:,0].set(k_index)
    #
    #
    #     for t in range(1,num_steps):
    #         #Strong Stability Preserving Runge-Kutta
    #         rhs = self.pod_deim_rhs_MLsampling(state, linear_matrix, varP)
    #         l1 = state + dt*rhs[:,0]
    #
    #         k_index, varP = self.deim_matrices_MLsampling(l1, params, train)
    #         rhs = self.pod_deim_rhs_MLsampling(l1, linear_matrix, varP)
    #         l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]
    #
    #         k_index, varP = self.deim_matrices_MLsampling(l2, params, train)
    #         rhs = self.pod_deim_rhs_MLsampling(l2, linear_matrix, varP)
    #         #state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
    #         state = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
    #
    #         state_tracker=state_tracker.at[:,t].set(state[:])
    #
    #
    #         # Recording the nonlinear term
    #         k_index, varP = self.deim_matrices_MLsampling(state, params, train)
    #         #print(k_index)
    #
    #         non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    #         nl_state_tracker=nl_state_tracker.at[:,t].set(non_linear_term[:,0])
    #
    #         k_index_tracker=k_index_tracker.at[:,t].set(k_index)
    #
    #
    #     return state_tracker, k_index_tracker
    
    
    
    # def pod_deim_ML_evolve_fixed(self, params, ytilde_init):
    #     '''
    #     Use RK3 to do a system evolution for pod_deim
    #     '''
        
    #     state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps),dtype='double')
    #     state_tracker = state_tracker.at[:, 0].set(ytilde_init)
    #     nl_state_tracker = jnp.zeros(shape=(jnp.shape(ytilde_init)[0],self.num_steps),dtype='double')
    #     k_index_tracker = jnp.zeros(shape=(int(self.sampling_factor),self.num_steps),dtype='int')
    #     #k_index_tracker = jnp.zeros(shape=(128,self.num_steps),dtype='int')
        

    #     # Setup fixed operations
    #     linear_matrix = self.linear_operator_fixed()
    #     self.deim_matrices_precompute()
    #     state = state_tracker[:,0]
        
    #     k_index, varP = self.deim_matrices_MLsampling(state, params)
    #     # Recording the nonlinear term
    #     non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    #     #self.nl_state_tracker[:,0] = non_linear_term[:,0]
    #     nl_state_tracker=nl_state_tracker.at[:,0].set(non_linear_term[:,0])
    #     k_index_tracker=k_index_tracker.at[:,0].set(k_index)
        
        
    #     for t in range(1,self.num_steps+1):
    #         rhs = self.pod_deim_rhs_MLsampling(state, linear_matrix, varP)
    #         l1 = state + dt*rhs[:,0]

    #         k_index, varP = self.deim_matrices_MLsampling(l1, params)
    #         rhs = self.pod_deim_rhs_MLsampling(l1, linear_matrix, varP)
    #         l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

    #         k_index, varP = self.deim_matrices_MLsampling(l2, params)
    #         rhs = self.pod_deim_rhs_MLsampling(l2, linear_matrix, varP)
    #         #state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
    #         state = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

    #         state_tracker=state_tracker.at[:,t].set(state[:])
            

    #         # Recording the nonlinear term
    #         k_index, varP = self.deim_matrices_MLsampling(state, params)
    #         #print(k_index)
            
    #         non_linear_term = self.nonlinear_operator_pod_deim(state, varP)
    #         nl_state_tracker=nl_state_tracker.at[:,t].set(non_linear_term[:,0])
            
    #         k_index_tracker=k_index_tracker.at[:,t].set(k_index)
            
    #     return state_tracker, k_index_tracker
            
            


class gp_evolution():
    '''
    Does a POD/POD-DEIM GP evolution
    '''
    def __init__(self,V,U,P,ytilde_init):
        '''
        ytilde - initial condition in POD space - shape: Kx1
        V - Truncated POD bases (precomputed) - shape: NxK
        V - Truncated POD bases for nonlinear snapshots (precomputed) - shape: NxM
        P - DEIM matrix - shape: NxM
        '''
        self.V = V
        self.U = U
        self.P = P
        self.num_steps = int(np.shape(tsteps)[0]-1)
        self.state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')
        self.state_tracker[:,0] = ytilde_init[:,0]
        self.nl_state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')

    def linear_operator_fixed(self):
        '''
        This method fixes the laplacian based linear operator
        '''
        N_ = np.shape(self.V)[0]
        self.laplacian_matrix = -2.0*np.identity(N_)

        # Setting upper diagonal
        i_range = np.arange(0,N_-1,dtype='int')
        j_range = np.arange(1,N_,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Setting lower diagonal
        i_range = np.arange(1,N_,dtype='int')
        j_range = np.arange(0,N_-1,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Periodicity
        self.laplacian_matrix[0,N_-1] = 1.0
        self.laplacian_matrix[N_-1,0] = 1.0
        self.laplacian_matrix = 1.0/(Rnum*dx*dx)*self.laplacian_matrix
        
        # Final Linear operator
        self.linear_matrix = np.matmul(np.matmul(np.transpose(self.V),self.laplacian_matrix),self.V)

    def linear_operator(self,Ytilde):
        '''
        Calculates the linear term on the RHS
        '''
        M_ = np.shape(self.V)[1]
        return np.matmul(self.linear_matrix,Ytilde).reshape(M_,1)

    def nonlinear_operator_pod(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the standard POD-GP technique
        '''
        N_ = np.shape(self.V)[0]
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return np.matmul(np.transpose(self.V),nonlinear_field_approx)

    def pod_gp_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod(state)
    
        return np.add(linear_term,-non_linear_term)

    def pod_gp_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_gp
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        state = np.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod(state)
        self.nl_state_tracker[:,0] = non_linear_term[:,0]

        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_gp_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l2)
            state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            self.state_tracker[:,t] = state[:]

            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod(state)
            self.nl_state_tracker[:,t] = non_linear_term[:,0]

    def deim_matrices_precompute(self):
        '''
        Precompute the DEIM matrices for nonlinear term in ODE
        '''
        mtemp = np.linalg.pinv(np.matmul(np.transpose(self.P),self.U)) #ok MxM
        mtemp = np.matmul(self.U,mtemp) # NxM
        mtemp = np.matmul(mtemp,np.transpose(self.P)) # NxN
        
        self.varP = np.matmul(np.transpose(self.V),mtemp) # KxN

    def nonlinear_operator_pod_deim(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the POD-DEIM technique
        '''
        N_ = np.shape(self.V)[0]
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return np.matmul(self.varP,nonlinear_field_approx)

    def pod_deim_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod_deim(state)
    
        return np.add(linear_term,-non_linear_term)

    def pod_deim_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_deim
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        self.deim_matrices_precompute()
        state = np.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod_deim(state)
        self.nl_state_tracker[:,0] = non_linear_term[:,0]
        
        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_deim_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l2)
            state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            self.state_tracker[:,t] = state[:]

            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod_deim(state)
            self.nl_state_tracker[:,t] = non_linear_term[:,0]





            
            
        
        



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
    spatial_resolution : 3072
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(features = 1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = nn.Dense(self.spatial_resolution)(x)
        x = x.reshape((128, self.spatial_resolution//128))
        # x = x/0.1
        x = nn.softmax(x, axis=0)
        return x


# class MLP2(nn.Module):
#     spatial_resolution: 3072
#     dropout_rate: float = 0.2
#
#     @nn.compact
#     def __call__(self, x, training: bool = True):
#         x = nn.Dense(1024)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(1024)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(2048)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         # x = nn.Dense(4096)(x)
#         # x = nn.LayerNorm()(x)
#         # x = nn.relu(x)
#         # x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(self.spatial_resolution)(x)
#         x = x.reshape((128, self.spatial_resolution//128))
#         # x = x/0.025
#         # x = nn.softmax(x, axis=0)
#         return x


class MLP2(nn.Module):
    spatial_resolution: 3072
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(2048)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        skip = x
        x = nn.Dense(2048)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x + skip
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.spatial_resolution)(x)
        x = x.reshape((128, self.spatial_resolution // 128))

        x = nn.log_softmax(x, axis=0)
        return x


# class MLP2(nn.Module):
#     spatial_resolution: 3072
#     dropout_rate: float = 0.2
#
#     @nn.compact
#     def __call__(self, x, training: bool = True):
#         x = nn.Dense(1024)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         skip = x
#         x = nn.Dense(1024)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = x + skip
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(2048)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         skip = x
#         x = nn.Dense(2048)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = x + skip
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(4096)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
#
#         x = nn.Dense(self.spatial_resolution)(x)
#         x = x.reshape((128, self.spatial_resolution // 128))
#
#         x = nn.log_softmax(x, axis=0)
#         return x



    

