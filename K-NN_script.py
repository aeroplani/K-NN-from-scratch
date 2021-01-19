import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
###########################################################
#load dataset
###########################################################


def load_batch(filename):
    """Loads a data batch
    """
    with open(filename, 'rb') as f:
        dicti = pickle.load(f, encoding='bytes')
    return dicti

def load_batch_and_arrange(filename):
    dicti=load_batch(filename)
    X = (dicti[b"data"]/ 255).T
    y = dicti[b"labels"]
    Y = (np.eye(10)[y]).T
    return X, Y, y
    

def standardize(scalerr, data):
    return scalerr.transform(data.T).T

def get_prepped_data(all_batch, v_n, reduced_d = False, d = 10):
    
    if all_batch:
        print("in all")
        X_train1, Y_train1, y_train1 =  load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_1")
        X_train2, Y_train2, y_train2 =  load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_2")
        X_train3, Y_train3, y_train3 =  load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_3")
        X_train4, Y_train4, y_train4 =  load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_4")
        X_train5, Y_train5, y_train5 =  load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_5")

        X_tr = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5),
                axis=1)
        Y_tr = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5),
                axis=1)
        y_tr = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
        
        if not reduced_d:
            X_v = X_tr[:, -v_n:]
            Y_v = Y_tr[:, -v_n:]
            y_v = y_tr[-v_n:]
            X_tr = X_tr[:, :-v_n]
            Y_tr = Y_tr[:, :-v_n]
            y_tr = y_tr[:-v_n]
            X_te, Y_te, y_te =  load_batch_and_arrange("datasets/cifar-10-batches-py/test_batch")
            
        if reduced_d:
            X_v = X_tr[:d, -v_n:]
            Y_v = Y_tr[:d, -v_n:]
            y_v = y_tr[-v_n:]
            X_tr = X_tr[:d, :-v_n]
            Y_tr = Y_tr[:d, :-v_n]
            y_tr = y_tr[:-v_n]
            X_te, Y_te, y_te =  load_batch_and_arrange("datasets/cifar-10-batches-py/test_batch")
            X_te = X_te[:d,:]
        
    else:
        print("in one")
        X_tr, Y_tr, y_tr = load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_1")
        X_v, Y_v, y_v = load_batch_and_arrange("datasets/cifar-10-batches-py/data_batch_2")
        X_te, Y_te, y_te = load_batch_and_arrange("datasets/cifar-10-batches-py/test_batch")
    
     
    #standardizing:
    scaler=StandardScaler().fit(X_tr.T)
    
    X_tr = standardize(scaler,X_tr)
    X_v = standardize(scaler,X_v)
    X_te = standardize(scaler,X_te)
    
    values = [X_tr, Y_tr, y_tr,X_v, Y_v, y_v,X_te, Y_te, y_te]
    keys = ["X_tr", "Y_tr", "y_tr","X_v", "Y_v", "y_v","X_te", "Y_t", "y_te"]
    dicti_data=dict(zip(keys, values))

    return dicti_data

###########################################################
#Helper fun
###########################################################

def relu(x):
        x[x<0] = 0
        return x

def softmaxa(x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=0) 
    
def b_normalize(s, mu, var):
        eps = np.finfo(np.float64).eps
        return (s - mu) / np.sqrt(var + eps)
    
def run_av(x_av, x, alpha):
    return alpha * x_av + (1-alpha) * x

###########################################################
#classifier class
###########################################################

class Classifier():
    def __init__(self, data, dim_list, b_norm):
        
        self.b_norm = b_norm
        self.alpha = 0.9
        self.N = data['X_tr'].shape[1]
        for i, j in data.items():
            setattr(self, i, j)
           
        self.initializeWb(dim_list)
        self.N_layers = len(self.W_list)
        
    def initializeWb(self, dim_list):
        mu = 0
        #sigma = 1e-3
        self.W_list = []
        self.b_list = []
       
        for i in range(len(dim_list) - 1):
            sigma = np.sqrt(2/ dim_list[i])
            self.W_list.append(np.random.normal(mu, (sigma)**2, (dim_list[i + 1], dim_list[i])))
            self.b_list.append(np.ones((dim_list[i + 1], 1)))
        
        if not self.b_norm:
            return self.W_list, self.b_list
        
        self.gamma_list = []
        self.beta_list = []
        self.mu_list = []
        self.var_list = []
        if self.b_norm:
            for i in range(len(dim_list) - 1):
                self.gamma_list.append(np.ones((dim_list[i + 1], 1)))
                self.beta_list.append(np.zeros((dim_list[i + 1], 1)))
                self.mu_list.append(np.zeros((dim_list[i + 1], 1)))
                self.var_list.append(np.zeros((dim_list[i + 1], 1)))
            return self.W_list, self.b_list, self.gamma_list, self.beta_list, self.mu_list, self.var_list               
            
    def forward_pass(self, X, train=False):
        x = X
        S, S_hat, mus, varss, X_l = [], [], [], [], []
        if self.b_norm:
            X_l.append(x)
            for i in range(self.N_layers):   
                s = np.matmul(self.W_list[i],x) + self.b_list[i]
                if i < self.N_layers-1:
                    if not train:
                        s_hat = b_normalize(s,self.mu_list[i],self.var_list[i])
                    if train:
                        mu = np.mean(s, axis=1, keepdims=True)
                        var = np.var(s, axis=1, keepdims=True)
                        mus.append(mu)
                        varss.append(var)
                        self.mu_list[i]  = run_av(self.mu_list[i], mu, self.alpha)
                        self.var_list[i] = run_av(self.var_list[i], var, self.alpha)
                        s_hat = b_normalize(s,mu,var)
                    x = relu(np.multiply(self.gamma_list[i], s_hat) + self.beta_list[i])
                    
                    #save to lists
                    X_l.append(x)
                    S.append(s)
                    S_hat.append(s_hat)
                else:
                    s = np.matmul(self.W_list[i],x) + self.b_list[i]
                    P = softmaxa(s)
        if not self.b_norm:
            for i in range(self.N_layers-1):
                s = relu(np.matmul(self.W_list[i],s) + self.b_list[i])
                X_l.append(s)
            P = softmaxa( np.matmul( self.W_list[-1], s) + self.b_list[-1])
        return X_l, P, S, S_hat, mus, varss

    def compute_cost(self, X, Y, lam, train=True):
        N = X.shape[1]
        P = self.forward_pass(X, train=train)[1]
        loss = -np.sum(Y*np.log(P))/N
        reg = sum(map(lambda W: np.sum(np.square(W)), self.W_list))
        cost = loss + lam * reg
        return loss, cost
    
    def compute_accuracy(self, X, y, train=True):
        N = X.shape[1]
        p = self.forward_pass(X, train=train)[1]
        pred = np.argmax(p,axis=0)
        acc = 0
        for i in range(N):
            if(pred[i] == y[i]):
                acc += 1
        return acc /N
 
    def batch_norm_back_pass(self, G_b, S_b, mu_b, v_b):
        eps = np.finfo(np.float32).eps
        
        N = G_b.shape[1]
        sig_1 = np.power(v_b + eps, -0.5)
        sig_2 = np.power(v_b + eps, -1.5)
        
        G_1 = np.multiply(G_b, sig_1)
        G_2 = np.multiply(G_b, sig_2)

        G_b = G_1 - np.sum(G_1, axis=1, keepdims=True)/N - np.multiply(( S_b - mu_b), ( np.sum(np.multiply(G_2, ( S_b - mu_b)), axis=1, keepdims=True)))/N
        return G_b
    
    def compute_gradients(self, X_b, Y_b, lam):
        grads_W, grads_b, grads_gamma, grads_beta = [], [], [], []
        N = X_b.shape[1]
        one_matrix = np.ones(N)
        for i in range(self.N_layers):
                grads_W.append(np.zeros_like(self.W_list[i]))
                grads_b.append(np.zeros_like(self.b_list[i]))
                grads_gamma.append(np.zeros_like(self.gamma_list[i]))
                grads_beta.append(np.zeros_like(self.beta_list[i]))

        X_l, P_b, S_b, S_hat_b, mus_b, vars_b = self.forward_pass(X_b, train=True)
        G_b = - (Y_b - P_b)
        
        grads_W[-1] = np.matmul(G_b,X_l[-1].T)/N + 2 * lam * self.W_list[-1]
        grads_b[-1] = np.reshape(np.matmul(G_b,one_matrix)/N, (self.b_list[-1].shape))
        
        
        G_b = np.dot(self.W_list[-1].T,G_b)
        X_l_ind = np.where(X_l[-1] > 0, 1, 0)
        G_b = np.multiply(G_b, X_l_ind  > 0)
    
        if self.b_norm:
            for l in range(self.N_layers - 2, -1, -1):
                grads_gamma[l] = np.reshape(np.matmul(np.multiply(G_b, S_hat_b[l]),one_matrix)/N, (grads_gamma[l].shape))
                grads_beta[l]  = np.reshape(np.matmul(G_b,one_matrix)/N, (grads_beta[l].shape))

                G_b = np.multiply(G_b, self.gamma_list[l])
                G_b = self.batch_norm_back_pass(G_b, S_b[l], mus_b[l], vars_b[l])

                grads_W[l] = np.matmul(G_b,X_l[l].T)/N + 2 * lam * self.W_list[l]
                grads_b[l] = np.reshape(np.matmul(G_b,one_matrix)/N, (grads_b[l].shape))
                if l > 0:
                    G_b = np.dot(self.W_list[l].T,G_b)
                    X_l_ind = np.where(X_l[l] > 0, 1, 0)
                    G_b = np.multiply(G_b, X_l_ind  > 0)
                    
            return grads_W, grads_b, grads_gamma, grads_beta

        if not self.b_norm:
            for l in range(self.N_layers - 2, -1, -1): 
                grads_W[l] = 1/N * np.matmul(G_b,X_l[l-1].T) + 2 * lam * self.W_list[l]
                grads_b[l] = np.reshape(np.matmul(G_b, one_matrix)/N, (self.b_list[l].shape))         
            return grads_W, grads_b

    def compute_gradients_num(self, X, Y, lam, h):
        grad_W = []
        grad_b = []
        grad_gamma = []
        grad_beta = []
        
        N = X.shape[1]
        # Initialization of grad
        for i in range(len(self.W_list)):
            grad_W.append(np.zeros((self.W_list[i].shape[0],self.W_list[i].shape[1])))
            grad_b.append(np.zeros((self.b_list[i].shape[0],self.b_list[i].shape[1])))
            grad_gamma.append(np.zeros((self.gamma_list[i].shape[0],self.gamma_list[i].shape[1])))
            grad_beta.append(np.zeros((self.beta_list[i].shape[0],self.beta_list[i].shape[1])))
                    
        for index in range(len(self.W_list)):
            for i in range(self.W_list[index].shape[0]):
                for j in range(self.W_list[index].shape[1]):
                    W0 = self.W_list[index][i][j]
                    self.W_list[index][i][j] += h
                    c1 = self.compute_cost(X, Y, lam)[1]
                    self.W_list[index][i][j] -=  (2*h)
                    c2 = self.compute_cost(X, Y, lam)[1]
                    self.W_list[index][i][j] = W0
                    grad_W[index][i][j] = (c1 - c2) / (2*h)
                    
            for i in range(len(self.b_list[index])):
                b0 = self.b_list[index][i]
                self.b_list[index][i] += h
                c1 = self.compute_cost(X, Y, lam)[1]
                self.b_list[index][i] -= (2*h)
                c2 = self.compute_cost(X, Y, lam)[1]
                self.b_list[index][i] = b0
                grad_b[index][i] = (c1 - c2) / (2*h)
                        
            for i in range(len(self.gamma_list[index])):
                gamma0 = self.gamma_list[index][i]
                self.gamma_list[index][i] = gamma0 + h
                c1 = self.compute_cost(X, Y, lam)[1]
                self.gamma_list[index][i] = gamma0 - 2*h
                c2 = self.compute_cost(X, Y, lam)[1]
                grad_gamma[index][i] = (c1 - c2) / (2*h)
                self.gamma_list[index][i] = gamma0
                
            for i in range(len(self.beta_list[index])):
                beta0 = self.beta_list[index][i]
                self.beta_list[index][i] = beta0 + h
                c1 = self.compute_cost(X, Y, lam)[1]
                self.beta_list[index][i] = beta0 - 2*h
                c2 = self.compute_cost(X, Y, lam)[1]
                grad_beta[index][i] = (c1 - c2) / (2*h)
                self.beta_list[index][i] = beta0
        return grad_W, grad_b, grad_gamma, grad_beta
    
    def plot_performance(self, n_epochs, train, val, title,y_label):
       epochs = np.arange(n_epochs)

       plt.plot(epochs, train, label="Training")
       plt.plot(epochs, val, label="Validation")
       plt.legend()
       plt.xlabel('Epoch')
       plt.ylabel(y_label)
       plt.grid()
       plt.savefig("plots/" + title + ".png")
       plt.close()
       
    def get_learning_rate(self,t,eta_max, eta_min, n_s):
        if t <= n_s:
            eta = eta_min + t/n_s * (eta_max - eta_min)
        elif n_s < t <= 2*n_s:
            eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min)
        
        if t == 2*n_s:
            print('new cycle')
            t=0
        t=t+1
        return eta,t

    def mini_batch_gd(self, X, Y, lam, batch_s, eta_min,eta_max, n_s, n_epochs):
        costs_train, loss_train, acc_train, costs_val, loss_val, acc_val = np.zeros((6,n_epochs))
        
        batch_amount = np.floor(X.shape[1] / batch_s)
        eta = eta_min
        
        t = 0
        k=2
        n_s = 5 * 45000 / batch_s
        #n_s = 800
        print(n_s)
        reps=1
        for m in range(reps):
            for n in range(n_epochs):
                for batch in range(int(batch_amount)):
                    
                    batch_size = int(self.N / batch_amount)
                    j_start = (batch) * batch_size 
                    j_end = (batch+1) * batch_size 
                    
                    X_b = X[:, j_start:j_end]
                    Y_b = Y[:, j_start:j_end]
                    
                    if not self.b_norm:
                        grad_W, grad_b = self.compute_gradients(
                            X_b, Y_b, lam)
                        
                        for i in range (self.N_layers):
                            self.W_list[i] -= eta * grad_W[i]
                            self.b_list[i] -= eta * grad_b[i]
                    
                    if self.b_norm:
                        grad_W, grad_b, grad_gamma, grad_beta = self.compute_gradients(
                            X_b, Y_b, lam)
                        
                        for i in range (self.N_layers):
                            self.W_list[i] -= eta * grad_W[i]
                            self.b_list[i] -= eta * grad_b[i]
                            self.gamma_list[i] -= eta * grad_gamma[i]
                            self.beta_list[i] -= eta * grad_beta[i]
                        
                    #learning rate
                    eta,t = self.get_learning_rate(t,eta_max, eta_min, n_s)
                
                    
                #get loss, cost and accuracy    
                loss_train[n], costs_train[n] = self.compute_cost(X, Y, lam)
                loss_val[n], costs_val[n] = self.compute_cost(self.X_v, self.Y_v, lam)
                acc_train[n] = self.compute_accuracy(self.X_tr, self.y_tr)
                acc_val[n] = self.compute_accuracy(self.X_v, self.y_v)
                #print(acc_train[n])
                
        return costs_train, costs_val, loss_train, loss_val, acc_train, acc_val
    
def check_grad(g1,g2):
    h = 1e-7
    return np.linalg.norm(g1 - g2,ord=2) / max(h ,np.linalg.norm(g1 ,ord=2) + np.linalg.norm(g2 ,ord=2))
        
        
###########################################################
#main
###########################################################
        
if __name__ == '__main__':
    task = 3
    labels =np.array([['airplane','automobile' ,'bird' ,'cat' ,'deer'] ,['dog' ,'frog' ,'horse', 'ship', 'truck']])
    
    if task == 2: 
        #data
        all_batch = True
        v_n = 50000 - 50
        
        #settings
        N = 10000
        K = 10
        d = 50
        m = 50
        
        #d =13
        dim_list = [d, 50,50, K]
        b_norm = True
        reduced_d = True
        
        data = get_prepped_data(all_batch, v_n, reduced_d, d)
        m1 = Classifier(data, dim_list, b_norm = True)
                
        X = data['X_tr']
        Y = data['Y_tr']
        print(X.shape)

        h = 1e-7
        lam = 0
        
        grad_W, grad_b, grad_gamma, grad_beta= m1.compute_gradients(X, Y, lam)
        grad_W1, grad_b1, grad_gamma1, grad_beta1 = m1.compute_gradients_num(X, Y, lam, h)
        
        for i in range (len(dim_list) - 1):
            r1=check_grad(grad_b[i],grad_b1[i])
            r2=check_grad(grad_W[i],grad_W1[i])
            r3=check_grad(grad_gamma[i],grad_gamma1[i])
            r4=check_grad(grad_beta[i],grad_beta1[i])
            print()
            print(r1, r2, r3, r4)
        
    if task == 3:
        #settings
        all_batch = True
        v_n = 5000 
        n_epochs= 20
        plot="9L_b"
        #lam=0.005
        batch_s=100
        eta_min=1e-5 
        eta_max=1e-1 
        n_s = 800 
        K = 10
        d = 3072
        dim_list = [d, 50, 50, K]
        b_norm = True
            #start
        data = get_prepped_data(all_batch, v_n)
        
            #lambda searches
        #lam_list = np.random.uniform(0.001,0.02, (10));
        lam_list= [0.008]
        for lam in lam_list:
            print(round(lam,3))
        
            #best lambda
        
        for lam in lam_list:
            m1 = Classifier(data, dim_list, b_norm)
            costs_train, costs_val, loss_train, loss_val, acc_train, acc_val = m1.mini_batch_gd(data['X_tr'], data['Y_tr'], lam,batch_s, eta_min, eta_max, n_s, n_epochs)
            
            
            X = data['X_tr']
            #print(X.shape)
            
                #plot cost, loss and accuracy
            print('plotting')
            # m1.plot_performance(n_epochs, costs_train, costs_val, plot + "_cost_plot", y_label="Cost")
            m1.plot_performance(n_epochs, loss_train, loss_val, plot+ "_loss_plot", y_label="Loss")
            # m1.plot_performance(n_epochs, acc_train, acc_val, plot + "_acc_plot", y_label="Accuracy")
    
            #get accuracies
            accu_train = m1.compute_accuracy(m1.X_tr, m1.y_tr)
            accu_val = m1.compute_accuracy(m1.X_v, m1.y_v)
            accu_test = m1.compute_accuracy(m1.X_te, m1.y_te)
            
            print(round(accu_test,3))
        

    
    