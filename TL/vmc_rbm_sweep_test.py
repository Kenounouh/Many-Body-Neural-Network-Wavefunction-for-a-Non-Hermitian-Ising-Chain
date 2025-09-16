#Author: Lavoisier Wah & Remmy Zen
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Enable XLA compilation
tf.config.optimizer.set_jit(True)

# For reproducibility purposes
tf.random.set_seed(111)
np.random.seed(111)
random.seed(111)

class VariationalMonteCarloRBM(tf.keras.Model):
    def __init__(self, N, Jz, g_real, g_imag, num_hidden, learning_rate, seed=1234,
                W_array=None, bv_array=None, bh_array=None):
        super(VariationalMonteCarloRBM, self).__init__()
        
        # Parameters
        self.N = N
        self.Jz = tf.cast(Jz, tf.float32)
        self.g_real = tf.cast(g_real, tf.float32)
        self.g_imag = tf.cast(g_imag, tf.float32)
        self.num_hidden = num_hidden
        self.seed = seed
        
        # Define A (even sites) and B (odd sites)
        self.sites_A = tf.constant(range(0, N, 2), dtype=tf.int32)
        self.sites_B = tf.constant(range(1, N, 2), dtype=tf.int32)
        
        # Set seed and optimizer
        tf.random.set_seed(self.seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)

        # Initialize weights and biases
        if W_array is None:
            self.W_array = np.random.uniform(size=(self.N, self.num_hidden), low=0.0, high=0.01)
        else:
            self.W_array = W_array
            
        if bv_array is None:
            self.bv_array = np.zeros((1, self.N))
        else:
            self.bv_array = bv_array
            
        if bh_array is None:
            self.bh_array = np.zeros((1, self.num_hidden))
        else:
            self.bh_array = bh_array

        self.W = tf.Variable(tf.convert_to_tensor(self.W_array, dtype=tf.float32), name="weights")
        self.bv = tf.Variable(tf.convert_to_tensor(self.bv_array, dtype=tf.float32), name="visible_bias")
        self.bh = tf.Variable(tf.convert_to_tensor(self.bh_array, dtype=tf.float32), name="hidden_bias")

    @tf.function(jit_compile=True)
    def sample(self, nsamples, init_data=None):
        if init_data is None:
            init_data = tf.cast(
                tf.where(
                    tf.random.uniform((nsamples, self.N), 0, 2, dtype=tf.int32) == 0,
                    -1, 1
                ),
                tf.float32
            )

        for _ in tf.range(10):
            init_data = self.get_new_visible(init_data)
            
        return init_data, self.logpsi(init_data)

    @tf.function(jit_compile=True)
    def get_new_visible(self, v):
        hprob = self.get_hidden_prob_given_visible(v)
        hstate = self.convert_from_prob_to_state(hprob)
        vprob = self.get_visible_prob_given_hidden(hstate)
        vstate = self.convert_from_prob_to_state(vprob)
        return vstate

    @tf.function(jit_compile=True)
    def get_hidden_prob_given_visible(self, v):
        return tf.sigmoid(2.0 * (tf.matmul(v, self.W) + self.bh))

    @tf.function(jit_compile=True)
    def get_visible_prob_given_hidden(self, h):
        return tf.sigmoid(2.0 * (tf.matmul(h, tf.transpose(self.W)) + self.bv))

    @tf.function(jit_compile=True)
    def convert_from_prob_to_state(self, prob):
        v = prob - tf.random.uniform(tf.shape(prob), 0, 1)
        return tf.where(tf.greater_equal(v, tf.zeros_like(v)), tf.ones_like(v), -1 * tf.ones_like(v))

    @tf.function(jit_compile=True)
    def logpsi(self, samples):
        theta = tf.matmul(samples, self.W) + self.bh
        sum_ln_thetas = tf.reduce_sum(tf.math.log(tf.cosh(theta)), axis=1, keepdims=True)
        ln_bias = tf.matmul(samples, tf.transpose(self.bv))
        return tf.squeeze(0.5 * (sum_ln_thetas + ln_bias))

    @tf.function(jit_compile=True)
    def localenergy(self, samples, logpsi):
        samples = tf.cast(samples, tf.float32)
        batch_size = tf.shape(samples)[0]
        
        eloc_real = tf.zeros(batch_size, dtype=tf.float32)
        eloc_imag = tf.zeros(batch_size, dtype=tf.float32)
        
        # Classical interaction term
        for n in tf.range(self.N):
            eloc_real += -self.Jz * samples[:, n] * samples[:, (n + 1) % self.N]

        # Off-diagonal contributions
        for j in tf.range(self.N):
            flip_samples = tf.tensor_scatter_nd_update(
                samples,
                tf.stack([tf.range(batch_size), tf.fill([batch_size], j)], axis=1),
                -samples[:, j]
            )
            flip_logpsi = self.logpsi(flip_samples)
            exp_term = tf.exp(flip_logpsi - logpsi)
            
            eloc_real -= self.g_real * exp_term
            if j % 2 == 0:
                eloc_imag -= self.g_imag * exp_term
            else:
                eloc_imag += self.g_imag * exp_term

        return eloc_real, eloc_imag

    @tf.function(jit_compile=True)
    def magnetization(self, samples):
        spins = tf.cast(samples, tf.float32)
        magnetization = tf.reduce_mean(spins, axis=1)
        return tf.abs(magnetization)

    @tf.function(jit_compile=True)
    def training_step(self, samples, alpha):
        with tf.GradientTape() as tape:
            logpsi = self.logpsi(samples)
            eloc_real, eloc_imag = self.localenergy(samples, logpsi)
            
            Eo_real = tf.reduce_mean(eloc_real)
            Eo_imag = tf.reduce_mean(eloc_imag)

            loss_real = tf.reduce_mean(
                2.0 * tf.multiply(logpsi, tf.stop_gradient(eloc_real)) - 
                2.0 * tf.stop_gradient(Eo_real) * logpsi
            )
            
            loss_imag = tf.reduce_mean(
                2.0 * tf.multiply(logpsi, tf.stop_gradient(eloc_imag)) - 
                2.0 * tf.stop_gradient(Eo_imag) * logpsi
            )
            
            loss = loss_real + alpha * loss_imag

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return eloc_real, eloc_imag

def main():
    # Configuration
    for N in range(100, 101, 10):
        g_imag = 0.16
        Jz = 1
        lr = 0.01
        nh = 34
        ns = 1024
        steps = 100
        seed = 111
        alpha = 0.1

        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        g_real_values = np.linspace(0, 3.0, 31)
        M = []
        
        with tf.device('/GPU:0'):
            with open(f'VMC_RBM_magnetization{N}.txt', "w+") as file:
                for i, g_real in enumerate(g_real_values):
                    vmc = VariationalMonteCarloRBM(N, Jz, g_real, g_imag, nh, lr, seed)
                    m = []
                    samples = None

                    for it in range(steps):
                        samples, _ = vmc.sample(ns, samples)
                        eloc_real, eloc_imag = vmc.training_step(samples, alpha)
                        
                        if it == steps - 1:
                            magnet = vmc.magnetization(samples)
                            m.append(magnet.numpy())
                    
                    mean_m = np.mean(m)
                    M.append(mean_m)
                    print(f"Energy: {np.mean(eloc_real)}+{np.mean(eloc_imag)}j, g_real: {g_real}, magnetization: {mean_m}")
                    file.write(f"{g_real} {g_imag} {mean_m}\n")

if __name__ == "__main__":
    main()
