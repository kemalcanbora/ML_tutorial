import numpy as np
import tensorflow as tf
import gym
np.random.seed(2)
tf.set_random_seed(2)  #reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)                      #reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

print(N_F)
print(N_A)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        
        # input 
        self.s = tf.placeholder(tf.float32, [1, n_features])
        # lebel
        self.a = tf.placeholder(tf.int32, None)
        # td_error
        self.td_error = tf.placeholder(tf.float32, None)
        
        layer1 = tf.layers.dense(
            inputs = self.s,
            units = 20,                 # number of hidden units
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(0, 0.1),  #weights
            bias_initializer = tf.constant_initializer(0.1),            #biases
        )
        
        self.acts_prob = tf.layers.dense(
            inputs = layer1,
            units = n_actions,
            activation = tf.nn.softmax,
            kernel_initializer = tf.random_normal_initializer(0, 0.1),  #weights
            bias_initializer = tf.constant_initializer(0.1),            #biases            
        )
          
        log_prob = tf.log(self.acts_prob[0, self.a])
        self.exp_v = tf.reduce_mean(log_prob * self.td_error)
        
        self.train_step = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
        
        
    def learn(self, s, a, td):    
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_step, self.exp_v], feed_dict)
        return exp_v
        
    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int



class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        
        # input
        self.s = tf.placeholder(tf.float32, [1, n_features])
        self.v_ = tf.placeholder(tf.float32, [1,1])
        self.r = tf.placeholder(tf.float32, None)

        layer1 = tf.layers.dense(
            inputs = self.s,
            units = 20,
            activation=tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(0, 0.1),  #weights
            bias_initializer = tf.constant_initializer(0.1),            #biases    
        )

        self.v = tf.layers.dense(
            inputs = layer1,
            units = 1,        # one value
            activation=None,
            kernel_initializer = tf.random_normal_initializer(0, 0.1),  #weights
            bias_initializer = tf.constant_initializer(0.1),            #biases    
        )

        self.td_error = self.r + GAMMA * self.v_ - self.v
        self.loss = tf.square(self.td_error)
        
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def learn(self, s, r, s_):
        s = s[np.newaxis, :]
        s_ = s_[np.newaxis, :]

        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_step], feed_dict={self.s: s, self.v_: v_, self.r:r})

        return td_error

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)

sess.run(tf.global_variables_initializer())
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/",sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()
            
        a = actor.choose_action(s)
        
        s_, r, done, info = env.step(a)
        
        if done: r=-20
            
        track_r.append(r)
        
        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)
        
        s = s_
        t += 1
        
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
