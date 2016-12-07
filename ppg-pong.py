""" Trains an agent with (stochastic) Parallel Policy Gradients on Pong. Uses OpenAI Gym. 

    Based on Andrej Karpathy's excellent example code: 
        https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    
    Adapted for MPI parallelism by Mark O'Connor:
        https://www.allinea.com/blog/201610/deep-learning-episode-4-supercomputer-vs-pong-ii
"""

import numpy as np
import cPickle as pickle
import gym
from mpi4py import MPI
from time import time, sleep
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-r', '--learning_rate', action='store', dest='learning_rate', default=1e-3)
parser.add_option('-w', '--reward_window', action='store', dest='reward_window', default=100)
parser.add_option('-d', '--episode_duration', action='store', dest='episode_duration', default=2.0)
parser.add_option('-t', '--target_score', action='store', dest='target_score', default=0.0)
parser.add_option('-e', '--end_at_rate', action='store', dest='end_at_rate', default=1e-6)
parser.add_option('-c', '--load_checkpoint', action='store', dest='load_checkpoint')
parser.add_option('-s', '--show_match', action='store_true', dest='show_match')
options, args = parser.parse_args()

# hyperparameters
H = 200 # number of hidden layer neurons
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

learning_rate = float(options.learning_rate)
reward_window = int(options.reward_window)
end_at_rate = float(options.end_at_rate)
episode_duration = float(options.episode_duration) # seconds for each episode
target_score = float(options.target_score)
load_checkpoint = options.load_checkpoint # resume from previous checkpoint?
show_match = options.show_match

# mpi initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
reward_sum = 0
update_count = 0
steps = 0
if load_checkpoint:
  if rank == 0: print 'Loading from checkpoint "%s"...' % load_checkpoint
  data = pickle.load(open(load_checkpoint, 'rb'))
  model, rmsprop_cache, update_count, steps, reward_sum = data
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def mpi_merge(source):
  dest = np.empty_like(source)
  comm.Allreduce(source, dest)
  return dest / float(comm.size)


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
recent_rewards = []

comm.Barrier()
if rank == 0:
    print 'Running with %d processes' % comm.size

time_start = time()
episode_start = time()
finished = False
initial_steps = steps
last_frame = time()
while not finished:
  if show_match:
      delay = time() - last_frame + 1.0 / 60.0
      if delay > 0.0:
          sleep(delay)
      env.render()
      last_frame = time()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  #if show_match:
  #    action = 2 if aprob > 0.5 else 3 # take our best shot in a real match
  #else:
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  steps += 1
  if done:
    env.reset() # loop through multiple games if necessary to complete our episode steps

  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if not show_match and (
           (episode_duration > 0 and time() - episode_start >= episode_duration)
        or (episode_duration <= 0.0 and done)): # -d 0 links us to 1 episode = 1 game, first to 21 points
    update_count += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= (np.std(discounted_epr) + 0.00001)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update 
    for k,v in model.iteritems():
      g = mpi_merge(grad_buffer[k]) # merged gradient
      rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
      model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
      grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # collect current overall reward
    total_reward = comm.allreduce(reward_sum, op=MPI.SUM)
    total_steps = comm.allreduce(steps, op=MPI.SUM)
    reward_sum = total_reward / float(comm.size)
    running_reward = reward_sum if running_reward is None else running_reward * 0.95 + reward_sum * 0.05

    recent_rewards.append(reward_sum)
    if len(recent_rewards) > reward_window:
      recent_rewards.pop(0)

    # show learning rate within reward window - are we converging?
    sample_n = max(1, len(recent_rewards)/2)
    before = recent_rewards[:sample_n]
    after  = recent_rewards[sample_n:]
    progress = sum(after)/max(1, len(after)) - sum(before)/max(1, len(before))
    progress_str = "%7.1f r/Kup" % (1000 * progress / sample_n) # effectively comparing two windowed averages sample_n updates apart

    # print output and save progress checkpoints
    if rank == 0:
      elapsed = time() - time_start
      print '%6d, %010d steps @ %6.0f/s: %5.1f | %8.4f @ %.1e lr, %s' % (elapsed, total_steps, (total_steps - initial_steps*comm.size)/ float(elapsed), reward_sum, running_reward, learning_rate, progress_str)
      finished = running_reward >= target_score
      if finished or update_count % 20 == 0:
        data = (model, rmsprop_cache, update_count, steps, reward_sum)
        pickle.dump(data, open('step-%010d.p' % (total_steps), 'wb'), protocol=2)

    # begin a new episode but do not reset the simulation
    finished = comm.bcast(finished)
    reward_sum = 0
    prev_x = None
    episode_start = time()

