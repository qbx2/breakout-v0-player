import tensorflow as tf
import numpy as np
import datetime
import threading
import random
import time
import sys
import gym
env = gym.make('Breakout-v0')

CPU_ONLY = False
TRAIN = True
BENCHMARK = False

if 'eval' in sys.argv:
	TRAIN = False
if 'cpu' in sys.argv:
	CPU_ONLY = True
if 'benchmark' in sys.argv:
	BENCHMARK = True

NUM_AGENT_THREAD = 4
LOG_INTERVAL = 1000
SAVE_INTERVAL = 50000

# hyperparameter settings
GAMMA = .95
LEARNING_RATE = .0002
DECAY_RATE = .99
MOMENTUM = 0
EPSILON = 1e-6

BATCH_SIZE = 32
OBSERVE = 50000
ACTION_HISTORY_LENGTH = 4
MAX_EXPLORE_FRAMES = 1000000
MIN_EXPLORE_RATE = .10
MAX_D_SIZE = 1000000 # maximum size of replay queue
C = 10000 # Q reset interval
SCREEN_DIMS = 84, 84

NUM_ACTIONS = env.action_space.n
ACTION_MEANINGS = env.get_action_meanings()

env = None

print('breakout-v0-player is running with TRAIN=%s'%TRAIN)

def conv2d(x, W, s, cpu_only=False):
	cpu_only = CPU_ONLY or cpu_only
	return tf.nn.conv2d(x, W, strides=[1, s, s, 1] if cpu_only else [1, 1, s, s], padding='VALID', data_format='NHWC' if cpu_only else 'NCHW')

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.02)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial, name=name)

def create_q(state, weights=None, cpu_only=False):
	cpu_only = CPU_ONLY or cpu_only

	if weights is not None:
		w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2 = weights

	if cpu_only:
		state = tf.transpose(state, perm=[0,2,3,1])

	# state: (x_1, x_2, ... x_n) of shape [-1, ACTION_HISTORY_LENGTH, HEIGHT, WIDTH]
	with tf.name_scope('conv1'):
		if weights is None:
			w_conv1 = weight_variable([8, 8, ACTION_HISTORY_LENGTH, 32], name='w_conv1')
			b_conv1 = bias_variable([32], name='b_conv1')
		h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(state, w_conv1, 4, cpu_only), b_conv1, data_format='NHWC' if cpu_only else 'NCHW'))

	with tf.name_scope('conv2'):
		if weights is None:
			w_conv2 = weight_variable([4, 4, 32, 64], name='w_conv2')
			b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2, 2, cpu_only), b_conv2, data_format='NHWC' if cpu_only else 'NCHW'))

	with tf.name_scope('conv3'):
		if weights is None:
			w_conv3 = weight_variable([3, 3, 64, 64], name='w_conv3')
			b_conv3 = bias_variable([64])
		h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv2, w_conv3, 1, cpu_only), b_conv3, data_format='NHWC' if cpu_only else 'NCHW'))

	if cpu_only:
		h_conv3 = tf.transpose(h_conv3, perm=[0,3,1,2])

	shape = h_conv3.get_shape().as_list()
	H, W = shape[2], shape[3]
	h_conv3_flattened = tf.reshape(h_conv3, [-1, 64*H*W], name='h_conv3_flatten')

	with tf.name_scope('fc1'):
		if weights is None:
			w_fc1 = weight_variable([64*H*W, 512])
			b_fc1 = bias_variable([512])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flattened, w_fc1) + b_fc1)

	with tf.name_scope('fc2'):
		if weights is None:
			w_fc2 = weight_variable([512, NUM_ACTIONS])
			b_fc2 = bias_variable([NUM_ACTIONS])
		h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2

	return h_fc2, (w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2)

def create_predicted_action(q_values):
	return tf.argmax(q_values, 1)

def create_max_q(q_values):
	return tf.reduce_max(q_values, reduction_indices=1)

def create_q_reduced_by_action(q_values, a):
	one_hot_encoded_a = tf.one_hot(a, NUM_ACTIONS, 1., 0.)
	q_value = tf.reduce_sum(q_values * one_hot_encoded_a, reduction_indices=1)
	return q_value

def create_loss(q_values, y, a):
	q_value = create_q_reduced_by_action(q_values, a)
	loss = tf.reduce_mean(tf.square(y - q_value))
	return loss

def create_train_op(loss):
	return tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, MOMENTUM, EPSILON).minimize(loss)

def create_preprocess(x):
	grayscale = tf.image.rgb_to_grayscale(x)
	resized = tf.image.resize_images(grayscale, *SCREEN_DIMS)/255.
	return resized

def start_session():
	global global_step, ph_new_global_step, assign_global_step
	global ph_state, ph_x
	global _preprocess, predicted_action, q_values, max_q, predicted_action_cpu, q_values_cpu, max_q_cpu
	global gamma_max_target_q, reset_target_q, gamma_target_q_reduced_by_action, predict_by_double_dqn
	global ph_y, ph_a
	global loss, train_op
	global input_summary, ph_avg_reward, reward_summary, ph_avg_score_per_episode, score_per_episode_summary, ph_avg_loss, loss_summary, ph_avg_max_q_value, max_q_value_summary, ph_exploration_rate, exploration_rate_summary
	
	with tf.Graph().as_default() as g:
		global_step = tf.Variable(0, name='step', trainable=False)
		ph_new_global_step = tf.placeholder(tf.int32, shape=[], name='new_global_step')
		assign_global_step = tf.assign(global_step, ph_new_global_step, name='assign_global_step')

		with tf.name_scope('input'):
			# preprocessed state(x_1, x_2, ..., x_n)
			ph_x = tf.placeholder(tf.int32, shape=[210, 160, 3])
			ph_state = tf.placeholder(tf.float32, shape=[None, ACTION_HISTORY_LENGTH, *SCREEN_DIMS], name='state')
			ph_y = tf.placeholder(tf.float32, shape=[None], name='y') # y = r or r + gamma * max_Q^(s, a)
			ph_a = tf.placeholder(tf.int64, shape=[None], name='a') # actions

		with tf.device('/gpu:0'):
			with tf.name_scope('Q'):
				q_values, theta = create_q(ph_state)

			with tf.name_scope('pi'):
				predicted_action = create_predicted_action(q_values)

			with tf.name_scope('max_Q'):
				max_q = create_max_q(q_values)

			with tf.name_scope('target_Q'):
				target_q_values, theta_m1 = create_q(ph_state)

			with tf.name_scope('target_Q_reduced_by_action'):
				target_q_reduced_by_action = create_q_reduced_by_action(target_q_values, ph_a)

			with tf.name_scope('gamma_target_Q_reduced_by_action'):
				gamma_target_q_reduced_by_action = GAMMA * target_q_reduced_by_action

			with tf.name_scope('predict_by_double_dqn'):
				predict_by_double_dqn = GAMMA * create_q_reduced_by_action(target_q_values, predicted_action)

			with tf.name_scope('max_target_Q'):
				max_target_q = create_max_q(target_q_values)

			with tf.name_scope('gamma_max_target_Q'):
				gamma_max_target_q = GAMMA * max_target_q

			with tf.name_scope('reset_target_Q'):
				reset_target_q = tf.group(*(tf.assign(lvalue, rvalue) for lvalue, rvalue in zip(theta_m1, theta)))

			with tf.name_scope('loss'):
				loss = create_loss(q_values, ph_y, ph_a)

			with tf.name_scope('train'):
				train_op = create_train_op(loss)

		with tf.device('/cpu:0'):
			with tf.name_scope('preprocess'):
				_preprocess = create_preprocess(ph_x)

			with tf.name_scope('Q_cpu'):
				q_values_cpu, _ = create_q(ph_state, theta, cpu_only=True)

			with tf.name_scope('pi_cpu'):
				predicted_action_cpu = create_predicted_action(q_values_cpu)

			with tf.name_scope('max_Q_cpu'):
				max_q_cpu = create_max_q(q_values_cpu)

			# summaries
			input_summary = tf.image_summary('input', tf.reshape(tf.transpose(ph_state[0:1,:,:,:], perm=[1,2,3,0]), [-1, *SCREEN_DIMS, 1]), max_images=ACTION_HISTORY_LENGTH)

			# update every input()
			ph_avg_reward = tf.placeholder(tf.float32, shape=[], name='avg_reward')
			reward_summary = tf.scalar_summary('_reward', ph_avg_reward)

			# update at new_episode()
			ph_avg_score_per_episode = tf.placeholder(tf.float32, shape=[], name='avg_score_per_episode')
			score_per_episode_summary = tf.scalar_summary('_score_per_episode', ph_avg_score_per_episode)

			# update at train()
			ph_avg_loss = tf.placeholder(tf.float32, shape=[], name='avg_loss')
			loss_summary = tf.scalar_summary('_loss', ph_avg_loss)

			# update at train()
			ph_exploration_rate = tf.placeholder(tf.float32, shape=[], name='avg_loss')
			exploration_rate_summary = tf.scalar_summary('_exploration_rate', ph_exploration_rate)

			# update at inference
			ph_avg_max_q_value = tf.placeholder(tf.float32, shape=[], name='avg_max_q_value')
			max_q_value_summary = tf.scalar_summary('_max_q_value', ph_avg_max_q_value)

		# start session
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		initializers = (tf.initialize_all_variables(), reset_target_q)

		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")

		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
			import os
			assert os.path.isdir('saved_networks')
			assert TRAIN

			for initializer in initializers:
				sess.run(initializer)

		g.finalize()

	return sess, saver

def save_networks(step):
	sess.run(assign_global_step, feed_dict={ph_new_global_step: step})
	saver.save(sess, 'saved_networks/' + 'network' + '-dqn', global_step=step)
	print('[%s] Successfully saved networks -'%datetime.datetime.now(), step)

def get_exploration_rate():
	return max(MIN_EXPLORE_RATE, 1. + (MIN_EXPLORE_RATE - 1.) * step / MAX_EXPLORE_FRAMES)

def train_step():
	global step, st, ps
	global total_loss, cnt_loss

	minibatch = random.sample(D, BATCH_SIZE)

	state_batch = []
	action_batch = []
	y_batch = []
	undone_indices = []
	undone_state_p1 = []

	for i, (t_state, t_action, t_reward, t_state_p1, t_done) in enumerate(minibatch):
		state_batch.append(t_state)
		action_batch.append(t_action)
		y_batch.append(t_reward)

		if t_done == False: # to calculate future rewards
			undone_indices.append(i)
			undone_state_p1.append(t_state_p1)

	# calculate future rewards
	predicted_q_values = sess.run(gamma_max_target_q, feed_dict={ph_state: undone_state_p1})

	# double DQN
	#predicted_q_values = sess.run(predict_by_double_dqn, feed_dict={ph_state: undone_state_p1})

	for i, j in enumerate(undone_indices):
		y_batch[j] += predicted_q_values[i]

	# train
	_, current_loss = sess.run([train_op, loss], feed_dict={ph_y: y_batch, ph_state: state_batch, ph_a: action_batch})
	
	# log loss
	cnt_loss += 1
	total_loss += current_loss
	t_cnt_loss = cnt_loss

	if t_cnt_loss == (LOG_INTERVAL // 10): # and TRAIN # is always True
		summary_writer.add_summary(sess.run(loss_summary, feed_dict={ph_avg_loss: total_loss/cnt_loss}), step)
		summary_writer.add_summary(sess.run(exploration_rate_summary, feed_dict={ph_exploration_rate: get_exploration_rate()}), step)

		total_loss = 0
		cnt_loss = 0

	step += 1

	if BENCHMARK and step%100==0:
		print((step-ps)/(time.time()-st),'iterations per second')
		st = time.time()
		ps = step
	
	if step % C == 0:
		sess.run(reset_target_q)
	
	if step % SAVE_INTERVAL == 0 and not BENCHMARK:
		print('Autosaving networks ...')
		save_networks(step)

def preprocess(x):
	return sess.run(_preprocess, feed_dict={ph_x: x})[:, :, 0]

def put_experience(s, a, r, s_p, t, D_lock=None):
	global D_index

	if D_lock:
		D_lock.acquire()

	new_exp = (s, a, r, s_p, t)

	if len(D) >= MAX_D_SIZE:
		D[D_index] = new_exp
		D_index += 1
		if D_index == len(D):
			D_index = 0
	else:
		D.append(new_exp)

	if D_lock:
		D_lock.release()

def agent_worker(agent_coord, D_lock=None):
	assert OBSERVE <= MAX_D_SIZE

	global D, total_loss, cnt_loss, st, ps

	env = gym.make('Breakout-v0')
	get_state = lambda current:prev_ob_list[-ACTION_HISTORY_LENGTH:] if current else prev_ob_list[-ACTION_HISTORY_LENGTH-1:-1]
	
	total_reward = 0
	cnt_reward = 0

	total_score_per_episode = 0
	cnt_score_per_episode = 0

	total_max_q_value = 0
	cnt_max_q_value = 0

	total_loss = 0
	cnt_loss = 0

	# benchmark
	st = time.time()
	ps = step

	while not agent_coord.should_stop():
		# new episode
		observation = env.reset()
		done = None
		score = 0
		cnt_same_state = 0
		last_score = None

		prev_ob_list = [preprocess(observation)] * (ACTION_HISTORY_LENGTH - 1) # previous observations

		while not agent_coord.should_stop():
			prev_ob_list.append(preprocess(observation))

			if not TRAIN:
				env.render()

			if done is not None and TRAIN:
				put_experience(get_state(False), action, min(1, reward), get_state(True), done, D_lock)

				if len(D) > (OBSERVE if not BENCHMARK else BATCH_SIZE):
					train_step()

			if done is not None and done:
				if not TRAIN:
					print('score:', score)
					time.sleep(1)
				break

			if TRAIN and (random.random() < get_exploration_rate()):
				action = env.action_space.sample()
			else:
				# evaluate
				ops = [predicted_action, max_q]

				if not TRAIN:
					ops = [predicted_action, max_q, q_values]
					
				feed_dict = {ph_state: (get_state(True),)}

				if cnt_max_q_value == LOG_INTERVAL:
					ops.extend([input_summary, max_q_value_summary])
					feed_dict[ph_avg_max_q_value] = total_max_q_value / cnt_max_q_value
					total_max_q_value = 0
					cnt_max_q_value = 0

				ret = sess.run(ops, feed_dict=feed_dict)
				action = ret[0][0]

				# prevent the agent from doing nothing
				if not TRAIN:
					if last_score == score:
						cnt_same_state += 1

						if cnt_same_state >= 50:
							action = 1 # FIRE
							cnt_same_state = 0
					else:
						cnt_same_state = 0

					last_score = score

				if len(D) >= OBSERVE:
					total_max_q_value += ret[1][0]
					cnt_max_q_value += 1

				if TRAIN:
					for summary in ret[2:]:
						summary_writer.add_summary(summary, step)
				else:
					print(ret[-1])
					print(ACTION_MEANINGS[action], '\t' if len(ACTION_MEANINGS[action]) >= 8 else '\t\t', ret[1][0])

			observation, reward, done, info = env.step(action)
			score += reward

			if len(D) >= OBSERVE:
				total_reward += reward
				cnt_reward += 1

				if cnt_reward == (LOG_INTERVAL*10):
					summary_writer.add_summary(sess.run(reward_summary, feed_dict={ph_avg_reward: total_reward/cnt_reward}), step)
					total_reward = 0
					cnt_reward = 0

		# episode done
		if len(D) >= OBSERVE:
			total_score_per_episode += score
			cnt_score_per_episode += 1

			if cnt_score_per_episode == (LOG_INTERVAL//10):
				summary_writer.add_summary(sess.run(score_per_episode_summary, feed_dict={ph_avg_score_per_episode:total_score_per_episode/cnt_score_per_episode}), step)
				total_score_per_episode = 0
				cnt_score_per_episode = 0

def main():
	global sess, saver, summary_writer, D, D_index, step
	sess, saver = start_session()
	step = sess.run(global_step)

	summary_writer=tf.train.SummaryWriter('logdir', sess.graph)
	coord = tf.train.Coordinator()

	D = [] # replay memory
	D_index = 0

	if TRAIN:
		D_lock = threading.Lock()
	
		agent_coord = tf.train.Coordinator()
		agent_threads = []

		for i in range(NUM_AGENT_THREAD):
			agent_thread = threading.Thread(target=agent_worker, args=(agent_coord, D_lock))
			agent_thread.start()
			agent_threads.append(agent_thread)

		print("Waiting for initial observation")

		while len(D) < (OBSERVE if not BENCHMARK else BATCH_SIZE):
			print("Current len(D):", len(D))
			time.sleep(1)

		agent_coord.request_stop()
		agent_coord.join(agent_threads)

	try:
		agent_worker(coord)
	except Exception as e:
		print(e)
		# Report exceptions to the coordinator.
		coord.request_stop(e)
	finally:
		coord.request_stop()

		if TRAIN and not BENCHMARK:
			print('Received should_stop - Saving networks ...')
			save_networks(step)

if __name__ == '__main__':
	main()