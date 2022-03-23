import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from numpy import linalg
from numpy.linalg import inv
import math
from math import cos, sin, pi
from scipy.integrate import odeint, solve_ivp

class Quad2DEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self): 
		self.capture = False
		self.FPS = 50 

		self.m = 1.1  # mass of quad, [kg]
		self.d = 0.24 # half of diameter, [m]
		self.Ixx = 0.002 # inertia matrix of quad, [kg m2]
		self.C_TQ = 0.013 # torques and thrusts coefficients
		self.total_power = 100                # [%]
		self.power_remaining = 100            # [%]
		self.power_discount_each_step = 0.01  # [%]

		# simulation parameters
		self.dt = 0.005 # discrete time step, t(2) - t(1), [sec]
		self.g  = 9.81  # standard gravity
		self.e3 = np.array([0.0, 0.0, 1.0])[np.newaxis].T

		# force
		self.force = self.m * self.g / 2.0 # magnitude of fixed thrust, [N], to overcome
									       # gravity & total mass (No air resistance)
		self.f = self.m * self.g
		self.min_force = 2.0  # [N]
		self.max_force = 10.0 # [N]
		self.f1 = self.force
		self.f2 = self.force

		# moment
		self.tau = 0.0

		# limits of states
		self.y_max_threshold     = 3.0 # [m]
		self.y_dot_max_threshold = 7.0 # [m/s]
		self.z_max_threshold     = 3.0 # [m]
		self.z_dot_max_threshold = 7.0 # [m/s]
		self.phi_max_threshold   = 60.0/180.0*math.pi # [rad]
		self.phi_dot_max_threshold = 1.0 # [rad/s]

		self.limits_y     = self.y_max_threshold # [m]
		self.limits_y_dot = self.y_dot_max_threshold # [m/s]
		self.limits_z     = self.z_max_threshold # [m]
		self.limits_z_dot = self.z_dot_max_threshold # [m/s]
		self.limits_phi   = self.phi_max_threshold     # [rad]
		self.limits_phi_dot = self.phi_dot_max_threshold # [rad/s]

		self.norm_limits_y     = self.limits_y / self.y_max_threshold # [m]
		self.norm_limits_y_dot = self.limits_y_dot / self.y_dot_max_threshold # [m/s]
		self.norm_limits_z     = self.limits_z / self.z_max_threshold # [m]
		self.norm_limits_z_dot = self.limits_z_dot / self.z_dot_max_threshold # [m/s]
		self.norm_limits_phi   = self.limits_phi / self.phi_max_threshold   # [rad]
		self.norm_limits_phi_dot = self.limits_phi_dot / self.phi_dot_max_threshold # [rad/s]

		# commands
		self.xd = np.array([0.0, 0.0]) / self.y_max_threshold # [m] 

		self.low  = -1.0
		self.high = 1.0

		# observation space
		self.observation_space = spaces.Box(
			self.low, 
			self.high, 
            shape=(6,),
			dtype=np.float64
		)
		# action space
		self.action_space = spaces.Box(
			low=self.min_force, 
			high=self.max_force, 
			shape=(2,),
			dtype=np.float64
		) 

		self.state = None
		self.viewer = None
		self.render_quad1  = None
		self.render_quad2  = None
		self.render_rotor1 = None
		self.render_rotor2 = None
		self.render_rotor3 = None
		self.render_rotor4 = None
		self.render_ref = None
		self.render_force_rotor1 = None
		self.render_force_rotor2 = None
		self.render_force_rotor3 = None
		self.render_force_rotor4 = None
		self.render_index = 1 
		
		self.seed()
		self.reset()


	def step(self, action):
		# Actions
		self.f1 = np.clip(action[0], self.min_force, self.max_force) # [N]
		self.f2 = np.clip(action[1], self.min_force, self.max_force)

		self.f = self.f1 + self.f2 # magnitude of total thrust
		self.tau = self.d*(self.f1 - self.f2) # magnitude of moment on quadrotor 

		## Solve ODEs
		_state = (self.state).flatten()

		# Reverse-normalization
		y       = _state[0] * self.y_max_threshold # [m]
		y_dot   = _state[1] * self.y_dot_max_threshold # [m/s]
		z       = _state[2] * self.z_max_threshold # [m]
		z_dot   = _state[3] * self.z_dot_max_threshold # [m/s]
		phi     = _state[4] * self.phi_max_threshold # [rad]
		phi_dot = _state[5] * self.phi_dot_max_threshold   # [rad/s]
		_state = np.concatenate((y, y_dot, z, z_dot, phi, phi_dot), axis=0)

		# solve: 'solve_ivp' Solver
		sol = solve_ivp(self.EoM, [0, self.dt], _state, method='DOP853') #RK45, LSODA, BDF, LSODA
		self.state = sol.y[:,-1]

		"""
		# solve: 1st-order
		# Equations of motion of the quadrotor UAV
		
		x_dot = v
		v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
		R_vec_dot = (R@self.hat(W)).reshape(9, 1, order='F')
		W_dot = inv(self.J)@(-self.hat(W)@self.J@W[np.newaxis].T + self.M)
		state_dot = np.concatenate([x_dot.flatten(), 
									v_dot.flatten(),                                                                          
									R_vec_dot.flatten(),
									W_dot.flatten()])
    
		self.state = _state + state_dot * self.dt
		"""

		# Normalization
		y       = self.state[0] / self.y_max_threshold # [m]
		y_dot   = self.state[1] / self.y_dot_max_threshold # [m/s]
		z       = self.state[2] / self.z_max_threshold # [m]
		z_dot   = self.state[3] / self.z_dot_max_threshold # [m/s]
		phi     = self.state[4] / self.phi_max_threshold # [rad]
		phi_dot = self.state[5] / self.phi_dot_max_threshold   # [rad/s]
		self.state = np.concatenate((y, y_dot, z, z_dot, phi, phi_dot), axis=0)

		# rewards	
		eX =  (self.state[0] - self.xd[0]) + (self.state[1] - self.xd[1])
		C_X = 2.0 
		reward = C_X*max(0, 1.0 - linalg.norm(eX, 2)) 

		done = False
		done = bool(
			   abs(self.state[0]) >= self.norm_limits_y
			or abs(self.state[1]) >= self.norm_limits_y_dot
			or abs(self.state[2]) >= self.norm_limits_z
			or abs(self.state[3]) >= self.norm_limits_z_dot
			or abs(self.state[4]) >= self.norm_limits_phi
			or abs(self.state[5]) >= self.norm_limits_phi_dot
		)

		return np.array(self.state), reward, done, {}


	def EoM(self, t, state):
		# https://youtu.be/iS5JFuopQsA
		y       = state[0] # [m]
		y_dot   = state[1] # [m/s]
		z       = state[2] # [m]
		z_dot   = state[3] # [m/s]
		phi     = state[4] # [rad]
		phi_dot = state[5] # [rad/s]

		# Equations of motion of the quadrotor UAV 2D
		y_dot    = y
		y_2dot   = self.f/self.m*sin(phi)
		z_dot    = z
		z_2dot   = self.g - self/self.m*cos(phi)
		phi_dot  = phi
		phi_2dot = self.tau/self.Ixx

		state_dot = np.concatenate([y_dot, y_2dot, 
									z_dot, z_2dot,                                                                         
									phi_dot, phi_2dot])
		
		return np.array(state_dot)


	def reset(self):

		self.state = np.array(np.zeros(6))

		x_error = 1.5
		_error  = 0.2

		self.state[0] = np.random.uniform(size = 1, low = -x_error, high = x_error) 
		self.state[1] = np.random.uniform(size = 1, low = -_error, high = _error) 
		self.state[2] = np.random.uniform(size = 1, low = -_error, high = _error) 
		self.state[3] = np.random.uniform(size = 1, low = -_error, high = _error) 
		# self.state[4] = np.random.uniform(size = 1, low = -_error, high = _error) 
		self.state[5] = np.random.uniform(size = 1, low = -_error, high = _error)
		
		return np.array(self.state)


	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas, cylinder, ring, arrow, scene, textures

		# states
		state_vis = (self.state).flatten()

		# Reverse-normalization
		norm_x = np.array([state_vis[0], state_vis[1], state_vis[2]]).flatten()     # [m]
		norm_v = np.array([state_vis[3], state_vis[4], state_vis[5]]).flatten() # [m/s]
		norm_W = np.array([state_vis[15], state_vis[16], state_vis[17]]).flatten()  # [rad/s]

		state_vis[0:3]   = norm_x * self.x_max_threshold # [m]
		state_vis[3:6]   = norm_v * self.v_max_threshold # [m/s]
		state_vis[15:18] = norm_W * self.W_max_threshold # [rad/s]

		x = np.array([state_vis[0], state_vis[1], state_vis[2]]).flatten() # [m]
		v = np.array([state_vis[3], state_vis[4], state_vis[5]]).flatten() # [m/s]
		R_vec = np.array([state_vis[6], state_vis[7], state_vis[8],
						  state_vis[9], state_vis[10], state_vis[11],
						  state_vis[12], state_vis[13], state_vis[14]]).flatten()
		W = np.array([state_vis[15], state_vis[16], state_vis[17]]).flatten()   # [rad/s]
		
		cmd_pos  = self.xd * self.x_max_threshold # [m]
		quad_pos = x

		# axis
		x_axis = np.array([state_vis[6], state_vis[7], state_vis[8]]).flatten()
		y_axis = np.array([state_vis[9], state_vis[10], state_vis[11]]).flatten()
		z_axis = np.array([state_vis[12], state_vis[13], state_vis[14]]).flatten()

		# init
		if self.viewer is None:
			# canvas
			self.viewer = canvas(title = 'Quadrotor with RL', width = 1024, height = 768, \
								center = vector(0, 0, cmd_pos[2]), background = color.white, \
								forward = vector(1, 0.3, 0.3), up = vector(0, 0, -1)) # forward = view point
			
			# quad body
			self.render_quad1 = box(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
									axis = vector(x_axis[0], x_axis[1], x_axis[2]), \
									length = 0.2, height = 0.05, width = 0.05) # vector(quad_pos[0], quad_pos[1], 0)
			self.render_quad2 = box(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
									axis = vector(y_axis[0], y_axis[1], y_axis[2]), \
									length = 0.2, height = 0.05, width = 0.05)
			# rotors
			rotors_offest = 0.02
			self.render_rotor1 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
										radius = 0.2, color = color.blue, opacity = 0.5)
			self.render_rotor2 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
										radius = 0.2, color = color.cyan, opacity = 0.5)
			self.render_rotor3 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
										radius = 0.2, color = color.blue, opacity = 0.5)
			self.render_rotor4 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
										radius = 0.2, color = color.cyan, opacity = 0.5)

			# forces arrow
			self.render_force_rotor1 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
											axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
											shaftwidth = 0.05, color = color.blue)
			self.render_force_rotor2 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
											axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
											shaftwidth = 0.05, color = color.cyan)
			self.render_force_rotor3 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
											axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
											shaftwidth = 0.05, color = color.blue)
			self.render_force_rotor4 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
											axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
											shaftwidth = 0.05, color = color.cyan)
									
			# commands
			self.render_ref = sphere(canvas = self.viewer, pos = vector(cmd_pos[0], cmd_pos[1], cmd_pos[2]), \
									 radius = 0.07, color = color.red, \
									 make_trail = True, trail_type = 'points', interval = 50)									
			
			# inertial axis				
			self.e1_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(1, 0, 0), \
								 shaftwidth = 0.04, color=color.blue)
			self.e2_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(0, 1, 0), \
				    			 shaftwidth = 0.04, color=color.green)
			self.e3_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(0, 0, 1), \
								 shaftwidth = 0.04, color=color.red)

			# body axis				
			self.render_b1_axis = arrow(canvas = self.viewer, 
										pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(x_axis[0], x_axis[1], x_axis[2]), \
										shaftwidth = 0.02, color = color.blue,
										make_trail = True, trail_color = color.yellow)
			self.render_b2_axis = arrow(canvas = self.viewer, 
										pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(y_axis[0], y_axis[1], y_axis[2]), \
										shaftwidth = 0.02, color = color.green)
			self.render_b3_axis = arrow(canvas = self.viewer, 
										pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
										axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
										shaftwidth = 0.02, color = color.red)

			# floor
			self.floor = box(pos = vector(0,0,0),size = vector(5,5,0.05), axis = vector(1,0,0), \
							 opacity = 0.2, color = color.black)
		
		if self.state is None: 
			return None
		
		# quad body
		self.render_quad1.pos.x = quad_pos[0]
		self.render_quad1.pos.y = quad_pos[1]
		self.render_quad1.pos.z = quad_pos[2]
		self.render_quad2.pos.x = quad_pos[0]
		self.render_quad2.pos.y = quad_pos[1]
		self.render_quad2.pos.z = quad_pos[2]
		
		self.render_quad1.axis.x = x_axis[0]
		self.render_quad1.axis.y = x_axis[1]	
		self.render_quad1.axis.z = x_axis[2]
		self.render_quad2.axis.x = y_axis[0]
		self.render_quad2.axis.y = y_axis[1]
		self.render_quad2.axis.z = y_axis[2]

		self.render_quad1.up.x = z_axis[0]
		self.render_quad1.up.y = z_axis[1]
		self.render_quad1.up.z = z_axis[2]
		self.render_quad2.up.x = z_axis[0]
		self.render_quad2.up.y = z_axis[1]
		self.render_quad2.up.z = z_axis[2]

		# rotors
		rotors_offest = -0.02
		rotor_pos = 0.5*x_axis
		self.render_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = 0.5*y_axis
		self.render_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*x_axis
		self.render_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*y_axis
		self.render_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

		self.render_rotor1.axis.x = rotors_offest*z_axis[0]
		self.render_rotor1.axis.y = rotors_offest*z_axis[1]
		self.render_rotor1.axis.z = rotors_offest*z_axis[2]
		self.render_rotor2.axis.x = rotors_offest*z_axis[0]
		self.render_rotor2.axis.y = rotors_offest*z_axis[1]
		self.render_rotor2.axis.z = rotors_offest*z_axis[2]
		self.render_rotor3.axis.x = rotors_offest*z_axis[0]
		self.render_rotor3.axis.y = rotors_offest*z_axis[1]
		self.render_rotor3.axis.z = rotors_offest*z_axis[2]
		self.render_rotor4.axis.x = rotors_offest*z_axis[0]
		self.render_rotor4.axis.y = rotors_offest*z_axis[1]
		self.render_rotor4.axis.z = rotors_offest*z_axis[2]

		self.render_rotor1.up.x = y_axis[0]
		self.render_rotor1.up.y = y_axis[1]
		self.render_rotor1.up.z = y_axis[2]
		self.render_rotor2.up.x = y_axis[0]
		self.render_rotor2.up.y = y_axis[1]
		self.render_rotor2.up.z = y_axis[2]
		self.render_rotor3.up.x = y_axis[0]
		self.render_rotor3.up.y = y_axis[1]
		self.render_rotor3.up.z = y_axis[2]
		self.render_rotor4.up.x = y_axis[0]
		self.render_rotor4.up.y = y_axis[1]
		self.render_rotor4.up.z = y_axis[2]

		# forces arrow
		rotor_pos = 0.5*x_axis
		self.render_force_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_force_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_force_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = 0.5*y_axis
		self.render_force_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_force_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_force_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*x_axis
		self.render_force_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_force_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_force_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*y_axis
		self.render_force_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
		self.render_force_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
		self.render_force_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

		force_offest = -0.05
		self.render_force_rotor1.axis.x = force_offest * self.f1 * z_axis[0] 
		self.render_force_rotor1.axis.y = force_offest * self.f1 * z_axis[1]
		self.render_force_rotor1.axis.z = force_offest * self.f1 * z_axis[2]
		self.render_force_rotor2.axis.x = force_offest * self.f2 * z_axis[0]
		self.render_force_rotor2.axis.y = force_offest * self.f2 * z_axis[1]
		self.render_force_rotor2.axis.z = force_offest * self.f2 * z_axis[2]
		self.render_force_rotor3.axis.x = force_offest * self.f3 * z_axis[0]
		self.render_force_rotor3.axis.y = force_offest * self.f3 * z_axis[1]
		self.render_force_rotor3.axis.z = force_offest * self.f3 * z_axis[2]
		self.render_force_rotor4.axis.x = force_offest * self.f4 * z_axis[0]
		self.render_force_rotor4.axis.y = force_offest * self.f4 * z_axis[1]
		self.render_force_rotor4.axis.z = force_offest * self.f4 * z_axis[2]

		# commands
		self.render_ref.pos.x = cmd_pos[0]
		self.render_ref.pos.y = cmd_pos[1]
		self.render_ref.pos.z = cmd_pos[2]

		# body axis	
		axis_offest = 0.8
		self.render_b1_axis.pos.x = quad_pos[0]
		self.render_b1_axis.pos.y = quad_pos[1]
		self.render_b1_axis.pos.z = quad_pos[2]
		self.render_b2_axis.pos.x = quad_pos[0]
		self.render_b2_axis.pos.y = quad_pos[1]
		self.render_b2_axis.pos.z = quad_pos[2]
		self.render_b3_axis.pos.x = quad_pos[0]
		self.render_b3_axis.pos.y = quad_pos[1]
		self.render_b3_axis.pos.z = quad_pos[2]

		self.render_b1_axis.axis.x = axis_offest * x_axis[0] 
		self.render_b1_axis.axis.y = axis_offest * x_axis[1] 
		self.render_b1_axis.axis.z = axis_offest * x_axis[2] 
		self.render_b2_axis.axis.x = axis_offest * y_axis[0] 
		self.render_b2_axis.axis.y = axis_offest * y_axis[1] 
		self.render_b2_axis.axis.z = axis_offest * y_axis[2] 
		self.render_b3_axis.axis.x = (axis_offest/2) * z_axis[0] 
		self.render_b3_axis.axis.y = (axis_offest/2) * z_axis[1]
		self.render_b3_axis.axis.z = (axis_offest/2) * z_axis[2]

		# self.render_b1_axis.up.x = z_axis[0]
		# self.render_b1_axis.up.y = z_axis[1]
		# self.render_b1_axis.up.z = z_axis[2]
		# self.render_b2_axis.up.x = z_axis[0]
		# self.render_b2_axis.up.y = z_axis[1]
		# self.render_b2_axis.up.z = z_axis[2]
		# self.render_b3_axis.up.x = z_axis[0]
		# self.render_b3_axis.up.y = z_axis[1]
		# self.render_b3_axis.up.z = z_axis[2]
		
		# Screen capture
		if self.capture == True:
			if (self.render_index % 5) == 0:
				self.viewer.capture('capture'+str(self.render_index))
			self.render_index += 1

		rate(self.FPS)

		return True


	def close(self):
		if self.viewer:
			self.viewer = None


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]