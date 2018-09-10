import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, debug=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.debug = debug

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        def pose_reward(x,h_offset=0.,curve=-0.1,max_score=1.,min_reward=0.):
            reward = (curve*(max_score/10.)*((x+h_offset)**2.))+max_score
            reward = min_reward if reward <= min_reward else reward
            return reward
        
        def linear_pose_reward(x,slope,base):
            return (slope*x) + base
        
        x = self.sim.pose[0:1].item()
        y = self.sim.pose[1:2].item()
        z = self.sim.pose[2:3].item()
        
        #rewards
        #pose_reward = np.linalg.norm(abs(self.sim.pose[:3] - self.target_pos)).sum() # 0 would be optimal in this case
        #if self.debug:
        #    print("pose before: " + str(pose_reward))
#        pose_reward = np.linalg.norm(-1*(pose_reward ** 2)).item() # penalize higher values more
        #pose_reward = pose_reward ** 2
        #if self.debug:
        #    print("pose **2: " + str(pose_reward))        
        #pose_reward = pose_reward * -1            
        #if self.debug:
        #    print("pose *-1: " + str(pose_reward))        
#        pose_reward = np.linalg.norm(pose_reward).item()            
#        if self.debug:
#            print("pose LINALG: " + str(pose_reward))
            
        #static z reward experiment
        # z_reward = 10. * z if z > 0 else 0
        #dynamic z reward experiment
        # z_reward = -1 * (z ** 2) + self.target_pos[2:3].item() # again 0 would be optimale and penalize odd values
        #z_reward = z_reward if z > 0 else z * 10
#        z_reward = 1.5*z
        x_reward = pose_reward(x,curve=-0.5)
        y_reward = pose_reward(y,curve=-0.5)
        z_reward = linear_pose_reward(z,0.4,-3.) if z <= 10 else linear_pose_reward(z,(-1./190.),(1.+(1./19.)))
        # z_reward = pose_reward(z,curve=-0.001,h_offset=-10.,max_score=1.)
        # z_reward = 0 if z <= 3. else z_reward
        # static_z_reward = -5.0 if z == 0. else z/10.
        
        #penalties
        #apply penalty for z
        #z_penalty = np.linalg.norm(abs(z - (z ** 2)))
        #z_penalty = -1 * (z_penalty ** 2) + (z ** 2)
        
        reward = (z_reward * x_reward) + (z_reward * y_reward) + z_reward # + static_z_reward
        
        if self.debug:
            print("--- get_reward function ---")            
            print("-- rewards --")
#            print("pose_reward: " + str(pose_reward))
            print("x_reward: " + str(x_reward))
            print("y_reward: " + str(y_reward))
            print("z_reward: " + str(z_reward))
#            print("static_z_reward: " + str(static_z_reward))
            print("-- penalties --")
#            print("z_penalty: " + str(z_penalty))
            print("-- -- --")
            print("final reward:" + str(reward))
            print("---")
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        if self.debug:
            print("--- step function ---")
            print("next_state: " + str(next_state))
            print("reward: " + str(reward))
            print("done: " + str(done))
            print("---")
            
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state