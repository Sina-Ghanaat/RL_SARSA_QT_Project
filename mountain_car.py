import numpy as np
import math
import gym
from gym import spaces

class MountainCar:
    def __init__(self, mass=0.2, friction=0.3, delta_t=0.1):
        """ Create a new mountain car object.
        
        It is possible to pass the parameter of the simulation.
        @param mass: the mass of the car (default 0.2) 
        @param friction:  the friction in Newton (default 0.3)
        @param delta_t: the time step in seconds (default 0.1)
        """
        self.position_list = list()
        self.noise = 0.3
        self.gravity = 9.8
        self.friction = friction
        self.delta_t = delta_t  # second
        self.mass = mass  # the mass of the car
        self.position_t = -0.5
        self.velocity_t = 0.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 1.5
        self.goal_position = 0.5

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
            )        
    def reset(self, exploring_starts=True, initial_position=-0.5):
        """ It reset the car to an initial position [-1.2, 0.5]
        
        @param exploring_starts: if True a random position is taken
        @param initial_position: the initial position of the car (requires exploring_starts=False)
        @return: it returns the initial position of the car and the velocity
        """
        if exploring_starts:
            initial_position = np.random.uniform(-1.2,0.5)
        if initial_position < -1.2:
            initial_position = -1.2
        if initial_position > 0.5:
            initial_position = 0.5
        self.position_list = []  # clear the list
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.position_list.append(initial_position)
        return [self.position_t, self.velocity_t]

    def set_noise(self, noise=0.3):
        if noise > 1:
            noise = 1
        elif noise < 0:
            noise = 0
        self.noise = noise

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2]
         where 0=move_left, 1=do_not_move, 2=move_right
        @return: (postion_t1, velocity_t1), reward, done
         where reward is always negative but when the goal is reached
         done is True when the goal is reached
        """

        if np.random.random() < self.noise:
            action = np.random.randint(0, 2)


        if(action >= 3):
            raise ValueError("[MOUNTAIN CAR][ERROR] The action value "
                             + str(action) + " is out of range.")
        done = False
        reward = -0.01
        action_list = [-0.2, 0, +0.2]
        action_t = action_list[action]
        velocity_t1 = self.velocity_t + \
                      (-self.gravity * self.mass * np.cos(3*self.position_t)
                       + (action_t/self.mass)
                       - (self.friction*self.velocity_t)) * self.delta_t
        position_t1 = self.position_t + (velocity_t1 * self.delta_t)
        # Check the limit condition (car outside frame)
        if position_t1 < -1.2:
            position_t1 = -1.2
            velocity_t1 = 0
        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t= velocity_t1
        self.position_list.append(position_t1)
        # Reward and done when the car reaches the goal
        if position_t1 >= 0.5:
            reward = +1.0
            done = True
        # Return state_t1, reward, done
        self.state = (position_t1, velocity_t1)
        return np.array(self.state), reward, done, {}


    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None