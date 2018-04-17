import numpy as np

class Tiling(object):
    def __init__(self, distance_min, distance_max, velocity_min, velocity_max, height_min, height_max, velocity_number=13):
        '''Creates tilings for all dimensions'''
        # Distance
        self.distance_tile = self.create_distance_tile(distance_min, distance_max)

        # Velocity
        self.velocity_number = velocity_number
        self.velocity_tile = self.create_velocity_tile_tile(velocity_min, velocity_max, velocity_number)

        # Height
        self.height_tile = self.create_height_tile(height_min, height_max)

    def create_distance_tile(self, distance_min, distance_max):
        '''Creates geomspace distance tiles'''
        pass

    def create_velocity_tile(self, velocity_min, velocity_max, velocity_number):
        '''Creates even spaced velocity tiles'''
        # Initialize first tile
        velocity_range = velocity_max-velocity_min
        velocity_tile_width = velocity_range/velocity_number
        initial_velocity_tile = np.linspace(velocity_min, velocity_max+velocity_tile_width, velocity_number+1)


    def create_height_tile(self, height_min, height_max):
        '''Creates centred logspaced tiles'''
        pass
