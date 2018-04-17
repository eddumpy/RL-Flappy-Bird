import numpy as np

class Tiling(object):
    def __init__(self, distance_min, distance_max, velocity_min, velocity_max, height_min, height_max, distance_number=15, velocity_number=13, height_number=10, overlap=5):
        '''Creates tilings for all dimensions'''
        # Distance
        self.distance_number = distance_number
        self.distance_tile = self.create_distance_tile(distance_min, distance_max, distance_number)

        # Velocity
        self.velocity_number = velocity_number
        self.velocity_tile = self.create_velocity_tile_tile(velocity_min, velocity_max, velocity_number)

        # Height
        self.height_number = height_number
        self.height_tile = self.create_height_tile(height_min, height_max, height_number)

        # Overlap and actions
        self.overlap = overlap
        self.action_number = action_number
        self.total_tiles = self.calculate_total_tiles()

    def create_distance_tile(self, distance_min, distance_max, distance_number):
        '''Creates geomspace distance tiles'''
        # Initialize first tile
        distance_range = distance_max-distance_min
        distance_tile_width = distance_range/distance_number
        geom_offset = 0.1
        initial_distance_tile = np.geomspace(geom_offset, distance_range+distance_tile_width+geom_offset, distance_number+1)
        initial_distance_tile -= (abs(distance_min) + geom_offset)
        distance_tiles = [initial_distance_tile]
        
        # Overlap remaining tiles
        for _ in range(self.overlap-1):
            offset = np.random.uniform(0, distance_tile_width)
            distance_tiles.append(initial_distance_tile-offset)

        return distance_tiles

    def create_velocity_tile(self, velocity_min, velocity_max, velocity_number):
        '''Creates even spaced velocity tiles'''
        # Initialize first tile
        velocity_range = velocity_max-velocity_min
        velocity_tile_width = velocity_range/velocity_number
        initial_velocity_tile = np.linspace(velocity_min, velocity_max+velocity_tile_width, velocity_number+1)
        velocity_tiles = [initial_velocity_tile]

        # Overlap remaining tiles
        for _ in range(self.overlap-1):
            offset = np.random.uniform(0, velocity_tile_width)
            velocity_tiles.append(initial_velocity_tile-offset)

        return velocity_tiles

    def create_height_tile(self, height_min, height_max, height_number):
        '''Creates centred logspaced tiles'''
        pass

    def calculate_total_tiles():
        '''Calculates total number of all tiles'''
        pass
