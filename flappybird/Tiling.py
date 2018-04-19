import numpy as np

class Tiling(object):
    def __init__(self, distance_min, distance_max, velocity_min, velocity_max, height_min, height_max, distance_number=15, velocity_number=13, height_number=10, overlap=5, action_number=2):
        '''Creates tilings for all dimensions'''
        # Overlap and actions
        self.overlap = overlap
        self.action_number = action_number

        # Distance
        self.distance_number = distance_number
        self.distance_tiles = self.create_distance_tile(distance_min, distance_max, distance_number)

        # Velocity
        self.velocity_number = velocity_number
        self.velocity_tiles = self.create_velocity_tile(velocity_min, velocity_max, velocity_number)

        # Height
        self.height_number = height_number
        self.height_tiles = self.create_height_tile(height_min, height_max, height_number)

        # Total tiles
        self.total_tiles = self.calculate_total_tiles()

    def create_distance_tile(self, distance_min, distance_max, distance_number):
        '''Creates geomspace distance tiles - output is nd array with element for each overlapping tile, each holding tile range with offset [[[]]]'''
        # Initialize first tile
        distance_range = distance_max-distance_min
        distance_tile_width = (distance_range/distance_number)/4
        geom_offset = 1
        initial_distance_tile = np.geomspace(geom_offset, distance_range+distance_tile_width+geom_offset, distance_number+1)
        initial_distance_tile -= (abs(distance_min) + geom_offset)
        distance_tiles = [initial_distance_tile]
        
        # Overlap remaining tiles
        for _ in range(self.overlap-1):
            offset = np.random.uniform(0, distance_tile_width)
            distance_tiles.append(initial_distance_tile-offset)

        return distance_tiles

    def create_velocity_tile(self, velocity_min, velocity_max, velocity_number):
        '''Creates even spaced velocity tiles - output is nd array with element for each overlapping tile, each holding tile range with offset [[[]]]'''
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
        '''Creates centred logspaced tiles - output is nd array with element for each overlapping tile, each holding tile range with offset [[[]]]'''
        # Initialize first tile
        height_range = height_max - height_min
        height_tile_width = (height_range/height_number)/2
        geom_offset = 1
        height_pos_spacing = np.geomspace(geom_offset, height_max+height_tile_width+geom_offset, height_number+1)
        height_pos_spacing -= geom_offset
        height_neg_spacing = np.geomspace(geom_offset, abs(height_min)+height_tile_width+geom_offset, height_number+1)
        height_neg_spacing = np.flip(((height_neg_spacing-geom_offset)*-1),0)
        initial_height_tile = np.unique(np.concatenate((height_neg_spacing, height_pos_spacing)))
        height_tiles = [initial_height_tile]

        # Overlap remaining tile
        for _ in range(self.overlap-1):
            offset = np.random.uniform(0, height_tile_width)
            height_tiles.append(initial_height_tile-offset)

        return height_tiles

    def calculate_total_tiles(self):
        '''Calculates total number of all tiles'''
        # Add all individual tiles and multiply by overlap and actions
        total_distance_tiles = self.distance_number+1
        total_height_tiles = self.distance_number+1
        total_velocity_tiles = self.velocity_number+1
        total = total_distance_tiles * total_height_tiles * total_velocity_tiles
        total *= self.overlap
        total *= self.action_number
        return total

    def get_features(self, distance, velocity, height):
        '''Returns on features for current state - triplestore array for distance, velocity and height'''
        distance_features = []
        velocity_features = []
        height_features = []

        for i in range(self.overlap):
            # Find on feature for each overlapping tile
            distance_tile = (np.digitize(distance, self.distance_tiles[i])).item()
            velocity_tile = (np.digitize(velocity, self.velocity_tiles[i])).item()
            height_tile = (np.digitize(height, self.height_tiles[i])).item()

            # Append to tile arrays
            distance_features.append(distance_tile)
            velocity_features.append(velocity_tile)
            height_features.append(height_tile)

        all_features = (distance_features, velocity_features, height_features)
        return all_features

    def get_indices(self, state, action):
        '''Takes the 'on' features and returns the indices of them'''
        if action == None:
            action_index = 0
        else:
            action_index = 1
        distance = state[0]
        velocity = state[1]
        height = state[2]
        F = self.get_features(distance, velocity, height)

        indices = []

        for i in range(self.overlap):
            d_index = F[0][i]
            v_index = F[1][i]
            h_index = F[2][i]

            index = int( \
                  (action_index * (self.total_tiles / self.action_number)) \
                + (i * self.distance_number * self.velocity_number * self.height_number) \
                + (d_index) \
                + (v_index * self.distance_number) \
                + (h_index * self.distance_number * self.velocity_number) )

            indices.append(index)

        return indices