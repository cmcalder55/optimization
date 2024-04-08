import csv
# from argparse import ArgumentParser


class Node:
    def __init__(self,
                 accessible: bool = True,
                 avg_flooding: float = 0,
                 social_type: str = None,
                 greenspace_density: float = 0):
        self.accessible = accessible    # Indicates whether the node can be changed; ie, whether it should be changed or ignored
        self.avg_flooding = avg_flooding  # Flooding level for node
        self.social_type = social_type  # This can be school, entertainment, pub_transport, hospital, fire, police, etc
        self.greenspace_density = greenspace_density  # Density of greenspace at location

    def __str__(self):
        return f'{self.accessible}-{self.avg_flooding}-{self.social_type}-{self.greenspace_density}'

    def from_str(self, data: str):
        attrib = data.split('-')
        self.avg_flooding = attrib[0]
        self.social_type = attrib[1]
        self.greenspace_density = attrib[2]


class Region:
    def __init__(self, grid_shape=(19, 16)):
        self.map = [[[Node()] for i in range(grid_shape[1])] # initialize all nodes
                    for j in range(grid_shape[0])]
        self.none = self.map
        self.shore = list()
        self.inaccessible = list()
        
        self.social_types = {'school': list(), 'entertainment': list(), 'hospital': list(), 
                             'fire_station': list(), 'police': list()}
        self.env_types = {'park': list(), 'waterfront': list(), 'flooding': list(), 'recreation': list()}
        self.comm_types = {'food': list(), 'retail': list() }

    def __str__(self):
        return str([[i[0].__str__() for i in j] for j in self.map])

    def __getslice__(self, i, j):
        return self.map[i, j][0]

    def update_social_type(self):
        self.social_types = {'school': list(), 'entertainment': list(), 'hospital': list(), 
                             'fire_station': list(), 'police': list()}
        for node in self.map:
            self.social_types[node[0].social_type].append(node)

    def from_csv(self, fpath):
        with open(fpath, 'r') as f:
            csv_reader = csv.reader(f)
            data = []
            for row in csv_reader:
                for cell in row:
                    data.append([Node().from_str(cell)])
        self.map = data

    def to_csv(self, fpath):
        with open(fpath, 'w') as file:
            csv_writer = csv.writer(file)
            for row in self.map:
                csv_writer.writerow([cell[0] for cell in row])

    def score(self):
        # TODO
        pass


if __name__ == '__main__':
    pop_size = 10
    possible_hobokens = [Region() for i in range(pop_size)]
    
    # for hoboken in possible_hobokens:
    #     hoboken.to_csv('hoboken.csv')
    
    print(Region())

