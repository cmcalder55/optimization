#%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import random
from IPython.display import clear_output
import math

class Genetic_rep:
  def __init__(self):
    self.fitness_score = 0
    # l = 0: green space         # l = 1: residential-type-1        # l = 2: residential-type-2 
    # l = 3: residential-type-3   # l = 4: commercial-type-1         # l = 5 : commercial-type-2
    # l = 6: commercial-type-3     .......
    #sample land codes from 8 differnet land codes/zone type with distribution proportional to weights, thus preserving zone ratios
    self.land_type = numpy.array([random.choices([0, 1, 2, 3, 4, 5, 6, 7], 
                                                 weights=(20, 10, 30, 40, 50, 20, 20, 15), k=28*8)]).reshape(28,8) 
    self.building_level = numpy.zeros(28*8).reshape(28,8)

    # set building height based on
    for row in range(0,self.land_type.shape[0]):
      for col in range(0,self.land_type.shape[1]):
        if self.land_type[row,col] == 0: #if green space, number of floors = 0
          self.building_level[row,col] = 0 
        if self.land_type[row,col] == 2: #historic district--restrict floors to 3 to 5
          self.building_level[row,col] = random.choices([3,4,5])[0]
        else:
          self.building_level[row,col] = random.choices([0,1,2,3,4,5,6,7,8,13,18,20], 
                                                        weights=(20, 10, 30, 40, 50, 20, 20, 15,10,3,2,2))[0]
                                                         # ^probabilty of getting tall building is  low

    self.chromosome = numpy.stack((self.land_type, self.building_level), axis=2)

  def print_all_genes(self):
    for row in range(0,self.chromosome.shape[0]):
      for col in range(0,self.chromosome.shape[1]):
        print(self.chromosome[row, col, :])

  def mutate_genes(self):
    for row in range(0,self.chromosome.shape[0]):
      for col in range(0,self.chromosome.shape[1]):
       if random.random() < 0.1:
         gene = self.chromosome[row, col, :]
         if gene[0,] == 0: #if green space, number of floors = 0
          self.building_level[row,col] = 0 
         if gene[0,] == 2: #historic district--restrict floors to 3 to 5
          self.building_level[row,col] = random.choices([3,4,5])[0]
         else:
          self.building_level[row,col] = random.choices([0,1,2,3,4,5,6,7,8,13,18,20], 
                                                        weights=(20, 10, 30, 40, 50, 20, 20, 15,10,3,2,2))[0]


  def fitness_eval(self):
    #proximity to designated positive/negative factors --- we assume every floor of every type of zone has certain number of people 
    #who have exposure to different factor that affects quality of life 
    #eg. Residitial zones have the 10 people per floor 
    residitial_zones_have_people_per_floor = 10
    #eg. Commercial zones -- 15 people per floor etc.
    commercial_zones_have_people_per_floor = 15

    #Environmental
    ################################# average exposure to flooding #######################################
    dist1 = []
    loss_of_qol_due_to_flooding_constant = 1.03  #long term affects -- so low value 
    for i in range(0, self.chromosome.shape[0]):
      for j in range(0, self.chromosome.shape[1]):
        land_zone_type = self.chromosome[i,j,1]
        x2 = i 
        y2 = j 
        x1 = i 
        y1 = 8 # hudson is along y = 8
        if (land_zone_type == 1 or land_zone_type == 2 or land_zone_type == 3):    
          dist1.append(math.hypot(x2 - x1, y2 - y1))
        if (land_zone_type == 4 or land_zone_type == 5 or land_zone_type == 6): 
          dist1.append(commercial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
    avg_flooding_fittness = -loss_of_qol_due_to_flooding_constant * sum(dist1)/len(dist1)
          
    ############################# distance to water, river front walkway, views ###############################
    dist2 = []
    gain_of_qol_due_to_living_close_to_rf = 1.066 #immediate affects -- high value 
    for i in range(0, self.chromosome.shape[0]):
      for j in range(0, self.chromosome.shape[1]):
        land_zone_type = self.chromosome[i,j,1]
        x2 = i 
        y2 = j 
        x1 = i 
        y1 = 8 # hudson is along y = 8
        if (land_zone_type == 1 or land_zone_type == 2 or land_zone_type == 3):   
          dist2.append(residitial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
        if (land_zone_type == 4 or land_zone_type == 5 or land_zone_type == 6): 
          dist2.append(commercial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
    avg_dist_to_rf_fittness = gain_of_qol_due_to_living_close_to_rf * sum(dist2)/len(dist2)

    #Social
    #proximity to entertainment
    #distance to schools
    #distance to hospitals/fire/police  

    ############################ distance to public transportation ######################################
    dist3 = []
    constant_exp_mny = 1.1 # increased exposure to metropolitan ny  
    for i in range(0, self.chromosome.shape[0]):
      for j in range(0, self.chromosome.shape[1]):
        land_zone_type = self.chromosome[i,j,1]
        x2 = i 
        y2 = j 
        x1 = 28 
        y1 = 8 # Hoboken terminal is the south east corner of the city 
        if (land_zone_type == 1 or land_zone_type == 2 or land_zone_type == 3): 
          dist3.append(residitial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
        if (land_zone_type == 4 or land_zone_type == 5 or land_zone_type == 6): 
          dist3.append(commercial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
    dist_public_transtp = constant_exp_mny * sum(dist3)/len(dist3)
 
    ##################################### proximity to greenspace ###################################
    dist4 = []
    for i in range(0, self.chromosome.shape[0]):
      for j in range(0, self.chromosome.shape[1]):
        land_zone_type = self.chromosome[i,j,1]
        if (land_zone_type == 0): 
          for i2 in range(0, self.chromosome.shape[0]):
            for j2 in range(0, self.chromosome.shape[1]):
              land_zone_type = self.chromosome[i2,j2,1]
              x2 = i2 
              y2 = j2 
              x1 = i
              y1 = j 
              if (land_zone_type == 1 or land_zone_type == 2 or land_zone_type == 3): 
                dist4.append(residitial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
              if (land_zone_type == 4 or land_zone_type == 5 or land_zone_type == 6): 
                dist4.append(commercial_zones_have_people_per_floor * math.hypot(x2 - x1, y2 - y1))
    proximity_to_green_space = 1.1 * sum(dist4)/len(dist4)

    #average density of greenspace
    #Economic
    #business density
    #proximity to high foot traffic (PATH station, university, etc)
    
    
    list_fitness_parameters = [dist_public_transtp,  avg_dist_to_rf_fittness,  proximity_to_green_space, avg_flooding_fittness]
    fitness_score = sum(list_fitness_parameters)/len(list_fitness_parameters)
    self.fitness_score = fitness_score
    return [fitness_score, list_fitness_parameters]

                                                        

  def plot_chromosome(self, clear_output_set = True):
    
    mpl.rcParams['figure.dpi'] = 200

    data = self.chromosome[:,:,0]
    building_heights = self.chromosome[:,:,1].flatten()
    xpos, ypos = numpy.indices(data.shape) 
    xpos = xpos.flatten()   
    ypos = ypos.flatten()
    zpos = numpy.zeros(xpos.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_box_aspect(1,1,1)
    ax.set_xlim3d(-1, 29)
    ax.set_ylim3d(-1, 9)
    ax.set_zlim3d(0, 50)

    colors = plt.cm.jet(data.flatten()/float(data.max()))
    ax.bar3d(xpos,ypos,zpos, 1,0.5,building_heights, color=colors) #uniform blocks of sides of ratio 2:1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    if clear_output_set == True:
      clear_output(wait=True)

  def mutate_chromosomes(self):
    rn = random.random()
    if rn > 0.5:
      #type 1 crossover 
      chunk1 = self.chromosome[:14,:,:]
      chunk2 = self.chromosome[14:,:,:]
      self.chromosome[:14,:,:] = chunk2
      self.chromosome[14:,:,:] = chunk1

    #type 2 crossover 
    if rn < 0.5:
      chunk1 = self.chromosome[:,:4,:]
      chunk2 = self.chromosome[:,4:,:]
      self.chromosome[:,:4,:] = chunk2
      self.chromosome[:,4:,:] = chunk1

    #type 3 crossover 
    #rand_rows = random.sample([item for item in range(0, 28)], k=int(random.random()*28))
    


class Genetic_algo():
  def __init__(self, init_population_size = 1000):
    self.population = [Genetic_rep() for _ in range(0, init_population_size)]
    self.pop_fit_list = []
    self.temp_child = Genetic_rep()

  def crossover(self, parent1, parent2):
    child = Genetic_rep()
    rn = random.random() 
    if rn < 0.5:
      #type 1 crossover 
      chunk1 = parent1.chromosome[:14,:,:]
      chunk2 = parent2.chromosome[14:,:,:]
      child.chromosome = numpy.concatenate((chunk1, chunk2), axis=0)
      return child

    if rn > 0.5:
      #type 2 crossover 
      chunk1 = parent1.chromosome[:,:4,:]
      chunk2 = parent2.chromosome[:,4:,:]
      child.chromosome = numpy.concatenate((chunk1, chunk2), axis=1)
      return child

  
  def crossover_parent_selection(self, pop_list):
    # return top 20 percent parents with highest fitness 
    pop_list.sort(key=lambda x: x.fitness_score, reverse=True)
    print(int((PERCENT_NEW_CHILDREN * 2) * len(pop_list)))
    top_20_percent_pop_list = pop_list[:int((PERCENT_NEW_CHILDREN * 2) * len(pop_list))]
    return top_20_percent_pop_list
 
  def plot_evolution(self, pop_fit_list, pop_size_list):
    mpl.rcParams['figure.dpi'] = 100

    plt.figure(2)
    plt.clf()
    plt.title("Evolution Graph")
    plt.xlabel("Generation")
    plt.ylabel("Population Size / Fitness")
    plt.plot(pop_fit_list)
    plt.plot(pop_size_list)
    plt.pause(0.001)
    #print("Episode:", len(values),", Moving Average:", ma[-1] )
    clear_output(wait=True)   

  def start_evolution(self, gen=10):

    pop_fit_list = []
    pop_size_list = []
    for i in range(0,gen):

      #sort, select parents, maybe mutate
      for individual in self.population:
          individual.fitness_eval()
      
      #random mutations 
      for individual in self.population[5:]: # Leave out top 5 from mutation 
        if random.random() < PROB_MUTATION_OCCURING:
          individual.mutate_genes()
      for individual in self.population[5:]:
        if random.random() < PROB_MUTATION_OCCURING:
          individual.mutate_chromosomes()    


      top_20 = self.crossover_parent_selection(self.population)
      
      #crossover, create children increase population size by 10 per
      for i in range(1,len(top_20)):  
        child = self.crossover(top_20[i-1], top_20[i])
        child.fitness_eval()
        self.population.append(child)

      self.population.sort(key=lambda x: x.fitness_score, reverse=True)  
        
      #evaluate fitness, sort, reduce by 5 per

      self.population = self.population[:int(len(self.population) - ((PERCENT_NEW_CHILDREN/2) * len(self.population)))]
    
      print(self.population[0].fitness_score)
      print(len(self.population))
       
      pop_fit_list.append(self.population[0].fitness_score) 
      pop_size_list.append(len(self.population))

      self.plot_evolution(pop_fit_list, pop_size_list)
      #repeat for i generations

    return [pop_fit_list, pop_size_list]
  
PROB_MUTATION_OCCURING = 0.5
PERCENT_NEW_CHILDREN = 0.1
  
if __name__=="__main__":
  
  evol1 = Genetic_algo(init_population_size=100)
  evol_prop = evol1.start_evolution(gen=10)
  print(evol_prop[0])
  print(evol_prop[1])     
  
  #Plot cityscape for best solution
  evol1.population[0].plot_chromosome()

