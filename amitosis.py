import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Populations(object):

    def __init__(self, nReps, N, nLoci, ploidy, genomic_mu, selcoef, amitosis):
        '''Construct a set of populations consisting of unmutated individuals. 
        Each individual has germline and somatic genomes like Tetrahymena.  

        Mutation rates per site are identical between the genomes. Mutations may
        be beneficial or deleterious but have equal effects.

        Parameters
        ----------
        nReps : int
            Number of replicate populations.
        N : int
            Size of each population.
        nLoci : int
            Number of fitness loci.
        ploidy : int
            Ploidy of somatic genome.  (Germline genome is assumed to be diploid.)
        genomic_mu : float
            Mutation rate per genome per generation. 
        selcoef : float
            Selection coefficient of each mutation.  Positive and negative
            values represent beneficial and deleterious mutations, respectively.
        amitosis : bool
            Whether the soma reproduces by amitosis; if False, it reproduces by mitosis.
        '''
        self.nReps = nReps
        self.N = N
        self.nLoci = nLoci
        self.soma = np.zeros((self.nReps, self.N, self.nLoci), dtype='int')
        self.germ = np.zeros((self.nReps, self.N, self.nLoci), dtype='int')
        self.ploidy = ploidy
        self.genomic_mu = genomic_mu
        self.mu = self.genomic_mu / (nLoci * ploidy)
        self.selcoef = selcoef
        self.generation = 0
        self.amitosis = amitosis
        self.get_fitness()
        self.fitness_mean = {}
        self.fitness_std = {}
        self.collect_data()  

    def get_fitness(self):
        '''Calculate the fitness of each individual in each population.
        '''
        s = (self.soma / self.ploidy) * self.selcoef
        self.fitness = np.prod(1 + s, axis =2)
        self.relative_fitness = self.fitness / np.expand_dims(np.sum(self.fitness, axis=1), axis=1)
        
    def mutate(self):
        '''Mutate each locus in each genome in each individual in each population.
        '''
        self.soma += np.random.binomial(self.ploidy - self.soma, self.mu)
        self.germ += np.random.binomial(2 - self.germ, self.mu)
    
    def select(self):
        '''Sample N individuals with replacement with probability proportional
        to their fitness to form the next generation.
        '''
        self.get_fitness()         
        cumsum_fit = np.cumsum(self.relative_fitness, axis=1)
        random_vals = np.random.random((self.nReps, self.N))
        self.selected = np.array(list(map(np.searchsorted, cumsum_fit, random_vals)))
    
    def reproduce(self):
        '''Allow sampled individuals to reproduce asexually.  Germline genome
        reproduces mitotically.  Somatic genome reproduces amitotically or mitotically.
        '''
        rReps = np.ones((self.nReps, self.N), dtype='int') * np.expand_dims(np.arange(self.nReps), axis=1)
        self.germ = self.germ[rReps, self.selected, ]
        if self.amitosis:
            wildtype = (self.ploidy - self.soma[rReps, self.selected, ]) * 2
            mutant = self.soma[rReps, self.selected, ] * 2
            self.soma = np.random.hypergeometric(mutant, wildtype, self.ploidy)
        else:
            # mitosis in soma
            self.soma = self.soma[rReps, self.selected, ]

    def get_next_generation(self):
        '''Take each population through a full generation of a life cycle
        comprising mutation, selection, and reproduction.
        '''
        self.mutate()
        self.select()
        self.reproduce()
        self.generation += 1

    def collect_data(self):
        fitness = self.fitness.mean(axis=1)
        self.fitness_mean.update({self.generation: fitness.mean()})
        self.fitness_std.update({self.generation: fitness.std(ddof=1)})

    def evolve(self, ngenerations, interval=1):
        '''Evolve each population for a certain number of generations.

        Parameters
        ----------
        ngenerations : int
            Number of generations of evolution.
        interval : int
            Interval between data collection in generations.
        '''
        for generation in range(ngenerations):
            self.get_next_generation()
            if (self.generation % interval) == 0:    
                self.collect_data()
        data = pd.DataFrame({'fitness_mean': self.fitness_mean, 'fitness_std': self.fitness_std})
        data.reset_index(inplace=True)
        data = data.rename(columns = {'index':'generation'})
        data['nReps'] = self.nReps
        data['N'] = self.N
        data['nLoci'] = self.nLoci
        data['ploidy'] = self.ploidy
        data['genomic_mu'] = self.genomic_mu
        data['selcoef'] = self.selcoef
        data['amitosis'] = self.amitosis
        return data
