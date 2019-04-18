from random import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class InitializationError(Exception):
    def __init__(self, message):
        self.message = message

class Particle(object):
    def __init__(self, x, y, z, vx=0, vy=0, vz=0, ax=0, ay=0, az=0, particle_type='reactant'):
        self.x = [x, y, z]                      # position
        self.v = [vx, vy, vz]                   # velocity
        self.a = [ax, ay, az]                   # acceleration
        self.particle_type = particle_type      # records whether the particle is a reactant or product

    def set_velocities(self, vx, vy, vz):
        self.v = [vx, vy, vz]

    def set_accelerations(self, ax, ay, az):
        self.a = [ax, ay, az]

    def __repr__(self):
        return f'Particle - position: ({self.x[0]:.4f}, {self.x[1]:.4f}, {self.x[2]:.4f}) velocities: ({self.v})\n'

class Model(object):
    def __init__(self, n_particles=125, t=10, rho=0.6, b=3, n_steps=1000,
                    r_cut=3.5, dt=0.001, reactivity=0, rxn_cutoff=0.8):
        self.n_particles = n_particles          # number of particles in the simulation
        self.t = t                              # dimensionless system temperature
        self.rho = rho                          # dimensionless system density
        self.n_steps = n_steps                  # number of steps to simulate
        self.radius_cutoff = r_cut              # cutoff radius
        self.dt = dt                            # dimensionless simulation time step
        self.rxn_cutoff = rxn_cutoff            # The maximum radius at which reactions can occur
        self.reactivity = reactivity            # the chance that two particles within the cutoff radius will react
        self.n_reactants = n_particles          # number of reactant particles
        self.n_products = 0                     # number of product particles
        self.volume = self.n_particles/rho      # volume of the simulation cube
        self.cube_length = self.volume**(1/3)   # length of a side of the cube
        self.particles = []
        self.n_rxns = 0                         # number of reactions that have occured
        self.kinetic_energy = 0
        self.potential_energy = 0
        self.virial = 0
        self.nd = 3                             # number of dimensions x, y, z
        self.kinetic_energy_hist = []
        self.potential_energy_hist = []
        self.total_energy = []
        self.virial_hist = []
        self.time = []
        self.rxns_hist = []
        self.particles_hist = []
        self.reactants_hist = []
        self.products_hist = []
        self.step_number = 0
        self.eq_step = 80

        # Check to make sure n_particles^(1/3) is an integer
        if (int(round(self.n_particles**(1/3)))**3 != self.n_particles):
            raise InitializationError('The number of particles must be the cube of an integer')

    def scale_velocities(self):
        '''Scale velocities to maintain a desired temperature.'''
        t = 2/3*self.kinetic_energy/self.n_particles
        scaler = (self.t / t) ** (0.5)
        for p in self.particles:
            for i in range(self.nd):
                p.v[i] = p.v[i]*scaler

    def _update_accelerations_and_react(self):
        '''Update the accelerations'''
        # Zero out the accelerations
        for p in self.particles:
            for i in range(self.nd):
                p.a[i] = 0

        n_particles = self.n_particles # this number will change as rxns occur
        i = 0
        while i < n_particles - 1:
            j = i+1
            while j < n_particles:
                p = self.particles[i]
                n = self.particles[j]
                dist_squared = 0
                d = [0, 0, 0]
                forces = [0, 0, 0]
                # Find distance between particles
                for k in range(self.nd):
                    d[k] = n.x[k] - p.x[k]
                    # Account for minimum image
                    if d[k] > self.cube_length/2:
                        d[k] -= self.cube_length
                    elif d[k] < -self.cube_length/2:
                        d[k] += self.cube_length
                    dist_squared += d[k]*d[k]
                dist = dist_squared**(0.5)
                if dist < self.radius_cutoff:
                    # Calculate potential and force
                    r6 = 1/(dist_squared**3)
                    r12 = r6*r6
                    u = 4*(r12 - r6)
                    force = 24/dist * (2*r12 - r6)
                    for k in range(self.nd):
                        forces[k] = force*d[k]/dist
                        p.a[k] -= forces[k]
                        n.a[k] += forces[k]
                    self.potential_energy += u
                    self.virial += force*dist
                if (dist <= self.rxn_cutoff) and (p.particle_type == 'reactant') \
                                                and (n.particle_type == 'reactant'):
                    if random() < self.reactivity and (self.step_number > self.eq_step):
                        # update first particle
                        p.particle_type = 'product'
                        for k in range(self.nd):
                            # Position does not change to prevent overlap
                            p.v[k] = (p.v[k] + n.v[k])/2
                            p.a[k] = (p.a[k] + n.a[k])/2

                        # remove second particle and update model
                        self.particles.remove(n)
                        self.n_particles -= 1
                        self.n_products += 1
                        self.n_reactants -= 2
                        n_particles -= 1
                        self.n_rxns += 1
                        j -= 1 # we have removed an element, so should step back
                j += 1
            i += 1


    def step(self):
        '''Make a time-step'''

        # Zero out the accumulated properties
        self.kinetic_energy = 0
        self.potential_energy = 0
        self.virial = 0

        # Update the positions and half the velocities
        for p in self.particles:
            for i in range(self.nd):
                new_pos = p.x[i] + p.v[i]*self.dt + 0.5*p.a[i]*self.dt**2
                new_pos = new_pos - self.cube_length*round(new_pos/self.cube_length) # apply periodic boundary conditions
                p.x[i] = new_pos
                p.v[i] = p.v[i] + p.a[i]*self.dt/2

        # Calculate Accelerations and perform reactions
        self._update_accelerations_and_react()

        # finish updating velocities
        for p in self.particles:
            for i in range(self.nd):
                p.v[i] = p.v[i] + 0.5*p.a[i] * self.dt
                self.kinetic_energy += 0.5*p.v[i]**2

        # Update model history
        self.kinetic_energy_hist.append(self.kinetic_energy)
        self.potential_energy_hist.append(self.potential_energy)
        self.virial_hist.append(self.virial)
        self.total_energy.append(self.kinetic_energy + self.potential_energy)
        self.rxns_hist.append(self.n_rxns)
        self.products_hist.append(self.n_products)
        self.reactants_hist.append(self.n_reactants)
        self.particles_hist.append(self.n_particles)
        if self.time:
            self.time.append(self.time[-1]+self.dt)
        else:
            self.time.append(0)

    def initialize(self):
        '''Initialize the simulation with zero net velocity'''
        # Initialize posiitions
        third_particles = round(self.n_particles**(1/3))
        dl = (self.cube_length - 1.6) / third_particles
        if dl < 0.8:
            print(dl)
            raise InitializationError('The system is too dense and causes overlap in initialization')
        self.particles = []
        for i in range(third_particles):
            for j in range(third_particles):
                for k in range(third_particles):
                    x = -0.5*self.cube_length + 0.8 + dl*i
                    y = -0.5*self.cube_length + 0.8 + dl*j
                    z = -0.5*self.cube_length + 0.8 + dl*k
                    new_particle = Particle(x, y, z)
                    self.particles.append(new_particle)

        # Initialize velocities
        v_total = [0, 0, 0]
        v_avg = self.t**0.5
        for p in self.particles:
            for i in range(self.nd):
                if random() >= 0.5:
                    p.v[i] = v_avg
                else:
                    p.v[i] = -v_avg
                v_total[i] += p.v[i]

        # Scale the net velocities to per particle
        v_total = [v/self.n_particles for v in v_total]

        # Subtract net velocity from each particle
        for p in self.particles:
            for i in range(self.nd):
                p.v[i] = p.v[i] - v_total[i]

        # Double check for net velocity
        v_total = [0, 0, 0]
        for p in self.particles:
            for i in range(self.nd):
                v_total[i] += p.v[i]

        print(f"Net velocity: {v_total}")


    def run(self, plot=False, saved_plot_path=''):
        '''Run the simulation'''
        # Keeps vars in scope
        fig = 0
        ax = 0

        if plot:
            fig = plt.figure()
            plt.ion()
            plt.show()

        for i in range(self.n_steps):
            self.step_number = i
            print(f'Step {self.step_number}')
            self.step()
            if self.step_number > self.eq_step:
                self.scale_velocities()
            # Update dynamic plots every now and then
            plt.clf()
            ax = fig.add_subplot(311, projection='3d')
            ax.scatter(
                [p.x[0] for p in self.particles],
                [p.x[1] for p in self.particles],
                [p.x[2] for p in self.particles],
                c=['b' if p.particle_type == 'reactant' else 'r' for p in self.particles]
            )
            ax.set_xlabel('X')
            ax.set_xlim3d(-0.5*self.cube_length, 0.5*self.cube_length)
            ax.set_ylim3d(-0.5*self.cube_length, 0.5*self.cube_length)
            ax.set_zlim3d(-0.5*self.cube_length, 0.5*self.cube_length)
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax = fig.add_subplot(312)
            plt.plot(self.time, self.kinetic_energy_hist, label='Kinetic energy')
            plt.plot(self.time, self.potential_energy_hist, label='Potential energy')
            # plt.plot(self.time, self.virial_hist, label='Virial')
            plt.plot(self.time, self.total_energy, label='Total energy')
            plt.legend()
            plt.grid()
            ax = fig.add_subplot(313)
            plt.plot(self.time, self.rxns_hist, label='Reactions')
            plt.plot(self.time, self.reactants_hist, label='Reactants')
            plt.plot(self.time, self.products_hist, label='Products')
            plt.plot(self.time, self.particles_hist, label='Total particles')
            plt.legend()
            plt.grid()

            plt.draw()
            plt.pause(0.05)
        if saved_plot_path:
            plt.savefig(saved_plot_path)

    def __repr__(self):
        return ''.join([p.__repr__() for p in self.particles])


if __name__ == "__main__":
    m = Model(n_steps=1000, reactivity=0.7, t=10)
    m.initialize()
    m.run(plot=True, saved_plot_path='run_log.png')
