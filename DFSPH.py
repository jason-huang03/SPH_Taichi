import taichi as ti
from sph_base import SPHBase
from particle_system import Container3d


class DFSPHSolver(SPHBase):
    def __init__(self, particle_system: Container3d):
        super().__init__(particle_system)
        
        self.surface_tension = 0.01
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05
    

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.container.particle_volumes[p_j] * self.cubic_kernel(R_mod)


    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_densities[p_i] = self.container.particle_volumes[p_i] * self.cubic_kernel(0.0)
            density_i = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_task, density_i)
            self.container.particle_densities[p_i] += density_i
            self.container.particle_densities[p_i] *= self.density_0
    

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        
        ############## Surface Tension ###############
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            diameter2 = self.container.particle_diameter * self.container.particle_diameter
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = R.dot(R)
            if R2 > diameter2:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.cubic_kernel(R.norm())
            else:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.cubic_kernel(ti.Vector([self.container.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.container.dim + 2)
        pos_j = self.container.particle_positions[p_j]
        # Compute the viscosity force contribution
        R = pos_i - pos_j
        v_xy = (self.container.particle_velocities[p_i] -
                self.container.particle_velocities[p_j]).dot(R)
        
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            f_v = d * self.viscosity * (self.container.particle_masses[p_j] / (self.container.particle_densities[p_j])) * v_xy / (
                R.norm()**2 + 0.01 * self.container.dh**2) * self.cubic_kernel_derivative(R)
            ret += f_v
  

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.container.particle_num[None]):
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.container.particle_accelerations[p_i] = d_v
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.container.particle_accelerations[p_i] = d_v
    

    @ti.kernel
    def update_position(self):
        # Update position
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
    

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.container.dim)])
            
            ret = ti.Vector([0.0, 0.0, 0.0, 0.0])
            
            self.container.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)
            
            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.container.particle_dfsph_alphas[p_i] = factor
            

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.container.particle_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
            ret[3] += grad_p_j.norm_sqr() # sum_grad_p_k
            for i in ti.static(range(3)): # grad_p_i
                ret[i] += grad_p_j[i]
    

    @ti.kernel
    def compute_density_change(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            self.container.for_all_neighbors(p_i, self.compute_density_change_task, ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if self.container.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0
     
            self.container.particle_densities_derivatives[p_i] = density_adv


    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        ret.density_adv += self.container.particle_volumes[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j]))
 
        # Compute the number of neighbors
        ret.num_neighbors += 1
    

    @ti.kernel
    def compute_density_star(self):
        for p_i in range(self.container.particle_num[None]):
            delta = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_star_task, delta)
            density_adv = self.container.particle_densities[p_i] /self.density_0 + self.dt[None] * delta
            self.container.particle_densities_star[p_i] = ti.max(density_adv, 1.0)


    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            ret += self.container.particle_volumes[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j]))
  

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.density_0 * self.container.particle_densities_star[idx_i] - offset
        return density_error / self.container.particle_num[None]

    @ti.kernel
    def compute_density_derivative_error(self, offset: float) -> float:
        density_error = 0.0
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.density_0 * self.container.particle_densities_derivatives[idx_i] - offset
        return density_error / self.container.particle_num[None]

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in range(self.container.particle_num[None]):
            if self.container.particle_materials[I] == self.container.material_fluid:
                field[I] *= time_step

    @ti.kernel
    def compute_kappa_v(self):
        for idx_i in range(self.container.particle_num[None]):
            self.container.particle_dfsph_kappa_v[idx_i] = self.container.particle_densities_derivatives[idx_i] * self.container.particle_dfsph_alphas[idx_i]

            
    def correct_divergence_error(self):
        # TODO: warm start 
        # Compute velocity of density change
        self.compute_density_change()
        inv_dt = 1 / self.dt[None]
        self.multiply_time_step(self.container.particle_dfsph_alphas, inv_dt)

        m_iterations_v = 0
        
        # Start solver
        avg_density_err = 0.0

        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = 1.0 / self.dt[None] * self.max_error_V * 0.01 * self.density_0
            # print("eta ", eta)
            if avg_density_err <= eta:
                break
            m_iterations_v += 1
        print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_density_err}")

        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

        self.multiply_time_step(self.container.particle_dfsph_alphas, self.dt[None])


    def divergence_solver_iteration(self):
        self.compute_kappa_v()
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_derivative_error(0.0)
        return density_err


    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            # evaluate rhs
            # b_i = self.container.particle_densities_derivatives[p_i]
            # k_i = b_i*self.container.particle_dfsph_alphas[p_i]
            k_i = self.container.particle_dfsph_kappa_v[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.container.dim)]), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            self.container.for_all_neighbors(p_i, self.divergence_solver_iteration_task, ret)
            self.container.particle_velocities[p_i] += ret.dv
        
    
    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            # b_j = self.container.particle_densities_derivatives[p_j]
            # k_j = b_j * self.container.particle_dfsph_alphas[p_j]
            k_j = self.container.particle_dfsph_kappa_v[p_j]
            k_i = ret.k_i
            k_sum = k_i + self.density_0 / self.density_0 * k_j  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.container.particle_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                ret.dv -= self.dt[None] * grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0
    
    @ti.kernel
    def compute_kappa(self):
        for idx_i in range(self.container.particle_num[None]):
            self.container.particle_dfsph_kappa[idx_i] = (self.container.particle_densities_star[idx_i] - 1.0) * self.container.particle_dfsph_alphas[idx_i]
    def correct_density_error(self):
        inv_dt2 = 1 / (self.dt[None] * self.dt[None])

        # TODO: warm start
        
        # Compute rho_adv
        self.compute_density_star()

        self.multiply_time_step(self.container.particle_dfsph_alphas, inv_dt2)

        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            if avg_density_err <= eta:
                break
            m_iterations += 1
        print(f"DFSPH - iterations: {m_iterations} Avg density Err: {avg_density_err:.4f}")
        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v
    
    def pressure_solve_iteration(self):
        self.compute_kappa()
        self.pressure_solve_iteration_kernel()
        self.compute_density_star()
        density_err = self.compute_density_error(self.density_0)
        return density_err 

    
    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            # Evaluate rhs
            # b_i = self.container.particle_densities_star[p_i] - 1.0
            # k_i = b_i * self.container.particle_dfsph_alphas[p_i]
            k_i = self.container.particle_dfsph_kappa[p_i]

            # TODO: if warmstart
            # get kappa V
            self.container.for_all_neighbors(p_i, self.pressure_solve_iteration_task, k_i)
    

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            # b_j = self.container.particle_densities_star[p_j] - 1.0
            # k_j = b_j * self.container.particle_dfsph_alphas[p_j]
            k_j = self.container.particle_dfsph_kappa[p_j]
            k_sum = k_i +  k_j 
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.container.particle_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                # Directly update velocities instead of storing pressure accelerations
                self.container.particle_velocities[p_i] -= self.dt[None] * grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0


    @ti.kernel
    def update_velocities(self):
        # compute new velocities only considering non-pressure forces
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    def step(self):

        self.compute_non_pressure_forces()
        self.update_velocities()
        self.correct_density_error()
        self.update_position()
        self.enforce_boundary_3D(self.container.material_fluid)

        self.container.initialize_particle_system()
        self.compute_density()
        self.compute_DFSPH_factor()
        self.correct_divergence_error()