import taichi as ti
from sph_base import SPHBase


class IISPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.a_ii = ti.field(dtype=float, shape=self.container.particle_max_num)
        self.density_deviation = ti.field(dtype=float, shape=self.container.particle_max_num)
        self.last_pressure = ti.field(dtype=float, shape=self.container.particle_max_num)
        self.avg_density_error = ti.field(dtype=float, shape=())

        self.container.particle_accelerations = ti.Vector.field(self.container.dim, dtype=float)
        self.pressure_accel = ti.Vector.field(self.container.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.container.particle_max_num)
        particle_node.place(self.container.particle_accelerations, self.pressure_accel)
        self.dt[None] = 2e-4

    @ti.kernel
    def predict_advection(self):
        # Compute a_ii
        for p_i in range(self.container.particle_num[None]):
            x_i = self.container.particle_positions[p_i]
            sum_neighbor = 0.0
            sum_neighbor_of_neighbor = 0.0
            m_Vi = self.container.particle_original_volumes[p_i]
            density_i = self.container.particle_densities[p_i]
            density_i2 = density_i * density_i
            density_02 = self.density_0 * self.density_0
            self.a_ii[p_i] = 0.0
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                sum_neighbor_inner = ti.Vector([0.0 for _ in range(self.container.dim)])
                for k in range(self.container.fluid_neighbors_num[p_i]):
                    density_k = self.container.particle_densities[k]
                    density_k2 = density_k * density_k
                    p_k = self.container.fluid_neighbors[p_i, j]
                    x_k = self.container.particle_positions[p_k]
                    sum_neighbor_inner += self.container.particle_original_volumes[p_k] * self.cubic_kernel_derivative(x_i - x_k) / density_k2

                kernel_grad_ij = self.cubic_kernel_derivative(x_i - x_j)
                sum_neighbor -= (self.container.particle_original_volumes[p_j] * sum_neighbor_inner).dot(kernel_grad_ij)

                sum_neighbor_of_neighbor -= (self.container.particle_original_volumes[p_j] * kernel_grad_ij).dot(kernel_grad_ij)
            sum_neighbor_of_neighbor *= m_Vi / density_i2
            self.a_ii[p_i] += (sum_neighbor + sum_neighbor_of_neighbor) * self.dt[None] * self.dt[None] * density_02

            # Boundary neighbors
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                sum_neighbor_inner = ti.Vector([0.0 for _ in range(self.container.dim)])
                for k in range(self.container.solid_neighbors_num[p_i]):
                    density_k = self.container.particle_densities[k]
                    density_k2 = density_k * density_k
                    p_k = self.container.solid_neighbors[p_i, j]
                    x_k = self.container.particle_positions[p_k]
                    sum_neighbor_inner += self.container.particle_original_volumes[p_k] * self.cubic_kernel_derivative(x_i - x_k) / density_k2

                kernel_grad_ij = self.cubic_kernel_derivative(x_i - x_j)
                sum_neighbor -= (self.container.particle_original_volumes[p_j] * sum_neighbor_inner).dot(kernel_grad_ij)

                sum_neighbor_of_neighbor -= (self.container.particle_original_volumes[p_j] * kernel_grad_ij).dot(kernel_grad_ij)
            sum_neighbor_of_neighbor *= m_Vi / density_i2
            self.a_ii[p_i] += (sum_neighbor + sum_neighbor_of_neighbor) * self.dt[None] * self.dt[None] * density_02

        # Compute source term (i.e., density deviation)
        # Compute the predicted v^star
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

        for p_i in range(self.container.particle_num[None]):
            x_i = self.container.particle_positions[p_i]
            density_i = self.container.particle_densities[p_i]
            divergence = 0.0
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                divergence += self.container.particle_original_volumes[p_j] * (self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))

            # Boundary neighbors
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                divergence += self.container.particle_original_volumes[p_j] * (self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))

            self.density_deviation[p_i] = self.density_0 - density_i - self.dt[None] * divergence * self.density_0

        # Clear all pressures
        for p_i in range(self.container.particle_num[None]):
            # self.last_pressure[p_i] = 0.0
            # self.ps.pressure[p_i] = 0.0
            self.last_pressure[p_i] = 0.5 * self.container.particle_pressures[p_i]

    def pressure_solve(self):
        iteration = 0
        while iteration < 1000:
            self.avg_density_error[None] = 0.0
            self.pressure_solve_iteration()
            iteration += 1
            if iteration % 100 == 0:
                print(f'iter {iteration}, density err {self.avg_density_error[None]}')
            if self.avg_density_error[None] < 1e-3:
                # print(f'Stop criterion satisfied at iter {iteration}, density err {self.avg_density_error[None]}')
                break

    @ti.kernel
    def pressure_solve_iteration(self):
        omega = 0.5
        # Compute pressure acceleration
        for p_i in range(self.container.particle_num[None]):
            # if self.ps.material[p_i] != self.ps.material_fluid:
            #     self.pressure_accel[p_i].fill(0)
            #     continue
            x_i = self.container.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.container.dim)])

            dpi = self.last_pressure[p_i] / self.container.particle_densities[p_i] ** 2
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                dpj = self.last_pressure[p_j] / self.container.particle_densities[p_j] ** 2
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.container.particle_original_volumes[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            # Boundary neighbors
            dpj = self.last_pressure[p_i] / self.density_0 ** 2
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.container.particle_original_volumes[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)
            self.pressure_accel[p_i] += d_v

        # Compute Ap and compute new pressure
        for p_i in range(self.container.particle_num[None]):
            x_i = self.container.particle_positions[p_i]
            Ap = 0.0
            dt2 = self.dt[None] * self.dt[None]
            accel_p_i = self.pressure_accel[p_i]
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                Ap += self.container.particle_original_volumes[p_j] * (accel_p_i - self.pressure_accel[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
            # Boundary neighbors
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                Ap += self.container.particle_original_volumes[p_j] * (accel_p_i - self.pressure_accel[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
            Ap *= dt2 * self.density_0
            # print(self.a_ii[1])
            if abs(self.a_ii[p_i]) > 1e-6:
                # Relaxed Jacobi
                self.container.particle_pressures[p_i] = ti.max(self.last_pressure[p_i] + omega * (self.density_deviation[p_i] - Ap) / self.a_ii[p_i], 0.0)
            else:
                self.container.particle_pressures[p_i] = 0.0

            if self.container.particle_pressures[p_i] != 0.0:
                # new_density = self.density_0
                # if p_i == 100:
                #     print(" Ap ", Ap, " density deviation ", self.density_deviation[p_i], 'a_ii ', self.a_ii[p_i])
                self.avg_density_error[None] += abs(Ap - self.density_deviation[p_i]) / self.density_0
        self.avg_density_error[None] /= self.container.particle_num[None]
        for p_i in range(self.container.particle_num[None]):
            # Update the pressure
            self.last_pressure[p_i] = self.container.particle_pressures[p_i]


    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            x_i = self.container.particle_positions[p_i]
            self.container.particle_densities[p_i] = self.container.particle_original_volumes[p_i] * self.cubic_kernel(0.0)
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                self.container.particle_densities[p_i] += self.container.particle_original_volumes[p_j] * self.cubic_kernel((x_i - x_j).norm())
            # Boundary neighbors
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                self.container.particle_densities[p_i] += self.container.particle_original_volumes[p_j] * self.cubic_kernel((x_i - x_j).norm())
            self.container.particle_densities[p_i] *= self.density_0

    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                self.pressure_accel[p_i].fill(0)
                continue
            self.pressure_accel[p_i].fill(0)
            x_i = self.container.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.container.dim)])

            dpi = self.container.particle_pressures[p_i] / self.container.particle_densities[p_i] ** 2
            # Fluid neighbors
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                dpj = self.container.particle_pressures[p_j] / self.container.particle_densities[p_j] ** 2
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.container.particle_original_volumes[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            # Boundary neighbors
            dpj = self.container.particle_pressures[p_i] / self.density_0 ** 2
            # dpj = 0.0
            ## Akinci2012
            for j in range(self.container.solid_neighbors_num[p_i]):
                p_j = self.container.solid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.container.particle_original_volumes[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            self.pressure_accel[p_i] = d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.container.particle_num[None]):
            # if self.ps.material[p_i] != self.ps.material_fluid:
            #     self.ps.acceleration[p_i].fill(0)
            #     continue
            x_i = self.container.particle_positions[p_i]
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.container.dim)])
            d_v[1] = self.g
            for j in range(self.container.fluid_neighbors_num[p_i]):
                p_j = self.container.fluid_neighbors[p_i, j]
                x_j = self.container.particle_positions[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
            self.container.particle_accelerations[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.pressure_accel[p_i]
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()

        self.predict_advection()
        self.pressure_solve()

        self.compute_pressure_forces()
        self.advect()
