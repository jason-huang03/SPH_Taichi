import taichi as ti
from sph_base import SPHBase
from particle_system import ParticleSystem


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system: ParticleSystem):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")
        
        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")
    

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.particle_materials[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.particle_volumes[p_j] * self.density_0 * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.particle_materials[p_i] != self.ps.material_fluid:
                continue
            self.ps.particle_densities[p_i] = self.ps.particle_volumes[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.particle_densities[p_i] += den
    

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        dpi = self.ps.particle_pressures[p_i] / self.ps.particle_densities[p_i] ** 2
        # Fluid neighbors
        if self.ps.particle_materials[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            density_j = self.ps.particle_densities[p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
            dpj = self.ps.particle_pressures[p_j] / (density_j * density_j)
            # Compute the pressure force contribution, Symmetric Formula
            ret += -self.density_0 * self.ps.particle_volumes[p_j] * (dpi + dpj) \
                * self.cubic_kernel_derivative(x_i-x_j)


    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.particle_materials[p_i] != self.ps.material_fluid:
                continue
            self.ps.particle_densities[p_i] = ti.max(self.ps.particle_densities[p_i], self.density_0)
            self.ps.particle_pressures[p_i] = self.stiffness * (ti.pow(self.ps.particle_densities[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in ti.grouped(self.ps.x):
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
            self.ps.particle_accelerations[p_i] += dv


    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        
        ############## Surface Tension ###############
        if self.ps.particle_materials[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.particle_masses[p_i] * self.ps.particle_masses[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.particle_masses[p_i] * self.ps.particle_masses[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.particle_velocities[p_i] -
                self.ps.particle_velocities[p_j]).dot(r)
        
        if self.ps.particle_materials[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.particle_masses[p_j] / (self.ps.particle_densities[p_j])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v


    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.particle_accelerations[p_i] = d_v
            if self.ps.particle_materials[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.particle_accelerations[p_i] = d_v


    @ti.kernel
    def advect(self):
        # Symplectic Euler. Update velocity first and then update position.
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.particle_velocities[p_i] += self.dt[None] * self.ps.particle_accelerations[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.particle_velocities[p_i]

    def step(self):
        self.ps.initialize_particle_system()
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
        self.enforce_boundary_3D(self.ps.material_fluid)


