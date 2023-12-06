from matplotlib.pyplot import axis
import taichi as ti
import numpy as np
from particle_system import DFSPHContainer3D

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system: DFSPHContainer3D):
        self.container = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.container.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.container.cfg.get_cfg("gravitation"))

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.container.cfg.get_cfg("density0")

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

    @ti.func
    def cubic_kernel(self, R_mod):
        res = ti.cast(0.0, ti.f32)
        h = self.container.dh
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k /= h ** self.container.dim
        q = R_mod / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, R):
        h = self.container.dh
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.container.dim
        R_mod = R.norm()
        q = R_mod / h
        res = ti.Vector([0.0 for _ in range(self.container.dim)])
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.container.particle_velocities[p_i] -
                self.container.particle_velocities[p_j]).dot(r)
        res = 2 * (self.container.dim + 2) * self.viscosity * (self.container.particle_masses[p_j] / (self.container.particle_densities[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.container.dh**2) * self.cubic_kernel_derivative(
                r)
        return res


    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.container.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in range(self.container.particle_num[None]):
            if not self.container.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.container.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.container.particle_original_volumes[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_solid:
            delta += self.cubic_kernel((self.container.particle_positions[p_i] - self.container.particle_positions[p_j]).norm())


    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in range(self.container.particle_num[None]):
            if not self.container.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.container.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.container.particle_original_volumes[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.container.particle_velocities[p_i] -= (
            1.0 + c_f) * self.container.particle_velocities[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in ti.grouped(self.container.particle_positions):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]: 
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]:
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding

                if pos[2] > self.container.domain_size[2] - self.container.padding:
                    collision_normal[2] += 1.0
                    self.container.particle_positions[p_i][2] = self.container.domain_size[2] - self.container.padding
                if pos[2] <= self.container.padding:
                    collision_normal[2] += -1.0
                    self.container.particle_positions[p_i][2] = self.container.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)


    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.container.particle_num[None]):
            if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_id[p_i] == object_id:
                mass = self.container.m_V0 * self.container.particle_densities[p_i]
                cm += mass * self.container.particle_positions[p_i]
                sum_m += mass
        cm /= sum_m
        return cm
    

    @ti.kernel
    def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
        return self.compute_com(object_id)


    @ti.kernel
    def solve_constraints(self, object_id: int) -> ti.types.matrix(3, 3, float):
        # compute center of mass
        cm = self.compute_com(object_id)
        # A
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(self.container.particle_num[None]):
            if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_id[p_i] == object_id:
                q = self.container.x_0[p_i] - self.container.rigid_rest_cm[object_id]
                p = self.container.particle_positions[p_i] - cm
                A += self.container.m_V0 * self.container.particle_densities[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)
        
        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)
        
        for p_i in range(self.container.particle_num[None]):
            if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_id[p_i] == object_id:
                goal = cm + R @ (self.container.x_0[p_i] - self.container.rigid_rest_cm[object_id])
                corr = (goal - self.container.particle_positions[p_i]) * 1.0
                self.container.particle_positions[p_i] += corr
        return R
        

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]

    #             if self.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = self.ps.x[p_j]
    #                 r = self.ps.x[p_i] - x_j
    #                 if r.norm() < self.ps.particle_diameter:
    #                     x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             self.ps.x[p_i] += 2.0 * x_delta # / cnt
                        


    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in self.container.object_id_rigid_body:
                if self.container.object_collection[r_obj_id]["isDynamic"]:
                    R = self.solve_constraints(r_obj_id)

                    if self.container.cfg.get_cfg("exportObj"):
                        # For output obj only: update the mesh
                        cm = self.compute_com_kernel(r_obj_id)
                        ret = R.to_numpy() @ (self.container.object_collection[r_obj_id]["restPosition"] - self.container.object_collection[r_obj_id]["restCenterOfMass"]).T
                        self.container.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                    # self.compute_rigid_collision()
                    self.enforce_boundary_3D(self.container.material_solid)


    def step(self):
        self.container.prepare_neighborhood_search()
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.container.dim == 2:
            self.enforce_boundary_2D(self.container.material_fluid)
        elif self.container.dim == 3:
            self.enforce_boundary_3D(self.container.material_fluid)
