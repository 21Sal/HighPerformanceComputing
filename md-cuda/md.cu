#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "args.h"
#include "boundary.h"
#include "data.h"
#include "setup.h"
#include "vtk.h"

void CUDAErrorCheck() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
        exit(0);
    }
}


__global__ void cuda_comp_accel(int * d_part_neighbour_list, double * d_part_x, double * d_part_y,
								double * d_part_ax, double * d_part_ay, int * d_part_i, int * d_part_j, double * d_pot_energy_arr,
								double cell_size, double r_cut_off, double r_cut_off_2,
								double Duc, double Uc, int y, int num_particles, int num_part_per_dim) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i = d_part_i[p];
	int j = d_part_j[p];

	// printf("p %d\n", p);

	for (int n = 0; n < 2*num_part_per_dim*num_part_per_dim; n++) {
		int q = d_part_neighbour_list[(p*2*num_part_per_dim*num_part_per_dim) + n];
		// printf("q %d\n", q);
		
		if (q > -1) {

			int iq = d_part_i[q];
			int jq = d_part_j[q];
			// since particles are stored relative to their cell, calculate the
			// actual x and y coordinates.

			double p_real_x = ((i-1) * cell_size) + d_part_x[p];
			double p_real_y = ((j-1) * cell_size) + d_part_y[p];
			double q_real_x = ((iq-1) * cell_size) + d_part_x[q];
			double q_real_y = ((jq-1) * cell_size) + d_part_y[q];
			
			// calculate distance in x and y, then absolute distance
			double dx = p_real_x - q_real_x;
			double dy = p_real_y - q_real_y;
			double r_2 = dx*dx + dy*dy;
			
			// if distance less than cut off, calculate force and 
			// use this to calculate acceleration in each dimension
			// calculate potential energy of each particle at the same time
			if (r_2 < r_cut_off_2) {
				double r_2_inv = 1.0 / r_2;
				double r_6_inv = r_2_inv * r_2_inv * r_2_inv;
				
				double f = (48.0 * r_2_inv * r_6_inv * (r_6_inv - 0.5));

				d_part_ax[p] += f*dx;

				d_part_ay[p] += f*dy;
				d_pot_energy_arr[p] += 4.0 * r_6_inv * (r_6_inv - 1.0) - Uc - Duc * (sqrt(r_2) - r_cut_off);
			}
		}
	}
}


/**
 * @brief This routine calculates the acceleration felt by each particle based on evaluating the Lennard-Jones 
 *        potential with its neighbours. It only evaluates particles within a cut-off radius, and uses cells to 
 *        reduce the search space. It also calculates the potential energy of the system. 
 * 
 * @return double The potential energy
 */
double comp_accel(double * pot_energy_arr, double * d_pot_energy_arr) {
	// zero acceleration for every particle
	for (int p = 0; p < num_particles; p++) {
		particles.ax[p] = 0.0;
		particles.ay[p] = 0.0;
		particles.num_neighbours[p] = 0;
	}

	for (int m = 0; m < num_particles*2*num_part_per_dim*num_part_per_dim; m++) {
		part_neighbour_list[m] = -1;
	}

	double pot_energy = 0.0;
	cudaMemset(d_pot_energy_arr, 0.0, sizeof(double) * num_particles);

	// Compare each particle with all particles in the 9 cells
	for (int i = 1; i < x+1; i++) {
		for (int j = 1; j < y+1; j++) {
			for (int k = 0; k < cells[i][j].count; k++) {
				int p = cells[i][j].part_ids[k];
				// Compare each particle with all particles in the 9 cells
				for (int a = -1; a <= 1; a++) {
					for (int b = -1; b <= 1; b++) {
						for (int l = 0; l < cells[i+a][j+b].count; l++) {
							int q = cells[i+a][j+b].part_ids[l];
							if (p == q) {
								continue;
							}
							part_neighbour_list[(p*2*num_part_per_dim*num_part_per_dim) + particles.num_neighbours[p]] = q;
							particles.num_neighbours[p]++;
						}
					}
				}
			}
		}
	}
	cudaMemcpy(d_part_neighbour_list, part_neighbour_list, sizeof(int)*num_particles*2*num_part_per_dim*num_part_per_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_ax, particles.ax, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_ay, particles.ay, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	CUDAErrorCheck();
	cudaDeviceSynchronize();

	int block_size = 256;
	int grid_size = num_particles / block_size;
	cuda_comp_accel<<<grid_size,block_size>>>(d_part_neighbour_list, d_part_x, d_part_y, d_part_ax, d_part_ay,
												d_part_i, d_part_j, d_pot_energy_arr, cell_size,
												r_cut_off, r_cut_off_2, Duc, Uc, y, num_particles, num_part_per_dim);
	CUDAErrorCheck();
	cudaMemcpy(pot_energy_arr, d_pot_energy_arr, sizeof(double) * num_particles, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int p = 0; p < num_particles; p++) {
		pot_energy += pot_energy_arr[p];
	}

	// return the average potential energy (i.e. sum / number)
	return pot_energy / num_particles;
}

/**
 * @brief This routine updates the velocity of each particle for half a time step and then 
 *        moves the particle for a whole time step
 * 
 */
__global__ void move_particles(double * d_part_vx, double * d_part_vy,
								double * d_part_x, double * d_part_y,
								double * d_part_ax, double * d_part_ay,
								int num_particles, double dt, double dth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// move all particles half a time step
	if (tid < num_particles) {
		// update velocity to obtain v(t + Dt/2)
		d_part_vx[tid] += dth * d_part_ax[tid];
		d_part_vy[tid] += dth * d_part_ay[tid];

		// update particle coordinates to p(t + Dt) (scaled to the cell_size)
		d_part_x[tid] += (dt * d_part_vx[tid]);
		d_part_y[tid] += (dt * d_part_vy[tid]);
	}
}

/**
 * @brief This routine updates the cell lists. If a particles coordinates are not within a cell
 *        any more, this function calculates the cell it should be in and performs the move.
 *        If a particle moves more than 1 cell in any direction, this indicates poor settings
 *        and therefore an error is generated.
 * 
 */
void update_cells() {
	// move particles that need to move cell lists
	for (int i = 1; i < x+1; i++) {
		for (int j = 1; j < y+1; j++) {
			// we have to store the next particle here, as the remove/add at the end may be destructive
			
			int cell_count = cells[i][j].count;
			int * cell_part_ids = cells[i][j].part_ids;
			for (int k = 0; k < cell_count; k++) {
				int p = cell_part_ids[k];

				// if a particles x or y value is greater than the cell size or less than 0, it must have moved cell
				// do a quick check to make sure its not moved 2 cells (since this means our time step is too large, or something else is going wrong)
				if ((particles.x[p] < 0.0) | (particles.x[p] >= cell_size) | (particles.y[p] < 0.0) | (particles.y[p] >= cell_size)) {
					if ((particles.x[p] < (-cell_size)) || (particles.x[p] >= (2*cell_size)) || (particles.y[p] < (-cell_size)) || (particles.y[p] >= (2*cell_size))) {
						fprintf(stderr, "A particle has moved more than one cell!\n");
						exit(1);
					}

					// work out whether we've moved a cell in the x and the y dimension
					int x_shift = (particles.x[p] < 0.0) ? -1 : (particles.x[p] >= cell_size) ? +1 : 0;
					int y_shift = (particles.y[p] < 0.0) ? -1 : (particles.y[p] >= cell_size) ? +1 : 0;
					
					// the new i and j are +/- 1 in each dimension,
					// but if that means we go out of simulation bounds, wrap it to x and 1
					int new_i = i+x_shift;
					if (new_i == 0) { new_i = x; }
					if (new_i == x+1) { new_i = 1; }
					int new_j = j+y_shift;
					if (new_j == 0) { new_j = y; }
					if (new_j == y+1) { new_j = 1; }
					// update x and y coordinates (i.e. remove the additional cell size)
					particles.x[p] = particles.x[p] + (x_shift * -cell_size);
					particles.y[p] = particles.y[p] + (y_shift * -cell_size);

					// remove the particle from its current cell list, then add it to the new cell list
					remove_particle(&(cells[i][j]), k);
					add_particle(&(cells[new_i][new_j]), p, new_i, new_j);
				}
			}
		}
	}
}

/**
 * @brief This updates the velocity of particles for the whole time step (i.e. adds the acceleration for another
 *        half step, since its already done half a time step in the move_particles routine). Additionally, this
 *        function calculated the kinetic energy of the system.
 * 
 * @return double The kinetic energy
 */
double update_velocity() {
	double kinetic_energy = 0.0;

	for (int p = 0; p < num_particles; p++) {
		// update velocity again by half time to obtain v(t + Dt)
		particles.vx[p] += dth * particles.ax[p];
		particles.vy[p] += dth * particles.ay[p];

		// calculate the kinetic energy by adding up the squares of the velocities in each dim
		kinetic_energy += (particles.vx[p] * particles.vx[p]) + (particles.vy[p] * particles.vy[p]);
	}

	// KE = (1/2)mv^2
	kinetic_energy *= (0.5 / num_particles);
	return kinetic_energy;
}

/**
 * @brief This is the main routine that sets up the problem space and then drives the solving routines.
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The exit code of the application
 */
int main(int argc, char *argv[]) {
	cudaEvent_t start, stop;
    float gpu_time;

	// Set default parameters
	set_defaults();
	// parse the arguments
	parse_args(argc, argv);
	// call set up to update defaults
	setup();

	if (verbose) print_opts();
	
	// create events for time profiling
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the event timer
    cudaEventRecord(start, 0);

	// set up problem
	problem_setup();

	// apply boundary condition (i.e. update pointers on the boundarys to loop periodically)
	apply_boundary();

	double * d_pot_energy_arr;
	double * pot_energy_arr = (double *) malloc(sizeof(double) * num_particles);
	
	cudaMalloc((void **) &d_part_i, sizeof(int) * num_particles);
	cudaMalloc((void **) &d_part_j, sizeof(int) * num_particles);
	cudaMalloc((void **) &d_part_x, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_y, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_ax, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_ay, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_vx, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_vy, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_pot_energy_arr, sizeof(double) * num_particles);
	cudaMalloc((void **) &d_part_neighbour_list, sizeof(int) * num_particles * (2*num_part_per_dim*num_part_per_dim));
	CUDAErrorCheck();

	cudaMemcpy(d_part_i, particles.cell_i, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_j, particles.cell_j, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_x, particles.x, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_y, particles.y, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_vx, particles.vx, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_part_vy, particles.vy, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	comp_accel(pot_energy_arr, d_pot_energy_arr);

	cudaMemcpy(particles.cell_i, d_part_i, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.cell_j, d_part_j, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.x, d_part_x, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.y, d_part_y, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.ax, d_part_ax, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.ay, d_part_ay, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.vx, d_part_vx, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(particles.vy, d_part_vy, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	double potential_energy = 0.0;
	double kinetic_energy = 0.0;

	int iters = 0;
	double t;
	for (t = 0.0; t < t_end; t+=dt, iters++) {
		// move particles half a time step
		cudaMemcpy(d_part_i, particles.cell_i, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_j, particles.cell_j, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_x, particles.x, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_y, particles.y, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_ax, particles.ax, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_ay, particles.ay, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_vx, particles.vx, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_vy, particles.vy, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
	    cudaDeviceSynchronize();
		
		int block_size = 256;
		int grid_size = num_particles / block_size;
		move_particles<<<grid_size,block_size>>>(d_part_x, d_part_y, d_part_ax, d_part_ay, d_part_vx, d_part_vy, num_particles, dt, dth);
		
		cudaMemcpy(particles.cell_i, d_part_i, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.cell_j, d_part_j, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.x, d_part_x, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.y, d_part_y, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.ax, d_part_ax, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.ay, d_part_ay, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.vx, d_part_vx, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.vy, d_part_vy, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();

		// update cell lists (i.e. move any particles between cell lists if required)
		update_cells();

		// update pointers (because the previous operation might break boundary cell lists)
		apply_boundary();
		
		// compute acceleration for each particle and calculate potential energy
		cudaMemcpy(d_part_i, particles.cell_i, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_j, particles.cell_j, sizeof(int)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_x, particles.x, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_y, particles.y, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_ax, particles.ax, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_part_ay, particles.ay, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		// cudaMemcpy(d_part_vx, particles.vx, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		// cudaMemcpy(d_part_vy, particles.vy, sizeof(double)*num_particles, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		potential_energy = comp_accel(pot_energy_arr, d_pot_energy_arr);

		cudaMemcpy(particles.cell_i, d_part_i, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.cell_j, d_part_j, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.x, d_part_x, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.y, d_part_y, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.ax, d_part_ax, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(particles.ay, d_part_ay, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		// cudaMemcpy(particles.vx, d_part_vx, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
		// cudaMemcpy(particles.vy, d_part_vy, sizeof(double)*num_particles, cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();

		// update velocity based on the acceleration and calculate the kinetic energy
		kinetic_energy = update_velocity();
	
		if (iters % output_freq == 0) {
			// calculate temperature and total energy
			double total_energy = kinetic_energy + potential_energy;
			double temp = kinetic_energy * 2.0 / 3.0;

			printf("Step %8d, Time: %14.8e (dt: %14.8e), Total energy: %14.8e (p:%14.8e,k:%14.8e), Temp: %14.8e\n", iters, t+dt, dt, total_energy, potential_energy, kinetic_energy, temp);
 
			// if output is enabled and checkpointing is enabled, write out
            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t+dt);
		}
	}

	// calculate the final energy and write out a final status message
	double final_energy = kinetic_energy + potential_energy;
	printf("Step %8d, Time: %14.8e, Final energy: %14.8e\n", iters, t, final_energy);
    printf("Simulation complete.\n");

	// if output is enabled, write the mesh file and the final state
	if (!no_output) {
		write_mesh();
		write_result(iters, t);
	}

	return 0;
}

