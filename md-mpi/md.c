#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "args.h"
#include "boundary.h"
#include "data.h"
#include "setup.h"
#include "vtk.h"


/**
 * @brief This routine calculates the acceleration felt by each particle based on evaluating the Lennard-Jones 
 *        potential with its neighbours. It only evaluates particles within a cut-off radius, and uses cells to 
 *        reduce the search space. It also calculates the potential energy of the system. 
 * 
 * @return double The potential energy
 */
double comp_accel() {
	// printf("starting comp_accel");
	// zero acceleration for every particle
	for (int p = 0; p < num_particles_per_proc; p++) {
		particles.ax[p_offset + p] = 0.0;
		particles.ay[p_offset + p] = 0.0;
	}

	double pot_energy = 0.0;

	for (int i = 1; i < sizei+1; i++) {
		for (int j = 1; j < sizej+1; j++) {
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
							// printf("in comp_accel");


							// since particles are stored relative to their cell, calculate the
							// actual x and y coordinates.
							double p_real_x = ((i-1) * cell_size) + particles.x[p];
							double p_real_y = ((j-1) * cell_size) + particles.y[p];
							double q_real_x = ((i+a-1) * cell_size) + particles.x[q];
							double q_real_y = ((j+b-1) * cell_size) + particles.y[q];
							
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

								particles.ax[p] += f*dx;

								particles.ay[p] += f*dy;


								pot_energy += 4.0 * r_6_inv * (r_6_inv - 1.0) - Uc - Duc * (sqrt(r_2) - r_cut_off);
							}
						}
					}
				}
			}
		}
	}

	// MPI_Allreduce(MPI_IN_PLACE, &pot_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// return the average potential energy (i.e. sum / number)
	return pot_energy / num_particles_per_proc;
}

/**
 * @brief This routine updates the velocity of each particle for half a time step and then 
 *        moves the particle for a whole time step
 * 
 */
void move_particles() {
	// move all particles half a time step
	for (int p = 0; p < num_particles_per_proc; p++) {
		// update velocity to obtain v(t + Dt/2)
		particles.vx[p_offset + p] += dth * particles.ax[p_offset + p];
		particles.vy[p_offset + p] += dth * particles.ay[p_offset + p];

		// update particle coordinates to p(t + Dt) (scaled to the cell_size)
		particles.x[p_offset + p] += (dt * particles.vx[p_offset + p]);
		particles.y[p_offset + p] += (dt * particles.vy[p_offset + p]);
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
	for (int i = 1; i < sizei+1; i++) {
		for (int j = 1; j < sizej+1; j++) {
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
					add_particle(&(cells[new_i][new_j]), p);
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

	for (int p = 0; p < num_particles_per_proc; p++) {
		// update velocity again by half time to obtain v(t + Dt)
		particles.vx[p_offset + p] += dth * particles.ax[p_offset + p];
		particles.vy[p_offset + p] += dth * particles.ay[p_offset + p];

		// calculate the kinetic energy by adding up the squares of the velocities in each dim
		kinetic_energy += (particles.vx[p_offset + p] * particles.vx[p_offset + p]) + (particles.vy[p_offset + p] * particles.vy[p_offset + p]);
	}

	// KE = (1/2)mv^2
	kinetic_energy *= (0.5 / num_particles_per_proc);
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

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int dims[2] = {0,0};
	int periods[2] = {1,1};
	int my_coords[2];

	MPI_Dims_create(size, 2 , dims);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
	
	MPI_Cart_coords(cart_comm, rank, 2, my_coords);
	
	MPI_Cart_shift(cart_comm, 0, 1, &west_rank, &east_rank);
   	MPI_Cart_shift(cart_comm, 1, 1, &south_rank, &north_rank);


	// Set default parameters
	set_defaults();
	// parse the arguments
	if (rank == 0) parse_args(argc, argv);
	// call set up to update defaults
	setup();

	if (rank == 0 && verbose) print_opts();
	
	// calculate the start and end indicies for parallel processors

	// calculated the size of the used portion of the array on each parallel processor
    sizej = x/dims[0];

	// calculated the size of the used portion of the array on each parallel processor
    sizei = y/dims[1];
	
	// set up a custom data type that is M doubles in a column.
    // The gap will be "sizej" (i.e. the size of a row on each processor)
    MPI_Type_vector(sizej+2, 1, sizei+2, MPI_INT, &mpi_part_ids_column);
    MPI_Type_vector(sizei+2, 1, sizej+2, MPI_INT, &mpi_count_column);

    MPI_Type_commit(&mpi_part_ids_column);
    MPI_Type_commit(&mpi_count_column);

	// set up problem
	problem_setup();
	// apply boundary condition (i.e. update pointers on the boundarys to loop periodically)
	printf("n %d\n", num_particles_per_proc);
	printf("before boundary\n");
	apply_boundary();
	printf("after boundary\n");
	
	comp_accel();

	double potential_energy = 0.0;
	double kinetic_energy = 0.0;

	int iters = 0;
	double t;
	for (t = 0.0; t < t_end; t+=dt, iters++) {
		// move particles half a time step
		move_particles();

		// update cell lists (i.e. move any particles between cell lists if required)
		update_cells();

		// update pointers (because the previous operation might break boundary cell lists)
		apply_boundary();
		
		// compute acceleration for each particle and calculate potential energy
		potential_energy = comp_accel();

		// update velocity based on the acceleration and calculate the kinetic energy
		kinetic_energy = update_velocity();
	
		if (iters % output_freq == 0) {
			// calculate temperature and total energy
			MPI_Allreduce(MPI_IN_PLACE, &potential_energy, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
			MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

			double total_energy = kinetic_energy + potential_energy;
			double temp = kinetic_energy * 2.0 / 3.0;
			
			if(rank == 0) {
				printf("Step %8d, Time: %14.8e (dt: %14.8e), Total energy: %14.8e (p:%14.8e,k:%14.8e), Temp: %14.8e\n", iters, t+dt, dt, total_energy, potential_energy, kinetic_energy, temp);
			}
 
			// if output is enabled and checkpointing is enabled, write out
            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t+dt);
		}
	}

	// calculate the final energy and write out a final status message
	MPI_Allreduce(MPI_IN_PLACE, &potential_energy, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
	MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
	double final_energy = kinetic_energy + potential_energy;
	
	if (rank == 0) {
		printf("Step %8d, Time: %14.8e, Final energy: %14.8e\n", iters, t, final_energy);
		printf("Simulation complete.\n");
		// if output is enabled, write the mesh file and the final state
		if (!no_output) {
			write_mesh();
			write_result(iters, t);
		}
	}

	MPI_Finalize();	

	return 0;
}

