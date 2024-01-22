#include <stdio.h>

#include "boundary.h"
#include "data.h"

/**
 * @brief Apply the boundary conditions. This effectively points the ghost cell areas
 *        to the same cell list as the opposite edge (i.e. wraps the domain).
 *        This has to be done after every cell list update, just to ensure that a destructive
 *        operations hasn't broken things.
 * 
 */
void apply_boundary() {

	MPI_Barrier(cart_comm);
	MPI_Barrier(MPI_COMM_WORLD);
	
	for (int j = 1; j < sizei+1; j++) {
		for (int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			east_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k] = cells[sizei][j].part_ids[k];
			west_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k] = cells[1][j].part_ids[k];
		}
		east_counts[(j-1)] = cells[sizei][j].count;
		west_counts[(j-1)] = cells[1][j].count;
	}

	for (int i = 0; i < sizej+2; i++) {
		for (int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			south_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k] = cells[i][sizej].part_ids[k];
			north_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k] = cells[i][1].part_ids[k];
		}
		south_counts[i] = cells[i][y].count;
		north_counts[i] = cells[i][1].count;
	}

	MPI_Allgather(&particles.ax[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_ax[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&particles.ay[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_ay[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&particles.vx[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_vx[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&particles.vy[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_vy[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&particles.x[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_x[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&particles.y[0], num_particles_per_proc, MPI_DOUBLE, &temp_part_y[0], num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

	MPI_Sendrecv(&east_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, east_rank, 1, &temp_east_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, 
				MPI_INT, west_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv(&west_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, west_rank, 1, &temp_west_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, 
				MPI_INT, east_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv(&east_counts[0], sizei, MPI_INT, east_rank, 1, &temp_east_counts[0], sizei, MPI_INT, west_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv(&west_counts[0], sizei, MPI_INT, west_rank, 1, &temp_west_counts[0], sizei, MPI_INT, east_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv(&north_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, north_rank, 1, &temp_north_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, 
				MPI_INT, south_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv(&south_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, south_rank, 1, &temp_south_part_ids[0], sizei*2*num_part_per_dim*num_part_per_dim, 
				MPI_INT, north_rank, 1, cart_comm, MPI_STATUS_IGNORE);	

	MPI_Sendrecv(&north_counts[0], sizei, MPI_INT, north_rank, 1, &temp_north_counts[0], sizei, MPI_INT, south_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv(&south_counts[0], sizei, MPI_INT, south_rank, 1, &temp_south_counts[0], sizei, MPI_INT, north_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Barrier(cart_comm);

	for (int p = 0; p < num_particles_total; p++) {
		particles.ax[p] = temp_part_ax[p];
		particles.ay[p] = temp_part_ay[p];
		particles.vx[p] = temp_part_vx[p];
		particles.vy[p] = temp_part_vy[p];
		particles.x[p] = temp_part_x[p];
		particles.y[p] = temp_part_y[p];
	}

	for (int j = 1; j < sizei+1; j++) {
		for(int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			cells[0][j].part_ids[k] = temp_east_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k];
			cells[sizei+1][j].part_ids[k] = temp_west_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k];
		}
		cells[0][j].count = temp_east_counts[(j-1)];
		cells[sizei+1][j].count = temp_west_counts[(j-1)] ;
	}

	for (int i = 0; i < sizej+2; i++) {
		for(int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			cells[i][0].part_ids[k] = temp_south_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k];
			cells[i][sizej+1].part_ids[k] = temp_north_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k];
		}
		cells[i][0].count = temp_south_counts[i];
		cells[i][sizej+1].count = temp_north_counts[i];
	}
}
