#include <stdio.h>
#include <stdlib.h>

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
	MPI_Request requests[10];

	temp_part_x = malloc(sizeof(double) * num_particles_total);
	temp_part_y = malloc(sizeof(double) * num_particles_total);
	temp_part_ax = malloc(sizeof(double) * num_particles_total);
	temp_part_ay = malloc(sizeof(double) * num_particles_total);
	temp_part_vx = malloc(sizeof(double) * num_particles_total);
	temp_part_vy = malloc(sizeof(double) * num_particles_total);

	east_part_ids = malloc(sizeof(int)*sizei*2*num_part_per_dim*num_part_per_dim);
	west_part_ids = malloc(sizeof(int)*sizei*2*num_part_per_dim*num_part_per_dim);
	north_part_ids = malloc(sizeof(int)*(sizej+2)*2*num_part_per_dim*num_part_per_dim);
	south_part_ids = malloc(sizeof(int)*(sizej+2)*2*num_part_per_dim*num_part_per_dim);

	east_counts = malloc(sizeof(int)*sizei);
	west_counts = malloc(sizeof(int)*sizei);
	north_counts = malloc(sizeof(int)*(sizej+2));
	south_counts = malloc(sizeof(int)*(sizej+2));

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
		south_counts[i] = cells[i][sizej].count;
		north_counts[i] = cells[i][1].count;
	}

	MPI_Iallgather(particles.ax, num_particles_per_proc, MPI_DOUBLE, temp_part_ax, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[0]);
	MPI_Iallgather(particles.ay, num_particles_per_proc, MPI_DOUBLE, temp_part_ay, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[1]);
	MPI_Iallgather(particles.vx, num_particles_per_proc, MPI_DOUBLE, temp_part_vx, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[2]);
	MPI_Iallgather(particles.vy, num_particles_per_proc, MPI_DOUBLE, temp_part_vy, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[3]);
	MPI_Iallgather(particles.x, num_particles_per_proc, MPI_DOUBLE, temp_part_x, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[4]);
	MPI_Iallgather(particles.y, num_particles_per_proc, MPI_DOUBLE, temp_part_y, num_particles_per_proc, MPI_DOUBLE, MPI_COMM_WORLD, &requests[5]);

	MPI_Sendrecv_replace(east_part_ids, sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, east_rank, 1, 
						west_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv_replace(west_part_ids, sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, west_rank, 1, 
						east_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv_replace(east_counts, sizei, MPI_INT, east_rank, 1, west_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv_replace(west_counts, sizei, MPI_INT, west_rank, 1, east_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv_replace(north_part_ids, sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, north_rank, 1,
						south_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Sendrecv_replace(south_part_ids, sizei*2*num_part_per_dim*num_part_per_dim, MPI_INT, south_rank, 1,
						north_rank, 1, cart_comm, MPI_STATUS_IGNORE);	

	MPI_Sendrecv_replace(north_counts, sizei, MPI_INT, north_rank, 1, south_rank, 1, cart_comm, MPI_STATUS_IGNORE);
	
	MPI_Sendrecv_replace(south_counts, sizei, MPI_INT, south_rank, 1, north_rank, 1, cart_comm, MPI_STATUS_IGNORE);

	MPI_Waitall(6, requests, MPI_STATUS_IGNORE);

	for (int p = 0; p < num_particles_total; p++) {
		particles.ax[p] = temp_part_ax[p];
		particles.ay[p] = temp_part_ay[p];
		particles.vx[p] = temp_part_vx[p];
		particles.vy[p] = temp_part_vy[p];
		particles.x[p] = temp_part_x[p];
		particles.y[p] = temp_part_y[p];
	}

	for (int j = 1; j < sizei+1; j++) {
		for (int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			cells[0][j].part_ids[k] = east_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k];
			cells[sizei+1][j].part_ids[k] = west_part_ids[((j-1)*2*num_part_per_dim*num_part_per_dim) + k];
		}
		cells[0][j].count = east_counts[(j-1)];
		cells[sizei+1][j].count = west_counts[(j-1)] ;
	}

	for (int i = 0; i < sizej+2; i++) {
		for (int k = 0; k < 2*num_part_per_dim*num_part_per_dim; k++) {
			cells[i][0].part_ids[k] = south_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k];
			cells[i][sizej+1].part_ids[k] = north_part_ids[(i*2*num_part_per_dim*num_part_per_dim) + k];
		}
		cells[i][0].count = south_counts[i];
		cells[i][sizej+1].count = north_counts[i];
	}

	free(temp_part_x);
	free(temp_part_y);
	free(temp_part_ax);
	free(temp_part_ay);
	free(temp_part_vx);
	free(temp_part_vy);

	free(east_part_ids);
	free(west_part_ids);
	free(north_part_ids);
	free(south_part_ids);

	free(east_counts);
	free(west_counts);
	free(north_counts);
	free(south_counts);
}
