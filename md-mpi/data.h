#ifndef DATA_H
#define DATA_H
#include <mpi.h>
// particle data type (with pointers to next and previous for linked list)
struct particle_t {
	double * x, * y; // position within cell
	double * ax, * ay; // acceleration
	double * vx, * vy; // velocity
};

// list for a cell, with a head
struct cell_list {
	int count;
	int size;
	int * part_ids;
};

extern double growth_factor;

// parameters for end time, cut off, cell size, grid size and number of particles
extern double t_end;
extern double r_cut_off;
extern double cell_size;
extern int x;
extern int y;
extern int num_particles_per_proc, num_particles_total, p_offset;

// number of iterations, timestep duration and half-timestep duration
extern int niters;
extern double dt;
extern double dth;

// square of the cut off (to prevent need for some sqrts later)
extern double r_cut_off_2;

// constants required to calculate the potential energy
extern double r_cut_off_2_inv; 
extern double r_cut_off_6_inv;	
extern double Uc;
extern double Duc;

// random seed (to allow reproducibility)
extern long seed;

// initial temperature and the number of particles per cell per dimension
extern double init_temp;
extern int num_part_per_dim;

// the cell list
extern struct cell_list ** cells;
extern struct particle_t particles;
extern int size, rank;
extern int sizej, sizei;
extern MPI_Comm cart_comm;
extern MPI_Datatype mpi_part_ids_column;
extern MPI_Datatype mpi_count_column;
extern MPI_Datatype mpi_cell_t;
extern int east_rank, west_rank, north_rank, south_rank;
extern int * east_part_ids, * west_part_ids, * north_part_ids, * south_part_ids;
extern int * east_counts, * west_counts, * north_counts, * south_counts;
extern int * temp_east_part_ids, * temp_west_part_ids, * temp_north_part_ids, * temp_south_part_ids;
extern int * temp_east_counts, * temp_west_counts, * temp_north_counts, * temp_south_counts;
extern double * temp_part_ax, * temp_part_ay, * temp_part_x, * temp_part_y, * temp_part_vx, * temp_part_vy;


void add_particle(struct cell_list * list, int part_id);
void remove_particle(struct cell_list * list, int idx);
struct cell_list ** alloc_2d_cell_list_array(int m, int n);
void free_2d_array(void ** array);

#endif