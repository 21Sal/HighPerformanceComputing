
#include "boundary.h"
#include "data.h"

/**
 * @brief Apply the boundary conditions. This effectively points the ghost cell areas
 *        to the same cell list as the opposite edge (i.e. wraps the domain).
 *        This has to be done after every cell list update, just to ensure that a destructive
 *        operations hasn't broken things.
 * 
 */
void apply_boundary(int sizej, MPI_Datatype my_column) {
	// Apply boundary conditions
	int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int left = (rank - 1) < 0 ? size - 1 : rank - 1;
	int right = (rank + 1) >= size ? 0 : rank + 1;

	// exchange column 1 with right-most ghost column
	MPI_Sendrecv(&cells[0][1], 1, my_column, left, 0, &cells[0][sizej], my_column, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	// exchange right-most regular column with left-most ghost column 
	MPI_Sendrecv(&cells[0][sizej], 1, my_column, left, 0, &cells[0][0], my_column, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// for (int j = 1; j < y+1; j++) {
	// 	cells[0][j].part_ids = cells[x][j].part_ids;
	// 	cells[0][j].count = cells[x][j].count;
	// 	cells[0][j].size = cells[x][j].size;

	// 	cells[x+1][j].part_ids = cells[1][j].part_ids;
	// 	cells[x+1][j].count = cells[1][j].count;
	// 	cells[x+1][j].size = cells[1][j].size;
	// }

	for (int i = 0; i < x+2; i++) {
		cells[i][0].part_ids = cells[i][y].part_ids;
		cells[i][0].count = cells[i][y].count;
		cells[i][0].size = cells[i][y].size;

		cells[i][y+1].part_ids = cells[i][1].part_ids;
		cells[i][y+1].count = cells[i][1].count;
		cells[i][y+1].size = cells[i][1].size;
	}
}