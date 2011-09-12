subroutine summa (m,n,k,Ablock,Bblock,Cblock,procGridX,procGridY,panel_size)
  implicit none
  integer, intent(in) :: m, n, k, procGridX, procGridY, panel_size
  double precision, intent(in) :: Ablock(m*k / procGridx / procGridY)
  double precision, intent(in) :: Bblock(k*n / procGridX / procGridY)
  double precision, intent(out) :: Cblock(m*n / procGridX / procGridY)

  include 'mpif.h'

  integer ierr

  call MPI_Barrier (MPI_COMM_WORLD, ierr)
  call MPI_Barrier (MPI_COMM_WORLD, ierr)
end subroutine
