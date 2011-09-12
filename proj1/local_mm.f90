subroutine local_mm (m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
  implicit none
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  double precision, intent(in) :: alpha, beta
  double precision, intent(in) :: A(m*k)
  double precision, intent(in) :: B(k*n)
  double precision, intent(out) :: C(m*n)


  integer ierr, row, col, k_iter, a_index, b_index, c_index
  double precision dotprod

  do col = 0, n-1
    do row = 0, m-1
      dotprod = 0
      do k_iter =  0, k-1
        a_index = (k_iter * lda) + row
        b_index = (col * ldb) + k_iter
	dotprod = dotprod + A(a_index + 1) * B(b_index + 1)
      end do
      c_index = (col * ldc) + row
      C(c_index + 1) = (alpha * dotprod) + (beta * C(c_index+1))
    end do
  end do


end subroutine
