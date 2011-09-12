all:
	@echo "======================================================================"
	@echo "Proj 1: Distribued Matrix Multiply"
	@echo ""
	@echo "Valid build targets:"
	@echo ""
	@echo "        unittest_mm : Build matrix multiply unittests"
	@echo "     unittest_summa : Build summa unittests"
	@echo "            time_mm : Build program to time local_mm"
	@echo "         time_summa : Build program to time summa"
	@echo "   run--unittest_mm : Submit unittest_mm job"
	@echo "run--unittest_summa : Submit unittest_summa job"
	@echo "       run--time_mm : Submit time_mm job"
	@echo "    run--time_summa : Submit time_summa job"
	@echo "             turnin : Create tarball with answers and results for T-Square"
	@echo "              clean : Removes generated files, junk"
	@echo "          clean-pbs : Removes genereated pbs files"
	@echo "======================================================================"

LANG = C


CC = mpicc
CFLAGS = -O -Wall -Wextra -lm

FC = mpif90
FFLAGS = -O



ifeq ($(LANG),C)
MM = local_mm.o
SUMMA = summa.o
else
MM = local_mm.o local_mm_wrapper.o
SUMMA = summa.o summa_wrapper.o
endif

local_mm.o : local_mm.c local_mm.f90 local_mm.h
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o $@ -c local_mm.c
else
	$(FC) $(FFLAGS) -o $@ -c local_mm.f90
endif

matrix_utils.o : matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm : unittest_mm.c matrix_utils.o $(MM)
	$(CC) $(CFLAGS) -o $@ $^

time_mm : time_mm.c matrix_utils.o $(MM)
	$(CC) $(CFLAGS) -o $@ $^

unittest_summa : matrix_utils.o $(MM) $(SUMMA) unittest_summa.o
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o $@ $^
else
	$(FC) $(FFLAGS) -o $@ $^
endif

time_summa : matrix_utils.o $(MM) $(SUMMA) time_summa.o
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o $@ $^
else
	$(FC) $(FFLAGS) -o $@ $^
endif

summa.o : summa.c summa.f90 summa.h local_mm.h
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o summa.o -c summa.c
else
	$(FC) $(FFLAGS) -o summa.o -c summa.f90
endif

unittest_summa.o : unittest_summa.c
	$(CC) $(CFLAGS) -o $@ -c $<

time_summa.o : time_summa.c
	$(CC) $(CFLAGS) -o $@ -c $<

summa_wrapper.o : summa_wrapper.c
	$(CC) $(CFLAGS) -o $@ -c $<

local_mm_wrapper.o : local_mm_wrapper.c
	$(CC) $(CFLAGS) -o $@ -c $<


.PHONY : clean
.PHONY : clean-pbs
	
clean : clean-pbs
	rm -f unittest_mm unittest_summa time_mm time_summa
	rm -f *.o
	rm -f turnin.tar.gz

clean-pbs : 
	@ [ -d archive ] || mkdir -p ./archive/
	if [ -f Proj1.e* ]; then mv -f Proj1.e* ./archive/; fi;
	if [ -f Proj1.o* ]; then mv -f Proj1.o* ./archive/; fi;

run--unittest_mm : unittest_mm clean-pbs
	qsub unittest_mm.pbs

run--unittest_summa : unittest_summa clean-pbs
	qsub unittest_summa.pbs

run--time_mm : time_mm clean-pbs
	qsub time_mm.pbs

run--time_summa : time_summa clean-pbs
	qsub time_summa.pbs

turnin : $(TURNIN_FILES)
	tar czvf turnin.tar.gz *

# eof
