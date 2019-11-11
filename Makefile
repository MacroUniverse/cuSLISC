# Makefile

exe = main.x
gputype = -D CUSLISC_GTX1080
debug = -D CUSLS_CHECKSETSYS -D CUSLS_CHECKBOUNDS -D CUSLS_CHECKTYPE

source = main.cu # SLISC/print.cpp
objects = main.o # print.o
compiler = nvcc
flags =  -arch=sm_60 -g -G -std=c++14 -D SLS_CUSLISC
# -g -G -O3

$(exe):$(objects)
	$(compiler) -o $(exe) $(flags) $(objects)

$(objects):$(source)
	$(compiler) -c $(flags) $(source) -D _CUSLISC_ $(gputype) $(debug)

# clean all except source
clean:
	rm -f *.o *.x *.gch
