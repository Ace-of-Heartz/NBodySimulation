#ifndef COMMON_HCL
#define COMMON_HCL

#define LOCKED -2
#define DEFAULT -1

#define NUMBER_OF_CELLS 8
#define WARP_SIZE 16 // Not sure about this

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif


#endif
