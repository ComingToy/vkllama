#include "shaders/vkllama_shaders.h"
#include <stdio.h>

int main(void)
{
	printf("matmul_shared_mem_comp_spv: 0x%p", vec_mul_add_comp_spv);
	return 0;
}
