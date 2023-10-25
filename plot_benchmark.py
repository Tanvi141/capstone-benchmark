import benchmark

provider_wise_res_mm, providers, matrix_configurations

providers = ["triton"]

matrix_configurations = []
matrix_configurations.extend([[2048, 2048, 1 * i, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
    
triton_configs_to_plot = 
{
    "m_block" : #[],
    "n_block" : #[],
    "k_block" : #[],
    "m_group" : #[],
    "stages" : #[],
    "warps" : #[]
}