import matmul.matmul as mm
from remote_plot import plt
import seaborn as sns
# import cutlass as plotcut
import triton, torch

SPACER = 25

def plot_graphs(provider_wise_res_mm, providers, matrix_configurations, mm=True):
    x = [i for i in range(1, len(matrix_configurations) + 1)]

    plt.figure(figsize=(15 + (len(matrix_configurations) - 12), 10))
    plt.xlabel("Matrix Configurations ")
    plt.ylabel("Speedup")

    plt.grid()
    
    if (mm):
        mc_labels = [str(matrix_configurations[i][:-1]) for i in range(len(matrix_configurations))]

        # CUTLASS graph
        # cutlass_y = []
        # for mc in matrix_configurations:
        #     cutlass_y.append(plotcut.get_max_tflops(mc[0], mc[1], mc[2]))

        # Obtaining speedups
        for i in range(len(provider_wise_res_mm[0])):
            for j in range(1, len(providers)):
                provider_wise_res_mm[j][i] /= provider_wise_res_mm[0][i]
            # cutlass_y[i] /= provider_wise_res_mm[0][i]
        
        for i in range(len(provider_wise_res_mm[0])):
            provider_wise_res_mm[0][i] = 1

        for i in range(len(provider_wise_res_mm)):
            plt.plot(x, provider_wise_res_mm[i], marker='o')
        
        # plt.plot(x, cutlass_y, marker='o')
            
        plt.title("Matmul " + "m_block, n_block, k_block, m_group, stages, warps")
        plt.xticks(x, mc_labels, rotation=20)
        plt.legend(providers)
        plt.show()

def write_mm(full_res_mm):
    with open('/home/ubuntu/benchmarks/mm_benchmark.txt', 'w') as f:
        for line in full_res_mm:
            f.write(line)

def run_all_matrix_configs(matrix_configurations, providers, activations):
    ps = ''
    for p in providers:
        ps += '\t\t' + p.center(SPACER)
    header = '{}\t\t{}\t\t{}'.format("M".center(SPACER), "N".center(SPACER), "K".center(SPACER))
    
    header_mm = header + ps + '\n'

    full_res_mm = [header_mm]

    provider_wise_res_mm = [[] for i in range(len(providers))]
    
    for matrix_configuration in matrix_configurations:
        temp_mm = []
        
        index = 0
        for provider in providers:
            for activation in activations:
                ans_mm, _, _ = mm.benchmark(matrix_configuration[0], matrix_configuration[1], 
                            matrix_configuration[2], provider, activation)

                if (ans_mm == None):
                    continue

                temp_mm.append(ans_mm)

                provider_wise_res_mm[index].append(ans_mm)

            index += 1

        res_base = "{}\t\t{}\t\t{}\t\t".format(str(matrix_configuration[0]).center(SPACER), str(matrix_configuration[1]).center(SPACER),
                        str(matrix_configuration[2]).center(SPACER))

        res_mm = res_base
        for i in temp_mm:
            res_mm += '\t\t{}'.format(str(i).center(SPACER))
        res_mm += '\n'

        full_res_mm.append(res_mm)

    # write_mm(full_res_mm)

    return provider_wise_res_mm

def run_all_triton_configs(matrix_configurations, providers):
    pass

if __name__ == "__main__":
    # matrix_configurations = [[2048, 2048, 2048, 1 * i] for i in [1, 2, 4, 8, 16, 32, 64, 128]]
    matrix_configurations = []

    matrix_configurations.extend([[2048, 2048, 1 * i, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]])
    # matrix_configurations.extend([[2048, 1 * i, 2048, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
    # matrix_configurations.extend([[1 * i, 2048, 2048, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])

    providers = ['cublas', 'triton']
    activations = ['leaky_relu']

    providers_temp = []
    for p in providers:
        for a in activations:
            providers_temp.append(p + ' + ' + a)

    providers.extend(providers_temp)

    provider_wise_res_mm = run_all_matrix_configs(matrix_configurations, providers, activations)
    print(provider_wise_res_mm)
    print(providers)
    plot_graphs(provider_wise_res_mm, providers, matrix_configurations, mm=True)

    # run_all_triton_configs(matrix_configurations, providers)
    