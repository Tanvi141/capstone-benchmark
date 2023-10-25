import matmul.batched_matmul as bmm
from remote_plot import plt
import seaborn as sns
import cutlass as plotcut
import triton, torch

SPACER = 25

def plot_graphs(provider_wise_res_bmm, providers, matrix_configurations, bmm=True):
    x = [i for i in range(1, len(matrix_configurations) + 1)]

    plt.figure(figsize=(15 + (len(matrix_configurations) - 12), 10))
    plt.xlabel("Matrix Configurations")
    plt.ylabel("TFLOPS")

    plt.grid()

    if (bmm):
        mc_labels = [str(matrix_configurations[i]) for i in range(len(matrix_configurations))]

        cutlass_y_batch = []
        for mc in matrix_configurations:
            cutlass_y_batch.append(plotcut.get_max_tflops_batched(mc[0], mc[1], mc[2], mc[3]))

        # Obtaining speedups
        for i in range(len(provider_wise_res_bmm[0])):
            for j in range(1, 4):
                provider_wise_res_bmm[j][i] /= provider_wise_res_bmm[0][i]
            cutlass_y_batch[i] /= provider_wise_res_bmm[0][i]
        
        for i in range(len(provider_wise_res_bmm[0])):
            provider_wise_res_bmm[0][i] = 1

        for i in range(len(provider_wise_res_bmm)):
            plt.plot(x, provider_wise_res_bmm[i], marker='o')

        plt.plot(x, cutlass_y_batch, marker='o')
        
        plt.title("Batched Matmul")
        plt.ylabel("Speedup")
        plt.xticks(x, mc_labels, rotation=20)
        plt.legend(providers + ['cutlass'])
        plt.show()

def write_bmm(full_res_bmm):
    with open('/home/ubuntu/benchmarks/bmm_benchmark.txt', 'w') as f:
        for line in full_res_bmm:
            f.write(line)

def write_mm(full_res_mm):
    with open('/home/ubuntu/benchmarks/mm_benchmark.txt', 'w') as f:
        for line in full_res_mm:
            f.write(line)

def run_all_matrix_configs(matrix_configurations, providers, activations):
    ps = ''
    for p in providers:
        ps += '\t\t' + p.center(SPACER)
    header = '{}\t\t{}\t\t{}'.format("M".center(SPACER), "N".center(SPACER), "K".center(SPACER))
    
    header_bmm = header + '\t\t{}'.format("B".center(SPACER)) + ps + '\n'
    header_mm = header + ps + '\n'

    full_res_bmm = [header_bmm]
    full_res_mm = [header_mm]

    provider_wise_res_bmm = [[] for i in range(len(providers))]
    
    for matrix_configuration in matrix_configurations:
        temp_bmm = []
        temp_mm = []
        
        index = 0
        for provider in providers:
            for activation in activations:
                ans_bmm, _, _ = bmm.benchmark(matrix_configuration[0], matrix_configuration[1], 
                            matrix_configuration[2], matrix_configuration[3], provider, activation)

                if (ans_bmm == None):
                    continue

                temp_bmm.append(ans_bmm)

                provider_wise_res_bmm[index].append(ans_bmm)

            index += 1

        res_base = "{}\t\t{}\t\t{}\t\t".format(str(matrix_configuration[0]).center(SPACER), str(matrix_configuration[1]).center(SPACER),
                        str(matrix_configuration[2]).center(SPACER))

        res_bmm = res_base + str(matrix_configuration[3]).center(SPACER)
        for i in temp_bmm:
            res_bmm += '\t\t{}'.format(str(i).center(SPACER))
        res_bmm += '\n'
        
        res_mm = res_base
        for i in temp_mm:
            res_mm += '\t\t{}'.format(str(i).center(SPACER))
        res_mm += '\n'

        full_res_bmm.append(res_bmm)
        full_res_mm.append(res_mm)

    # write_bmm(full_res_bmm)
    # write_mm(full_res_mm)

    return provider_wise_res_bmm

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

    provider_wise_res_bmm = run_all_matrix_configs(matrix_configurations, providers, activations)
    print(provider_wise_res_bmm)
    print(providers)
    plot_graphs(provider_wise_res_bmm, providers, matrix_configurations, bmm=False, mm=True)

    # run_all_triton_configs(matrix_configurations, providers)
    