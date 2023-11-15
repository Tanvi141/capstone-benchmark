import benchmark
import sys 
matrix_configurations = []

# matrix_configurations.extend([[2048, 2048, 1 * i, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
# matrix_configurations.extend([[2048, 1 * i, 2048, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
matrix_configurations.extend([[1 * i, 2048, 64, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])


def run_benchmarkers():
    # matrix_configurations = [[2048, 2048, 2048, 1 * i] for i in [1, 2, 4, 8, 16, 32, 64, 128]]
    
    providers = ["triton", "cublas"]
    activations = ["leaky_relu"]
    # activations = [None]

    providers_temp = []
    for p in providers:
        for a in activations:
            if a is not None:
                providers_temp.append(p + ' + ' + a)

    providers.extend(providers_temp)
    
    provider_wise_res_mm = benchmark.run_all_matrix_configs(matrix_configurations, providers, activations)
    # print(provider_wise_res_mm)
    if len(provider_wise_res_mm[0]) != len(matrix_configurations):
        raise Exception("Could not run configuration")

    with open ("benchmark_results.txt", "a") as f:
        for i in range(len(providers)):
            for typ, results in zip(["m"], [provider_wise_res_mm[i]]):
                for config in range(len(matrix_configurations)):
                    f.write(providers[i]+",")
                    f.write(typ+",")
                    f.write(sys.argv[1]+",")
                    # print(matrix_configurations[config])
                    for sub_config in matrix_configurations[config]:
                        f.write(str(sub_config)+",")
                    # print(len(results), results)
                    f.write(str(results[config])+",")
                    f.write("\n")

def plot_configs():
    results = {}
    results["cublas + leaky_relu"] = {tuple(k): -1 for k in matrix_configurations}
    with open("benchmark_results.txt", "r") as f:
        line = f.readline().strip()
        while line:
            vals = line.split(",")
        # cublas,m,64,64,32,8,2,4,2048,2048,1,128,0.431,

            provider, typ, config_str, m, n, k, b, tflops, _ = vals
            # config_str = m_block+","+n_block+","+k_block+","+m_group+","+stages+","+warps
            
            if "+" not in provider:
                line = f.readline().strip()
                continue

            if provider.startswith("cublas"):
                plotline = provider
            else:
                plotline = config_str
            
            if plotline not in results:
                results[plotline] = {tuple(k): -1 for k in matrix_configurations}
            matrix_config = (int(m), int(n), int(k), int(b))
        
            results[plotline][matrix_config] = float(tflops)
            line = f.readline().strip()
    print(results)


    providers = results.keys()
    print(providers)
    provider_wise_res_mm = []
    for provider in providers:
        provider_res = []
        for dim in matrix_configurations:
            provider_res.append(results[provider][tuple(dim)])
        provider_wise_res_mm.append(provider_res)
    benchmark.plot_graphs(provider_wise_res_mm, providers, matrix_configurations, mm=True)

if __name__ == "__main__":
    if len(sys.argv[1:]) > 1:
        raise Exception("Too many arguments")
    elif len(sys.argv[1:]) == 1:
        try:
            run_benchmarkers()
        except Exception as e:
            print("Error", e)
    else:
        plot_configs()
        