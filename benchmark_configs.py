import benchmark
import sys 

def run_benchmarkers():
    # matrix_configurations = [[2048, 2048, 2048, 1 * i] for i in [1, 2, 4, 8, 16, 32, 64, 128]]
    matrix_configurations = []

    matrix_configurations.extend([[2048, 2048, 1 * i, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
    # matrix_configurations.extend([[2048, 1 * i, 2048, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
    # matrix_configurations.extend([[1 * i, 2048, 2048, 128] for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])

    providers = ["triton"]
    activations = ["leaky_relu"]
    # activations = [None]

    providers_temp = []
    for p in providers:
        for a in activations:
            if a is not None:
                providers_temp.append(p + ' + ' + a)

    providers.extend(providers_temp)
    
    provider_wise_res_mm = benchmark.run_all_matrix_configs(matrix_configurations, providers, activations)
    print(provider_wise_res_mm)
    if len(provider_wise_res_mm[0]) != len(matrix_configurations):
        raise Exception("Could not run configuration")

    with open ("benchmark_results.txt", "a") as f:
        for i in range(len(providers)):
            for typ, results in zip(["m"], [provider_wise_res_mm[i]]):
                for config in range(len(matrix_configurations)):
                    f.write(providers[i]+",")
                    f.write(typ+",")
                    print(matrix_configurations[config])
                    for sub_config in matrix_configurations[config]:
                        f.write(str(sub_config)+",")
                    print(len(results), results)
                    f.write(str(results[config])+",")
                    f.write("\n")

if __name__ == "__main__":
    if len(sys.argv[1:]) > 1:
        raise Exception("Too many arguments")
    elif len(sys.argv[1:]) == 1:
        try:
            run_benchmarkers()
        except Exception as e:
            print("Error", e)
    else:
        pass