#include "memory_allocation.h"

__global__ void add(int *d_a, int *d_b, int *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        h_c[i] = d_a[i] + d_b[i];
}

__global__ void sub(int *d_a, int *d_b, int *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        h_c[i] = d_a[i] - d_b[i];
}

__global__ void mult(int *d_a, int *d_b, int *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        h_c[i] = d_a[i] * d_b[i];
}

__global__ void mod(int *d_a, int *d_b, int *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        h_c[i] = d_a[i] % d_b[i];
}

// Allocate host input A (pageable) and B (pinned)
__host__ std::tuple<int *, int *> allocateRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(int);
    
    int *h_a = (int *)malloc(size);             // Pageable
    int *h_b; cudaMallocHost((void **)&h_b, size);  // Pinned

    for (int i = 0; i < numElements; ++i)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    return {h_a, h_b};
}

// Read input from CSV
__host__ std::tuple<int *, int *, int> readCsv(std::string filename)
{
    std::vector<int> tempResult;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open file");

    std::string line;
    int val;

    getline(file, line);
    std::stringstream ss1(line);
    while (ss1 >> val)
    {
        tempResult.push_back(val);
        if (ss1.peek() == ',') ss1.ignore();
    }

    int numElements = tempResult.size();
    int *h_a = (int *)malloc(numElements * sizeof(int));
    std::copy(tempResult.begin(), tempResult.end(), h_a);
    tempResult.clear();

    getline(file, line);
    std::stringstream ss2(line);
    while (ss2 >> val)
    {
        tempResult.push_back(val);
        if (ss2.peek() == ',') ss2.ignore();
    }

    int *h_b;
    cudaMallocHost((int **)&h_b, numElements * sizeof(int));
    std::copy(tempResult.begin(), tempResult.end(), h_b);
    file.close();

    return {h_a, h_b, numElements};
}

// Allocate device memory
__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements)
{
    int *d_a, *d_b;
    size_t size = numElements * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    return {d_a, d_b};
}

// Copy host to device memory
__host__ void copyFromHostToDevice(int *h_a, int *h_b, int *d_a, int *d_b, int numElements)
{
    size_t size = numElements * sizeof(int);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
}

// Launch kernel
__host__ void executeKernel(int *d_a, int *d_b, int *h_c, int numElements, int threadsPerBlock, std::string op)
{
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    if (op == "sub")
        sub<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);
    else if (op == "mult")
        mult<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);
    else if (op == "mod")
        mod<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);
    else
        add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Deallocate memory
__host__ void deallocateMemory(int *d_a, int *d_b)
{
    cudaFree(d_a);
    cudaFree(d_b);
}

// Reset device
__host__ void cleanUpDevice()
{
    cudaDeviceReset();
}

// Output result to file
__host__ void outputToFile(std::string partId, int *h_a, int *h_b, int *h_c, int n, std::string op)
{
    std::ofstream file("output-" + partId + ".txt", std::ofstream::app);

    file << "Mathematical Operation: " << op << "\n";
    file << "PartID: " << partId << "\n";
    file << "Input A: ";
    for (int i = 0; i < n; ++i) file << h_a[i] << " ";
    file << "\nInput B: ";
    for (int i = 0; i < n; ++i) file << h_b[i] << " ";
    file << "\nResult: ";
    for (int i = 0; i < n; ++i) file << h_c[i] << " ";
    file << "\n";
    file.close();
}

// Parse command line args
__host__ std::tuple<int, std::string, int, std::string, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int numElements = 10, threadsPerBlock = 256;
    std::string partId = "test", op = "add", file = "NULL";

    for (int i = 1; i < argc; i += 2)
    {
        std::string opt(argv[i]), val(argv[i+1]);
        if (opt == "-n") numElements = std::stoi(val);
        else if (opt == "-p") partId = val;
        else if (opt == "-t") threadsPerBlock = std::stoi(val);
        else if (opt == "-o") op = val;
        else if (opt == "-f") file = val;
    }

    return {numElements, partId, threadsPerBlock, file, op};
}

// Input generator (random or CSV)
__host__ std::tuple<int *, int *, int> setUpInput(std::string file, int numElements)
{
    if (file != "NULL")
    {
        return readCsv(file);
    }
    else
    {
        auto [a, b] = allocateRandomHostMemory(numElements);
        return {a, b, numElements};
    }
}

int main(int argc, char *argv[])
{
    auto [numElements, partId, threadsPerBlock, file, op] = parseCommandLineArguments(argc, argv);
    auto [h_a, h_b, n] = setUpInput(file, numElements);
    numElements = n;

    int *h_c;
    cudaMallocManaged(&h_c, numElements * sizeof(int));  // Unified memory

    auto [d_a, d_b] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_a, h_b, d_a, d_b, numElements);
    executeKernel(d_a, d_b, h_c, numElements, threadsPerBlock, op);
    cudaDeviceSynchronize();  // Needed for unified memory

    outputToFile(partId, h_a, h_b, h_c, numElements, op);
    deallocateMemory(d_a, d_b);
    cudaFree(h_c);
    cleanUpDevice();

    return 0;
}
