#include "header.cuh"
#include <iostream>
#include <cmath>
#include <vector>
#define BLOCK_SIZE 32

using namespace std;


class Physics{
public:
    double G = 6.67430e-11;
    __host__ __device__ Physics(){}

    __host__ __device__ double getNorm(Object &oi, Object &oj){
        Pos p1 = oi.getPosition();
        Pos p2 = oj.getPosition();
        Pos dp = p2 - p1;

        return sqrt(dp.x*dp.x + dp.y*dp.y + dp.z*dp.z);
    }

    __host__ __device__ Pos getVectorDistance(Object &oi, Object &oj){
        Pos p1 = oi.getPosition();
        Pos p2 = oj.getPosition();
        return p2 - p1;
    }

    __host__ __device__ Pos getUnitVector(Object &oi, Object &oj){
        Pos vec = getVectorDistance(oi, oj);
        double norm = getNorm(oi, oj);
        if(norm == 0) return Pos(0, 0, 0);
        return vec / norm;
    }

    __host__ __device__ Pos getForce(Object &oi,Object &oj){
        double r = getNorm(oi,oj);
        double F = G * (oi.m * oj.m) / (r * r + 1e-6f);
        Pos direction = getUnitVector(oi,oj);
        return direction * F;
    }



};

__global__ void reduceForceKernel(int curForceIdx, Object *d_objList, Pos *d_total_force, Physics *physicsEngine, int size){
    __shared__ Pos ds_force[BLOCK_SIZE];
    int t = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int part = (size + total_threads - 1) / total_threads;

    Pos sum(0, 0, 0);

    for(int p = 0;p < part;++p){
        int start = p * total_threads;
        int idx = start + blockDim.x * blockIdx.x + threadIdx.x;

        if(idx < size && idx != curForceIdx){
            ds_force[t] = physicsEngine->getForce(d_objList[curForceIdx], d_objList[idx]);
        }else{
            ds_force[t] = Pos(0, 0, 0);
        }
        __syncthreads();

        for(int s = BLOCK_SIZE / 2; s > 0; s >>= 1){
            if(t < s){
                ds_force[t] = ds_force[t] + ds_force[t + s];
            }
            __syncthreads();
        }

        if(t == 0) sum = sum + ds_force[0];
        __syncthreads();
    }

    if(t == 0) d_total_force[curForceIdx] = sum;
}

__global__ void computeObjectKernel(Object *d_objList,Pos *d_total_force,int size,double dt){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        Object &obj = d_objList[idx];
        obj.a = d_total_force[idx] / obj.m;
        obj.v = obj.v + obj.a * dt;
        obj.pos = obj.pos + obj.v * dt;
    }
}

class Kernel{
public:
    void computeForce(int curForceIdx,vector<Object> &objList,vector<Pos> &totalForce,Physics &physicsEngine){
        int block_size = BLOCK_SIZE;
        int n = objList.size();
        int block = (n + block_size - 1) / block_size;
        dim3 dimGrid(block,1,1);
        dim3 dimBlock(block_size,1,1);
        
        Object *d_objList;
        Pos *d_total_force;
        Physics *d_physicsEngine;
        
        cudaMalloc(&d_objList, n * sizeof(Object));
        cudaMalloc(&d_total_force, n * sizeof(Pos));
        cudaMalloc(&d_physicsEngine, sizeof(Physics));
        
        cudaMemcpy(d_objList, objList.data(), n * sizeof(Object), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_force, totalForce.data(), n * sizeof(Pos), cudaMemcpyHostToDevice);
        cudaMemcpy(d_physicsEngine, &physicsEngine, sizeof(Physics), cudaMemcpyHostToDevice);
        
        reduceForceKernel<<<dimGrid, dimBlock>>>(curForceIdx, d_objList, d_total_force, d_physicsEngine, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(totalForce.data(), d_total_force, n * sizeof(Pos), cudaMemcpyDeviceToHost);
        
        cudaFree(d_objList);
        cudaFree(d_total_force);
        cudaFree(d_physicsEngine);
    }
    void computeObject(vector<Object> &objList,vector<Pos> &totalForce,double dt){
        int block_size = BLOCK_SIZE;
        int n = objList.size();
        int block = (n + block_size - 1) / block_size;
        dim3 dimGrid(block,1,1);
        dim3 dimBlock(block_size,1,1);

        Object *d_objList;
        Pos *d_total_force;

        cudaMalloc(&d_objList, n * sizeof(Object));
        cudaMalloc(&d_total_force, n * sizeof(Pos));

        cudaMemcpy(d_objList, objList.data(), n * sizeof(Object), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_force, totalForce.data(), n * sizeof(Pos), cudaMemcpyHostToDevice);

        computeObjectKernel<<<dimGrid,dimBlock>>>(d_objList,d_total_force,n,dt);
        cudaDeviceSynchronize();

        cudaMemcpy(objList.data(), d_objList, n * sizeof(Object), cudaMemcpyDeviceToHost);

        cudaFree(d_objList);
        cudaFree(d_total_force);
    }
    
    
};

class Space{
public:
    double dt;//time step (seconds)
    Physics physicsEngine;
    Kernel kernel;
    int spaceSize;
    vector<Object> objList;
    Space(int size,double dt): dt(dt){
        spaceSize = size;
    }
    void addObject(Object o){
        objList.push_back(o);
    }

    void computeStepCPU(){
        vector<Pos> totalForce(objList.size(), Pos(0,0,0));
        //UPDATE FORCE
        for(int i = 0;i < objList.size();++i){
            for(int j = 0;j < objList.size();++j){
                if(i == j)continue;
                totalForce[i] = totalForce[i] + physicsEngine.getForce(objList[i],objList[j]);
            }
        }
        //UPDATE A V POS
        for(int i = 0;i < objList.size();++i){
            Object &obj = objList[i];
            obj.a = totalForce[i] / obj.m;
            obj.v = obj.v + obj.a * dt;
            obj.pos = obj.pos + obj.v * dt;
        }
        
    }
    void computeStepGPU(){
        vector<Pos> totalForce(objList.size(), Pos(0,0,0));
        //UPDATE FORCE
        for(int i = 0;i < objList.size();++i){
            kernel.computeForce(i,objList,totalForce,physicsEngine);
        }
        //UPDATE A V POS
        kernel.computeObject(objList,totalForce,dt);
        
    }
    void printState() {
        for (size_t i = 0; i < objList.size(); i++) {
            cout << "Obj" << i << " (" << objList[i].name << ")"
                 << " pos=(" << objList[i].pos.x << ", "
                             << objList[i].pos.y << ", "
                             << objList[i].pos.z << ")"
                 << " vel=(" << objList[i].v.x << ", "
                             << objList[i].v.y << ", "
                             << objList[i].v.z << ")"
                 << endl;
        }
        cout << "---------------------------------" << endl;
    }

    bool operator==(Space &o){
        for(int i = 0;i < objList.size();++i){
            if(
                (o.objList[i].a != objList[i].a) ||
                (o.objList[i].v != objList[i].v) ||
                (o.objList[i].pos != objList[i].pos)
            )return false;
        }
        return true;
    }

};

void run_test(){
    Space spaceCPU(1000,86400); // dt = 86400 seconds = 1 day
    Space spaceGPU(1000,86400); // dt = 86400 seconds = 1 day
    
    //Mass Velocity Acceleration Position
    Object sun(1.989e30, Pos(0, 0, 0), Pos(0, 0, 0),Pos(0, 0, 0));
    Object earth(5.972e24, Pos(0, 29783, 0), Pos(0, 0, 0), Pos(1.5e11, 0, 0));
    sun.setName("Sun");
    earth.setName("earth");

    spaceCPU.addObject(sun);
    spaceCPU.addObject(earth);

    spaceGPU.addObject(sun);
    spaceGPU.addObject(earth);

    for (int step = 0; step < 10; step++) {
        cout << "Step " << step << endl;
        cout << "CPU" << endl;
        spaceCPU.printState();
        cout << "GPU" << endl;
        spaceGPU.printState();
        spaceCPU.computeStepCPU();
        spaceGPU.computeStepGPU();
        cout << "Verify : " << (spaceCPU == spaceGPU ? "TRUE ✅" : "FALSE : ❌") << endl;
        
        
    }
}

void run(){
    Config config = readConfig("config.txt");
    vector<Object> objects = readObjectsFromCSV("object_input.csv");

    Space spaceGPU(config.space_size, config.dt);
    Space spaceCPU(config.space_size, config.dt);

    for (const Object& obj : objects) {
        spaceGPU.addObject(obj);
        spaceCPU.addObject(obj);
    }

    // Calculate based on video duration and framerate
    int total_frames = (int)(config.simulation_time_seconds * config.fps);
    int total_steps = total_frames;
    int steps_per_frame = 1;

    cout << "Running simulation..." << endl;
    cout << "Video duration: " << config.simulation_time_seconds << " seconds at " << config.fps << " fps" << endl;
    cout << "Total frames: " << total_frames << endl;
    cout << "Total steps: " << total_steps << endl;
    cout << "Time per step (dt): " << config.dt << " seconds (" << config.dt/86400.0 << " days)" << endl;
    cout << "Total simulated time: " << (total_steps * config.dt) << " seconds (" << (total_steps * config.dt)/86400.0 << " days)" << endl;

    // Initialize output CSV files
    initTimeSeriesCSV("output_gpu.csv");
    initTimeSeriesCSV("output_cpu.csv");

    //simulation
    for (int step = 0; step < total_steps; step++) {
        if (step % steps_per_frame == 0) {
            int frame_num = step / steps_per_frame;
            cout << "Frame " << frame_num << " (Step " << step << ")" << endl;
            // cout << "CPU" << endl;
            // spaceCPU.printState();
            // cout << "GPU" << endl;
            // spaceGPU.printState();

            appendFrameToCSV("output_cpu.csv", frame_num, spaceCPU.objList);
            appendFrameToCSV("output_gpu.csv", frame_num, spaceGPU.objList);
        }

        spaceCPU.computeStepCPU();
        spaceGPU.computeStepGPU();

        // if (step % steps_per_frame == 0) {
        //     cout << "Verify : " << (spaceCPU == spaceGPU ? "TRUE ✅" : "FALSE : ❌") << endl;
        // }
    }

    cout << "\nSimulation complete! Output saved to:" << endl;
    cout << "  - output_gpu.csv (GPU results)" << endl;
    cout << "  - output_cpu.csv (CPU results)" << endl;

}

void run_benchmark(){
    Config config = readConfig("config.txt");
    vector<Object> objects = readObjectsFromCSV("object_input.csv");

    Space spaceGPU(config.space_size, config.dt);
    Space spaceCPU(config.space_size, config.dt);

    for (const Object& obj : objects) {
        spaceGPU.addObject(obj);
        spaceCPU.addObject(obj);
    }

    int total_frames = (int)(config.simulation_time_seconds * config.fps);
    int total_steps = total_frames;

    cout << "=== BENCHMARK MODE ===" << endl;
    cout << "Objects: " << objects.size() << endl;
    cout << "Steps: " << total_steps << endl;
    cout << "Time per step: " << config.dt/86400.0 << " days" << endl;
    cout << "\n";

    // CPU Benchmark
    cout << "Running CPU simulation..." << endl;
    Timer cpuTimer;
    cpuTimer.start();

    for (int step = 0; step < total_steps; step++) {
        if (step % 10 == 0) {
            cout << "  CPU Progress: " << step << "/" << total_steps << "\r" << flush;
        }
        spaceCPU.computeStepCPU();
    }

    double cpuTime = cpuTimer.elapsed_sec();
    cout << "\nCPU Time: " << cpuTime << " seconds" << endl;
    cout << "CPU Speed: " << total_steps / cpuTime << " steps/second" << endl;
    cout << "\n";

    // GPU Benchmark
    cout << "Running GPU simulation..." << endl;
    GPUTimer gpuTimer;
    gpuTimer.start();

    for (int step = 0; step < total_steps; step++) {
        if (step % 10 == 0) {
            cout << "  GPU Progress: " << step << "/" << total_steps << "\r" << flush;
        }
        spaceGPU.computeStepGPU();
    }

    float gpuTime = gpuTimer.elapsed_sec();
    cout << "\nGPU Time: " << gpuTime << " seconds" << endl;
    cout << "GPU Speed: " << total_steps / gpuTime << " steps/second" << endl;
    cout << "\n";

    // Results
    cout << "=== RESULTS ===" << endl;
    cout << "CPU Time: " << cpuTime << " s" << endl;
    cout << "GPU Time: " << gpuTime << " s" << endl;
    cout << "Speedup: " << (cpuTime / gpuTime) << "x" << endl;
}

int main(){
    bool test = false;
    bool benchmark = true;

    if(test) run_test();
    else if(benchmark) run_benchmark();
    else run();

    return 0;
}