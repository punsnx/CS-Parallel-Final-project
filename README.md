# Gravitational N-Bodies Simulation
- By 6610402230 Sirisuk Tharntham
- the project is to simulate object movement orbit in vector vector space with parallel algorithm optimization with CUDA


## Compile and Run
1. add object to object_input.csv
2. set the config.txt for application paramerters
3. Compile and Run the application
```
❯ nvcc -arch=sm_80 main_gpu.cu -o app.out && ./app.out
```

## Visualization
1.Run all visualize.ipynb for simulate from calculated object positons in csv files to mp4
2.simulate_input.ipynb use for simulate test input to benchmark CPU and GPU code you can edit size of input for generate then run all to generate 
