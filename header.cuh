#ifndef HEADER_CUH
#define HEADER_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

// Main Structure Types

struct Pos{
    double x;
    double y;
    double z;
    __host__ __device__ Pos(double x, double y, double z) : x(x), y(y), z(z){}
    __host__ __device__ Pos() : x(0), y(0), z(0){}
    __host__ __device__ Pos operator-(const Pos& o) const {
        return Pos(x - o.x, y - o.y, z - o.z);
    }
    __host__ __device__ Pos operator+(const Pos& o) const {
        return Pos(x + o.x, y + o.y, z + o.z);
    }
    __host__ __device__ Pos operator*(double s) const {
        return Pos(x * s, y * s, z * s);
    }
    __host__ __device__ Pos operator/(double s) const {
        return Pos(x / s, y / s, z / s);
    }
    __host__ __device__ bool operator==(Pos s) const {
        const double epsilon = 1e-5;
        return fabs(x - s.x) < epsilon && fabs(y - s.y) < epsilon && fabs(z - s.z) < epsilon;
    }
    __host__ __device__ bool operator!=(Pos s) const {
        const double epsilon = 1e-5;
        return fabs(x - s.x) >= epsilon || fabs(y - s.y) >= epsilon || fabs(z - s.z) >= epsilon;
    }
};

class Object{
public:
    string name;
    double m;//mass
    Pos v;//velocity
    Pos a;//mass
    Pos pos;
    __host__ __device__ Object(double m, Pos v,Pos a, Pos pos)
        : m(m), v(v), a(a), pos(pos){}
    __host__ __device__ Pos getPosition(){
        return pos;
    }
    __host__ __device__ void setName(string name){
        this->name = name;
    }
};

struct Config {
    double space_size;
    double dt;
    double simulation_time_seconds;
    int fps;
};


// below is IO functions


// remove specific characters from a string
inline string removeChar(const string& str, char c) {
    string result = "";
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] != c) {
            result += str[i];
        }
    }
    return result;
}

// parse (x,y,z) format
inline Pos parsePosition(const string& str) {
    // Remove parentheses and spaces
    string clean = removeChar(str, '(');
    clean = removeChar(clean, ')');
    clean = removeChar(clean, ' ');

    // Split by comma
    stringstream ss(clean);
    string token;
    vector<double> values;

    while (getline(ss, token, ',')) {
        values.push_back(stod(token));
    }

    if (values.size() == 3) {
        return Pos(values[0], values[1], values[2]);
    }
    return Pos(0, 0, 0);
}

// Read configuration from file
// Format: each line should be "key=value"
// Example config.txt:
//   space_size=1000
//   dt=86400
//   simulation_time_seconds=3153600
//   fps=30
inline Config readConfig(const string& filename) {
    Config config;
    config.space_size = 1000;
    config.dt = 86400;
    config.simulation_time_seconds = 3153600;
    config.fps = 30;

    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Warning: Could not open config file: " << filename << endl;
        cerr << "Using default values" << endl;
        return config;
    }

    string line;
    while (getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Parse key=value
        size_t pos = line.find('=');
        if (pos == string::npos) continue;

        string key = line.substr(0, pos);
        string value = line.substr(pos + 1);

        // Assign to config
        if (key == "space_size") {
            config.space_size = stod(value);
        } else if (key == "dt") {
            config.dt = stod(value);
        } else if (key == "simulation_time_seconds") {
            config.simulation_time_seconds = stod(value);
        } else if (key == "fps") {
            config.fps = stoi(value);
        }
    }

    file.close();
    cout << "Loaded config from " << filename << endl;
    return config;
}

// parse CSV fields that may contain parentheses
inline vector<string> parseCSVLine(const string& line) {
    vector<string> fields;
    string current_field = "";
    int paren_depth = 0;

    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];

        if (c == '(') {
            paren_depth++;
            current_field += c;
        } else if (c == ')') {
            paren_depth--;
            current_field += c;
        } else if (c == ',' && paren_depth == 0) {
            // Comma outside parentheses - field separator
            fields.push_back(current_field);
            current_field = "";
        } else {
            current_field += c;
        }
    }

    // Add the last field
    if (!current_field.empty()) {
        fields.push_back(current_field);
    }

    return fields;
}

// Read objects from CSV file
// CSV Format:
// NAME,MASS,VELOCITY,ACCELERATION,POSITION
// object_name,mass_value,(vx,vy,vz),(ax,ay,az),(px,py,pz)
inline vector<Object> readObjectsFromCSV(const string& filename) {
    vector<Object> objects;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open CSV file: " << filename << endl;
        exit(1);
    }

    string line;
    // Skip header line
    getline(file, line);

    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<string> fields = parseCSVLine(line);

        if (fields.size() < 5) {
            cerr << "Warning: Skipping malformed line: " << line << endl;
            continue;
        }

        string name = fields[0];
        string mass_str = fields[1];
        string vel_str = fields[2];
        string acc_str = fields[3];
        string pos_str = fields[4];

        // Convert to object properties
        double mass = stod(mass_str);
        Pos velocity = parsePosition(vel_str);
        Pos acceleration = parsePosition(acc_str);
        Pos position = parsePosition(pos_str);

        Object obj(mass, velocity, acceleration, position);
        obj.setName(name);
        objects.push_back(obj);
    }

    file.close();
    cout << "Loaded " << objects.size() << " objects from " << filename << endl;
    return objects;
}

// Save objects to CSV file
// CSV Format:
// NAME,MASS,VELOCITY,ACCELERATION,POSITION
// obj0,mass,(vx,vy,vz),(ax,ay,az),(px,py,pz)
inline void saveObjectsToCSV(const string& filename, const vector<Object>& objects) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not create CSV file: " << filename << endl;
        return;
    }

    // Write header
    file << "NAME,MASS,VELOCITY,ACCELERATION,POSITION" << endl;

    // Write each object
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        file << obj.name << ","
             << obj.m << ","
             << "(" << obj.v.x << "," << obj.v.y << "," << obj.v.z << "),"
             << "(" << obj.a.x << "," << obj.a.y << "," << obj.a.z << "),"
             << "(" << obj.pos.x << "," << obj.pos.y << "," << obj.pos.z << ")"
             << endl;
    }

    file.close();
    cout << "Saved " << objects.size() << " objects to " << filename << endl;
}

// Initialize output CSV file with header (for time series data)
// CSV Format with frame number:
// FRAME,NAME,MASS,VELOCITY,ACCELERATION,POSITION
inline void initTimeSeriesCSV(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not create CSV file: " << filename << endl;
        return;
    }
    file << "FRAME,NAME,MASS,VELOCITY,ACCELERATION,POSITION" << endl;
    file.close();
}

// Append frame data to time series CSV file
inline void appendFrameToCSV(const string& filename, int frame_num, const vector<Object>& objects) {
    ofstream file(filename, ios::app);  // Open in append mode

    if (!file.is_open()) {
        cerr << "Error: Could not open CSV file for appending: " << filename << endl;
        return;
    }

    // Write each object for this frame
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        file << frame_num << ","
             << obj.name << ","
             << obj.m << ","
             << "(" << obj.v.x << "," << obj.v.y << "," << obj.v.z << "),"
             << "(" << obj.a.x << "," << obj.a.y << "," << obj.a.z << "),"
             << "(" << obj.pos.x << "," << obj.pos.y << "," << obj.pos.z << ")"
             << endl;
    }

    file.close();
}

// Time Measurement Utilities

class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }

    double elapsed_ms() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        return duration.count();
    }

    double elapsed_sec() {
        return elapsed_ms() / 1000.0;
    }
};

// GPU Time Measurement using CUDA Events
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    GPUTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    float elapsed_ms() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }

    float elapsed_sec() {
        return elapsed_ms() / 1000.0f;
    }
};

#endif // HEADER_CUH
