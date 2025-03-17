#define _USE_MATH_DEFINES // for C++
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include "CurveActiveForceLength.h"
#include "CurveFiberForceLength.h"

// Simplified muscle force function
double muscleForce(double lMtilde, double act = 1.0, double pennation = 0.0) {
    // Create curve objects (static to avoid recreating them for each call)
    static CurveActiveForceLength afl_curve;
    static CurveFiberForceLength pfl_curve;
    
    // Calculate active and passive components
    double afl = afl_curve.calcValue(lMtilde);
    double pfl = pfl_curve.calcValue(lMtilde);
    
    // Total muscle force
    double fM = act * afl + pfl;
    
    // Apply pennation angle (converting degrees to radians)
    return fM * cos(pennation);
}

// Function to write data in a columnar format that's extremely fast to read
// This format is essentially a binary CSV with columns: lMtilde, activation, pennation, force
void writeColumnarData(const std::string& filename, 
                      const std::vector<double>& lm_values,
                      const std::vector<double>& activation_values,
                      const std::vector<double>& pennation_values) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to create file: " << filename << std::endl;
        return;
    }
    
    // Write a simple header with format version and column count
    const char* header = "COLDAT01";  // 8-byte signature + version
    file.write(header, 8);
    
    // Write column names length and column names as strings
    const char* col_names = "lMtilde,activation,pennation,force";
    uint32_t col_names_len = static_cast<uint32_t>(strlen(col_names));
    file.write(reinterpret_cast<const char*>(&col_names_len), sizeof(uint32_t));
    file.write(col_names, col_names_len);
    
    // Write number of rows
    uint64_t num_rows = static_cast<uint64_t>(lm_values.size()) * 
                      static_cast<uint64_t>(activation_values.size()) * 
                      static_cast<uint64_t>(pennation_values.size());
    file.write(reinterpret_cast<const char*>(&num_rows), sizeof(uint64_t));
    
    // Write column sizes
    uint32_t num_columns = 4;  // lMtilde, activation, pennation, force
    file.write(reinterpret_cast<const char*>(&num_columns), sizeof(uint32_t));
    
    // Write data type (float32) for each column
    for (uint32_t i = 0; i < num_columns; ++i) {
        uint32_t data_type = 1;  // 1 = float32
        file.write(reinterpret_cast<const char*>(&data_type), sizeof(uint32_t));
    }
    
    // Allocate a buffer to store data - we'll write in chunks for efficiency
    const uint64_t chunk_size = 10000;  // Number of rows per chunk
    uint64_t total_chunks = (num_rows + chunk_size - 1) / chunk_size;
    
    std::vector<float> buffer(chunk_size * num_columns);
    
    // Precompute the total size of each dimension for index calculation
    uint64_t act_size = activation_values.size();
    uint64_t pen_size = pennation_values.size();
    
    // Process each chunk
    uint64_t processed_rows = 0;
    uint64_t chunk_row_count = 0;
    size_t buffer_idx = 0;
    
    std::cout << "Generating " << num_rows << " data points in " << total_chunks << " chunks..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (uint64_t chunk = 0; chunk < total_chunks; ++chunk) {
        // Fix: ensure consistent types by casting everything to uint64_t
        chunk_row_count = static_cast<uint64_t>(
            std::min(chunk_size, num_rows - processed_rows));
        buffer_idx = 0;
        
        for (uint64_t i = 0; i < chunk_row_count; ++i) {
            // Calculate 3D indices from flat index
            uint64_t row_idx = processed_rows + i;
            uint64_t lm_idx = row_idx / (act_size * pen_size);
            uint64_t act_idx = (row_idx / pen_size) % act_size;
            uint64_t pen_idx = row_idx % pen_size;
            
            // Get actual values
            double lm = lm_values[lm_idx];
            double act = activation_values[act_idx];
            double pen = pennation_values[pen_idx];
            
            // Calculate force
            double force = muscleForce(lm, act, pen);
            
            // Write to buffer
            buffer[buffer_idx++] = static_cast<float>(lm);
            buffer[buffer_idx++] = static_cast<float>(act);
            buffer[buffer_idx++] = static_cast<float>(pen);
            buffer[buffer_idx++] = static_cast<float>(force);
        }
        
        // Write buffer to file
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer_idx * sizeof(float));
        processed_rows += chunk_row_count;
        
        // Print progress
        if (chunk % 10 == 0 || chunk == total_chunks - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double progress = static_cast<double>(processed_rows) / num_rows;
            double eta = elapsed / progress - elapsed;
            
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                      << (progress * 100) << "% (" << processed_rows << "/" << num_rows 
                      << ") - ETA: " << static_cast<int>(eta) << "s   " << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    file.close();
    
    std::cout << "\nGenerated " << num_rows << " data points in " << duration << " seconds" << std::endl;
    std::cout << "Average generation rate: " << (num_rows / duration) << " points/second" << std::endl;
    std::cout << "Data written to: " << filename << std::endl;
    
    // Get file size
    std::ifstream size_check(filename, std::ios::binary | std::ios::ate);
    if (size_check) {
        std::streamsize file_size = size_check.tellg();
        size_check.close();
        std::cout << "File size: " << (file_size / (1024 * 1024)) << " MB" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t lm_samples = 1000;
    size_t act_samples = 1000;
    size_t pen_samples = 1000;
    double lm_min = 0.5;
    double lm_max = 3.0;
    double mu = 2.0;
    double sigma = 0.8;
    bool use_normal_dist = true;
    std::string output_file = "muscle_data_columnar.dat";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--lm-samples" && i + 1 < argc) {
            lm_samples = std::stoi(argv[++i]);
        } else if (arg == "--act-samples" && i + 1 < argc) {
            act_samples = std::stoi(argv[++i]);
        } else if (arg == "--pen-samples" && i + 1 < argc) {
            pen_samples = std::stoi(argv[++i]);
        } else if (arg == "--lm-min" && i + 1 < argc) {
            lm_min = std::stod(argv[++i]);
        } else if (arg == "--lm-max" && i + 1 < argc) {
            lm_max = std::stod(argv[++i]);
        } else if (arg == "--mu" && i + 1 < argc) {
            mu = std::stod(argv[++i]);
        } else if (arg == "--sigma" && i + 1 < argc) {
            sigma = std::stod(argv[++i]);
        } else if (arg == "--uniform") {
            use_normal_dist = false;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --lm-samples N       Number of muscle length samples (default: " << lm_samples << ")\n"
                      << "  --act-samples N      Number of activation samples (default: " << act_samples << ")\n"
                      << "  --pen-samples N      Number of pennation angle samples (default: " << pen_samples << ")\n"
                      << "  --lm-min X           Minimum muscle length if using uniform distribution (default: " << lm_min << ")\n"
                      << "  --lm-max X           Maximum muscle length if using uniform distribution (default: " << lm_max << ")\n"
                      << "  --mu X               Mean of normal distribution for muscle length (default: " << mu << ")\n"
                      << "  --sigma X            Standard deviation for normal distribution (default: " << sigma << ")\n"
                      << "  --uniform            Use uniform distribution for muscle length instead of normal\n"
                      << "  --output FILE        Output file name (default: " << output_file << ")\n";
            return 0;
        }
    }
    
    std::cout << "Generating muscle force data with:\n"
              << "  lM samples: " << lm_samples << "\n"
              << "  Activation samples: " << act_samples << "\n"
              << "  Pennation samples: " << pen_samples << "\n";
    
    if (use_normal_dist) {
        std::cout << "  lM distribution: Normal with mean " << mu << " and std dev " << sigma << "\n";
    } else {
        std::cout << "  lM distribution: Uniform from " << lm_min << " to " << lm_max << "\n";
    }
    
    std::cout << "  Output file: " << output_file << "\n\n";
    
    // Calculate total data points and estimated size
    uint64_t total_points = static_cast<uint64_t>(lm_samples) * 
                           static_cast<uint64_t>(act_samples) * 
                           static_cast<uint64_t>(pen_samples);
    double estimated_size_mb = total_points * (4 * sizeof(float)) / (1024.0 * 1024.0);
    
    std::cout << "Total data points: " << total_points << "\n";
    std::cout << "Estimated file size: " << std::fixed << std::setprecision(2) << estimated_size_mb << " MB\n\n";
    
    // Generate muscle length values
    std::vector<double> lm_values(lm_samples);
    
    if (use_normal_dist) {
        // Use normal distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(mu, sigma);
        
        for (size_t i = 0; i < lm_samples; ++i) {
            lm_values[i] = dist(gen);
            // Clip values to valid range [0.0, 5.0]
            if (lm_values[i] < 0.0) lm_values[i] = 0.0;
            if (lm_values[i] > 5.0) lm_values[i] = 5.0;
        }
        
        // Sort values for better cache locality
        std::sort(lm_values.begin(), lm_values.end());
    } else {
        // Use uniform distribution
        for (size_t i = 0; i < lm_samples; ++i) {
            lm_values[i] = lm_min + (lm_max - lm_min) * i / (lm_samples - 1);
        }
    }
    
    // Generate activation values (0 to 1)
    std::vector<double> activation_values(act_samples);
    for (size_t i = 0; i < act_samples; ++i) {
        activation_values[i] = static_cast<double>(i) / (act_samples - 1);
    }
    
    // Generate pennation values (0 to 0.7)
    std::vector<double> pennation_values(pen_samples);
    for (size_t i = 0; i < pen_samples; ++i) {
        pennation_values[i] = 0.7 * static_cast<double>(i) / (pen_samples - 1);
    }
    
    // Generate and write data
    writeColumnarData(output_file, lm_values, activation_values, pennation_values);
    
    return 0;
}