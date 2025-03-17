#define _USE_MATH_DEFINES // for C++
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include "CurveActiveForceLength.h"
#include "CurveFiberForceLength.h"

// Simplified muscle force function (unchanged)
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

// Function to generate NPY header
std::string generateNpyHeader(size_t num_elements, bool is_fortran_order = false, const std::string& dtype = "<f4") {
    // Magic string and version
    std::string header = "\x93NUMPY";
    header.push_back(0x01); // major version
    header.push_back(0x00); // minor version
    
    // Generate dict header
    std::ostringstream dict_header;
    dict_header << "{'descr': '" << dtype << "', 'fortran_order': ";
    dict_header << (is_fortran_order ? "True" : "False");
    dict_header << ", 'shape': (" << num_elements << ",), }";

    // Pad with spaces to make header length a multiple of 16
    // -10 for the magic string, version, and header length
    size_t header_len_so_far = 10 + dict_header.str().length();
    size_t padded_len = ((header_len_so_far + 15) / 16) * 16;
    for (size_t i = header_len_so_far; i < padded_len; i++) {
        dict_header << ' ';
    }
    
    // Generate final header
    std::string dict_header_str = dict_header.str();
    
    // Compute header length as uint16
    uint16_t header_len = static_cast<uint16_t>(dict_header_str.length());
    
    // Add header length to header (little-endian)
    header.push_back(header_len & 0xFF);
    header.push_back((header_len >> 8) & 0xFF);
    
    // Add dict header
    header += dict_header_str;
    
    return header;
}

// Function to write data to NPY file
void writeNpyFile(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to create file: " << filename << std::endl;
        return;
    }
    
    // Generate and write header
    std::string header = generateNpyHeader(data.size());
    file.write(header.c_str(), header.length());
    
    // Write data
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    file.close();
    
    // Get file size
    std::ifstream size_check(filename, std::ios::binary | std::ios::ate);
    if (size_check) {
        std::streamsize file_size = size_check.tellg();
        size_check.close();
        std::cout << "File " << filename << " size: " << (file_size / (1024 * 1024)) << " MB" << std::endl;
    }
}

// Function to generate data and write to NPY files
void generateAndWriteNpyData(const std::string& output_dir,
                           const std::vector<double>& lm_values,
                           const std::vector<double>& activation_values,
                           const std::vector<double>& pennation_values) {
    
    // Calculate total number of data points
    uint64_t num_rows = static_cast<uint64_t>(lm_values.size()) * 
                      static_cast<uint64_t>(activation_values.size()) * 
                      static_cast<uint64_t>(pennation_values.size());
    
    // Create output files
    std::string lm_file = output_dir + "/lMtilde.npy";
    std::string act_file = output_dir + "/activation.npy";
    std::string pen_file = output_dir + "/pennation.npy";
    std::string force_file = output_dir + "/force.npy";
    
    // Allocate a buffer to store data - we'll write in chunks for efficiency
    const uint64_t chunk_size = 10000000;  // Number of rows per chunk
    uint64_t total_chunks = (num_rows + chunk_size - 1) / chunk_size;
    
    // Create output streams
    std::ofstream lm_out(lm_file, std::ios::binary);
    std::ofstream act_out(act_file, std::ios::binary);
    std::ofstream pen_out(pen_file, std::ios::binary);
    std::ofstream force_out(force_file, std::ios::binary);
    
    if (!lm_out || !act_out || !pen_out || !force_out) {
        std::cerr << "Error: Unable to create output files" << std::endl;
        return;
    }
    
    // Write NPY headers
    std::string lm_header = generateNpyHeader(num_rows);
    std::string act_header = generateNpyHeader(num_rows);
    std::string pen_header = generateNpyHeader(num_rows);
    std::string force_header = generateNpyHeader(num_rows);
    
    lm_out.write(lm_header.c_str(), lm_header.length());
    act_out.write(act_header.c_str(), act_header.length());
    pen_out.write(pen_header.c_str(), pen_header.length());
    force_out.write(force_header.c_str(), force_header.length());
    
    // Buffers for chunk data
    std::vector<float> lm_buffer(chunk_size);
    std::vector<float> act_buffer(chunk_size);
    std::vector<float> pen_buffer(chunk_size);
    std::vector<float> force_buffer(chunk_size);
    
    // Precompute the total size of each dimension for index calculation
    uint64_t act_size = activation_values.size();
    uint64_t pen_size = pennation_values.size();
    
    // Process each chunk
    uint64_t processed_rows = 0;
    
    std::cout << "Generating " << num_rows << " data points in " << total_chunks << " chunks..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (uint64_t chunk = 0; chunk < total_chunks; ++chunk) {
        uint64_t chunk_row_count = std::min(chunk_size, num_rows - processed_rows);
        
        // Fill buffers for this chunk
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
            
            // Store in buffers
            lm_buffer[i] = static_cast<float>(lm);
            act_buffer[i] = static_cast<float>(act);
            pen_buffer[i] = static_cast<float>(pen);
            force_buffer[i] = static_cast<float>(force);
        }
        
        // Write buffers to files
        lm_out.write(reinterpret_cast<const char*>(lm_buffer.data()), chunk_row_count * sizeof(float));
        act_out.write(reinterpret_cast<const char*>(act_buffer.data()), chunk_row_count * sizeof(float));
        pen_out.write(reinterpret_cast<const char*>(pen_buffer.data()), chunk_row_count * sizeof(float));
        force_out.write(reinterpret_cast<const char*>(force_buffer.data()), chunk_row_count * sizeof(float));
        
        processed_rows += chunk_row_count;
        
        // Print progress with a nicer progress bar
        if (chunk % 5 == 0 || chunk == total_chunks - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double progress = static_cast<double>(processed_rows) / num_rows;
            double eta = elapsed / progress - elapsed;
            
            // Create a progress bar
            const int barWidth = 50;
            int pos = static_cast<int>(barWidth * progress);
            
            std::cout << "\r[";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            
            std::cout << "] " << std::fixed << std::setprecision(1) 
                      << (progress * 100) << "% (" << processed_rows << "/" << num_rows 
                      << ") - ETA: " << static_cast<int>(eta) << "s   " << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // Close files
    lm_out.close();
    act_out.close();
    pen_out.close();
    force_out.close();
    
    std::cout << "\nGenerated " << num_rows << " data points in " << duration << " seconds" << std::endl;
    std::cout << "Average generation rate: " << (num_rows / duration) << " points/second" << std::endl;
    std::cout << "Data written to: " << output_dir << std::endl;
    
    // Get file sizes
    std::ifstream lm_check(lm_file, std::ios::binary | std::ios::ate);
    std::ifstream act_check(act_file, std::ios::binary | std::ios::ate);
    std::ifstream pen_check(pen_file, std::ios::binary | std::ios::ate);
    std::ifstream force_check(force_file, std::ios::binary | std::ios::ate);
    
    if (lm_check && act_check && pen_check && force_check) {
        std::streamsize lm_size = lm_check.tellg();
        std::streamsize act_size = act_check.tellg();
        std::streamsize pen_size = pen_check.tellg();
        std::streamsize force_size = force_check.tellg();
        
        std::cout << "File sizes:" << std::endl;
        std::cout << "  lMtilde.npy: " << (lm_size / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  activation.npy: " << (act_size / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  pennation.npy: " << (pen_size / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  force.npy: " << (force_size / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Total: " << ((lm_size + act_size + pen_size + force_size) / (1024 * 1024)) << " MB" << std::endl;
    }
}

// Function to create directory if it doesn't exist
bool createDirectory(const std::string& path) {
    #ifdef _WIN32
    int result = system(("mkdir " + path).c_str());
    #else
    int result = system(("mkdir -p " + path).c_str());
    #endif
    return result == 0;
}

int main(int argc, char* argv[]) {
    // Default parameters (unchanged)
    size_t lm_samples = 1000;
    size_t act_samples = 1000;
    size_t pen_samples = 1000;
    double lm_min = 0.5;
    double lm_max = 3.0;
    double mu = 2.0;
    double sigma = 0.8;
    bool use_normal_dist = true;
    std::string output_dir = "muscle_data_npy";
    
    // Parse command line arguments (unchanged)
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
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
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
                      << "  --output-dir DIR     Output directory name (default: " << output_dir << ")\n";
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
    
    std::cout << "  Output directory: " << output_dir << "\n\n";
    
    // Create output directory
    if (!createDirectory(output_dir)) {
        std::cerr << "Error: Unable to create output directory: " << output_dir << std::endl;
        return 1;
    }
    
    // Calculate total data points and estimated size
    uint64_t total_points = static_cast<uint64_t>(lm_samples) * 
                           static_cast<uint64_t>(act_samples) * 
                           static_cast<uint64_t>(pen_samples);
    double estimated_size_mb = total_points * (4 * sizeof(float)) / (1024.0 * 1024.0);
    
    std::cout << "Total data points: " << total_points << "\n";
    std::cout << "Estimated file size: " << std::fixed << std::setprecision(2) << estimated_size_mb << " MB\n\n";
    
    // Generate muscle length values (unchanged)
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
    
    // Generate activation values (0 to 1) (unchanged)
    std::vector<double> activation_values(act_samples);
    for (size_t i = 0; i < act_samples; ++i) {
        activation_values[i] = static_cast<double>(i) / (act_samples - 1);
    }
    
    // Generate pennation values (0 to 0.7) (unchanged)
    std::vector<double> pennation_values(pen_samples);
    for (size_t i = 0; i < pen_samples; ++i) {
        pennation_values[i] = 0.7 * static_cast<double>(i) / (pen_samples - 1);
    }
    
    // Generate and write data
    generateAndWriteNpyData(output_dir, lm_values, activation_values, pennation_values);
    
    return 0;
}