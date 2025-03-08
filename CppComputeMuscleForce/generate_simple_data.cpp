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
#include "CurveActiveForceLength.h"
#include "CurveFiberForceLength.h"

// Simplified muscle force function equivalent to your Python version
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
    return fM * cos(pennation * M_PI / 180.0);
}

// Function to create a file and write header information
bool initializeOutputFile(const std::string& filename, 
                         uint32_t dim1, uint32_t dim2, uint32_t dim3) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to create or open file '" << filename << "'" << std::endl;
        return false;
    }
    
    // Write a simple header to identify the file format
    const char* header = "NPZDATA";
    file.write(header, 7);
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&dim1), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&dim2), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&dim3), sizeof(uint32_t));
    
    file.close();
    return true;
}

// Function to append a batch of data to the output file
bool appendBatchToFile(const std::string& filename,
                       const std::vector<double>& lm_values,
                       const std::vector<double>& activation_range,
                       const std::vector<double>& pennation_range,
                       int start_lm_idx, int end_lm_idx,
                       int start_act_idx, int end_act_idx,
                       int start_pen_idx, int end_pen_idx,
                       bool writing_L, bool writing_A, bool writing_P, bool writing_F) {
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file) {
        std::cerr << "Error: Unable to open file for appending: " << filename << std::endl;
        return false;
    }
    
    for (int i = start_lm_idx; i < end_lm_idx; ++i) {
        for (int j = start_act_idx; j < end_act_idx; ++j) {
            for (int k = start_pen_idx; k < end_pen_idx; ++k) {
                
                if (writing_L) {
                    // L array (muscle length)
                    double val = lm_values[i];
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
                
                if (writing_A) {
                    // A array (activation)
                    double val = activation_range[j];
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
                
                if (writing_P) {
                    // P array (pennation)
                    double val = pennation_range[k];
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
                
                if (writing_F) {
                    // F array (force)
                    double val = muscleForce(lm_values[i], activation_range[j], pennation_range[k]);
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }
        }
    }
    
    file.close();
    return true;
}

// Simple function to save data to CSV for debugging/visualization
void saveToCSV(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    file << "value" << std::endl;
    for (const auto& val : data) {
        file << val << std::endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    // Default parameters
    int sample_size = 2000;  // Number of muscle lengths
    int act_samples = 1000; // Number of activation samples
    int pen_samples = 1000; // Number of pennation samples
    double mu = 2.0;       // Mean for normal distribution
    double sigma = 0.8;    // Standard deviation
    std::string output_file = "lM_act_pen_force.bin";
    int batch_size_lm = 100;    // Number of muscle lengths per batch
    int batch_size_act = 100;  // Number of activations per batch
    int batch_size_pen = 100;  // Number of pennations per batch
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--samples" && i + 1 < argc) {
            sample_size = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--act-samples" && i + 1 < argc) {
            act_samples = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--pen-samples" && i + 1 < argc) {
            pen_samples = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--mu" && i + 1 < argc) {
            mu = std::stod(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--sigma" && i + 1 < argc) {
            sigma = std::stod(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_file = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--batch-lm" && i + 1 < argc) {
            batch_size_lm = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--batch-act" && i + 1 < argc) {
            batch_size_act = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--batch-pen" && i + 1 < argc) {
            batch_size_pen = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --samples N       Number of muscle length samples (default: " << sample_size << ")" << std::endl;
            std::cout << "  --act-samples N   Number of activation samples (default: " << act_samples << ")" << std::endl;
            std::cout << "  --pen-samples N   Number of pennation samples (default: " << pen_samples << ")" << std::endl;
            std::cout << "  --mu X            Mean of normal distribution (default: " << mu << ")" << std::endl;
            std::cout << "  --sigma X         Standard deviation (default: " << sigma << ")" << std::endl;
            std::cout << "  --output FILE     Output file name (default: " << output_file << ")" << std::endl;
            std::cout << "  --batch-lm N      Batch size for muscle length (default: " << batch_size_lm << ")" << std::endl;
            std::cout << "  --batch-act N     Batch size for activation (default: " << batch_size_act << ")" << std::endl;
            std::cout << "  --batch-pen N     Batch size for pennation (default: " << batch_size_pen << ")" << std::endl;
            return 0;
        }
    }
    
    // Adjust batch sizes if needed
    batch_size_lm = std::min(batch_size_lm, sample_size);
    batch_size_act = std::min(batch_size_act, act_samples);
    batch_size_pen = std::min(batch_size_pen, pen_samples);
    
    std::cout << "===============================================" << std::endl;
    std::cout << "Generating muscle force data with batching..." << std::endl;
    std::cout << "Muscle length samples: " << sample_size << std::endl;
    std::cout << "Activation samples: " << act_samples << std::endl;
    std::cout << "Pennation samples: " << pen_samples << std::endl;
    std::cout << "mu: " << mu << ", sigma: " << sigma << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Batch sizes: " << batch_size_lm << " (lM) x " 
              << batch_size_act << " (act) x " << batch_size_pen << " (pen)" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        // Generate all muscle length values at once (typically small enough to fit in memory)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(mu, sigma);
        
        std::vector<double> lm_values(sample_size);
        for (int i = 0; i < sample_size; ++i) {
            lm_values[i] = dist(gen);
            // Clip values to [0.0, 5.0]
            if (lm_values[i] < 0.0) lm_values[i] = 0.0;
            if (lm_values[i] > 5.0) lm_values[i] = 5.0;
        }
        
        std::cout << "Generated " << sample_size << " muscle length samples" << std::endl;
        
        // Create activation and pennation ranges
        std::vector<double> activation_range(act_samples);
        std::vector<double> pennation_range(pen_samples);
        
        for (int i = 0; i < act_samples; ++i) {
            activation_range[i] = i / static_cast<double>(act_samples - 1);
        }
        
        for (int i = 0; i < pen_samples; ++i) {
            pennation_range[i] = (i / static_cast<double>(pen_samples - 1)) * 0.7;
        }
        
        std::cout << "Created activation range with " << act_samples << " samples: [0.0, ..., 1.0]" << std::endl;
        std::cout << "Created pennation range with " << pen_samples << " samples: [0.0, ..., 0.7]" << std::endl;
        
        // Calculate total elements and estimated file size
        uint64_t total_elements = static_cast<uint64_t>(sample_size) * 
                                 static_cast<uint64_t>(act_samples) * 
                                 static_cast<uint64_t>(pen_samples);
        double file_size_gb = (total_elements * 4 * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << "Total grid points: " << total_elements << std::endl;
        std::cout << "Estimated file size: " << std::fixed << std::setprecision(2) << file_size_gb << " GB" << std::endl;
        
        // Initialize output file with header
        if (!initializeOutputFile(output_file, sample_size, act_samples, pen_samples)) {
            std::cerr << "Failed to initialize output file. Aborting." << std::endl;
            return 1;
        }
        
        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // First pass: Write L array in batches
        std::cout << "Writing L array (muscle length)..." << std::endl;
        int lm_batches = (sample_size + batch_size_lm - 1) / batch_size_lm;
        int act_batches = (act_samples + batch_size_act - 1) / batch_size_act;
        int pen_batches = (pen_samples + batch_size_pen - 1) / batch_size_pen;
        
        int total_batches = lm_batches * act_batches * pen_batches * 4; // 4 arrays: L, A, P, F
        int batch_count = 0;
        
        for (int lm_batch = 0; lm_batch < lm_batches; ++lm_batch) {
            int start_lm = lm_batch * batch_size_lm;
            int end_lm = std::min(start_lm + batch_size_lm, sample_size);
            
            for (int act_batch = 0; act_batch < act_batches; ++act_batch) {
                int start_act = act_batch * batch_size_act;
                int end_act = std::min(start_act + batch_size_act, act_samples);
                
                for (int pen_batch = 0; pen_batch < pen_batches; ++pen_batch) {
                    int start_pen = pen_batch * batch_size_pen;
                    int end_pen = std::min(start_pen + batch_size_pen, pen_samples);
                    
                    if (!appendBatchToFile(output_file, lm_values, activation_range, pennation_range,
                                          start_lm, end_lm, start_act, end_act, start_pen, end_pen,
                                          true, false, false, false)) {
                        std::cerr << "Error writing L batch. Aborting." << std::endl;
                        return 1;
                    }
                    
                    batch_count++;
                    if (batch_count % 10 == 0 || batch_count == total_batches) {
                        std::cout << "\rProgress: " << batch_count << "/" << total_batches 
                                 << " batches (" << (batch_count * 100) / total_batches << "%)     " << std::flush;
                    }
                }
            }
        }
        
        // Second pass: Write A array in batches
        std::cout << "\nWriting A array (activation)..." << std::endl;
        for (int lm_batch = 0; lm_batch < lm_batches; ++lm_batch) {
            int start_lm = lm_batch * batch_size_lm;
            int end_lm = std::min(start_lm + batch_size_lm, sample_size);
            
            for (int act_batch = 0; act_batch < act_batches; ++act_batch) {
                int start_act = act_batch * batch_size_act;
                int end_act = std::min(start_act + batch_size_act, act_samples);
                
                for (int pen_batch = 0; pen_batch < pen_batches; ++pen_batch) {
                    int start_pen = pen_batch * batch_size_pen;
                    int end_pen = std::min(start_pen + batch_size_pen, pen_samples);
                    
                    if (!appendBatchToFile(output_file, lm_values, activation_range, pennation_range,
                                          start_lm, end_lm, start_act, end_act, start_pen, end_pen,
                                          false, true, false, false)) {
                        std::cerr << "Error writing A batch. Aborting." << std::endl;
                        return 1;
                    }
                    
                    batch_count++;
                    if (batch_count % 10 == 0 || batch_count == total_batches) {
                        std::cout << "\rProgress: " << batch_count << "/" << total_batches 
                                 << " batches (" << (batch_count * 100) / total_batches << "%)     " << std::flush;
                    }
                }
            }
        }
        
        // Third pass: Write P array in batches
        std::cout << "\nWriting P array (pennation)..." << std::endl;
        for (int lm_batch = 0; lm_batch < lm_batches; ++lm_batch) {
            int start_lm = lm_batch * batch_size_lm;
            int end_lm = std::min(start_lm + batch_size_lm, sample_size);
            
            for (int act_batch = 0; act_batch < act_batches; ++act_batch) {
                int start_act = act_batch * batch_size_act;
                int end_act = std::min(start_act + batch_size_act, act_samples);
                
                for (int pen_batch = 0; pen_batch < pen_batches; ++pen_batch) {
                    int start_pen = pen_batch * batch_size_pen;
                    int end_pen = std::min(start_pen + batch_size_pen, pen_samples);
                    
                    if (!appendBatchToFile(output_file, lm_values, activation_range, pennation_range,
                                          start_lm, end_lm, start_act, end_act, start_pen, end_pen,
                                          false, false, true, false)) {
                        std::cerr << "Error writing P batch. Aborting." << std::endl;
                        return 1;
                    }
                    
                    batch_count++;
                    if (batch_count % 10 == 0 || batch_count == total_batches) {
                        std::cout << "\rProgress: " << batch_count << "/" << total_batches 
                                 << " batches (" << (batch_count * 100) / total_batches << "%)     " << std::flush;
                    }
                }
            }
        }
        
        // Fourth pass: Calculate and write F array in batches
        std::cout << "\nCalculating and writing F array (forces)..." << std::endl;
        for (int lm_batch = 0; lm_batch < lm_batches; ++lm_batch) {
            int start_lm = lm_batch * batch_size_lm;
            int end_lm = std::min(start_lm + batch_size_lm, sample_size);
            
            for (int act_batch = 0; act_batch < act_batches; ++act_batch) {
                int start_act = act_batch * batch_size_act;
                int end_act = std::min(start_act + batch_size_act, act_samples);
                
                for (int pen_batch = 0; pen_batch < pen_batches; ++pen_batch) {
                    int start_pen = pen_batch * batch_size_pen;
                    int end_pen = std::min(start_pen + batch_size_pen, pen_samples);
                    
                    if (!appendBatchToFile(output_file, lm_values, activation_range, pennation_range,
                                          start_lm, end_lm, start_act, end_act, start_pen, end_pen,
                                          false, false, false, true)) {
                        std::cerr << "Error writing F batch. Aborting." << std::endl;
                        return 1;
                    }
                    
                    batch_count++;
                    if (batch_count % 10 == 0 || batch_count == total_batches) {
                        std::cout << "\rProgress: " << batch_count << "/" << total_batches 
                                 << " batches (" << (batch_count * 100) / total_batches << "%)     " << std::flush;
                    }
                }
            }
        }
        
        // Calculate elapsed time
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "\nForce calculations and file writing completed in " << elapsed.count() << " seconds" << std::endl;
        
        // Save lMtilde values to CSV for visualization
        saveToCSV("lMtilde_values.csv", lm_values);
        std::cout << "lMtilde values saved to lMtilde_values.csv" << std::endl;
        
        // Get final file size
        std::ifstream file_check(output_file, std::ios::binary | std::ios::ate);
        if (file_check) {
            std::streamsize size = file_check.tellg();
            file_check.close();
            std::cout << "Actual file size: " << size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        }
        
        std::cout << "Data generation complete!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 1;
    }
}