#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>

using namespace std;

// CUDA kernel for generating password guesses
__global__ void generateGuessesKernel(
    char* ordered_values_flat,
    int* value_offsets,
    int* value_lengths,
    char* base_guess,
    int base_guess_len,
    char* results,
    int max_result_len,
    int num_values,
    bool is_single_segment
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_values) return;
    
    // Calculate result position
    char* result_ptr = results + idx * max_result_len;
    
    if (is_single_segment) {
        // For single segment PT, just copy the value
        int value_offset = value_offsets[idx];
        int value_len = value_lengths[idx];
        
        for (int i = 0; i < value_len && i < max_result_len - 1; i++) {
            result_ptr[i] = ordered_values_flat[value_offset + i];
        }
        result_ptr[value_len] = '\0';
    } else {
        // For multi-segment PT, concatenate base_guess with current value
        int pos = 0;
        
        // Copy base guess
        for (int i = 0; i < base_guess_len && pos < max_result_len - 1; i++) {
            result_ptr[pos++] = base_guess[i];
        }
        
        // Append current value
        int value_offset = value_offsets[idx];
        int value_len = value_lengths[idx];
        
        for (int i = 0; i < value_len && pos < max_result_len - 1; i++) {
            result_ptr[pos++] = ordered_values_flat[value_offset + i];
        }
        result_ptr[pos] = '\0';
    }
}

// CUDA资源清理回调函数
void CUDART_CB cleanupResources(cudaStream_t stream, cudaError_t status, void* userData) {
    void** resources = (void**)userData;
    // 释放内存
    for (int i = 0; i < 5; i++) {
        if (resources[i]) {
            cudaFree(resources[i]);
        }
    }
    // 释放资源数组
    delete[] resources;
}

// 添加缺失的 CalProb 函数实现
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

// 添加缺失的 init 函数实现
void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

// 添加缺失的 PT::NewPTs 函数实现
vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                PT new_pt = *this;
                new_pt.pivot = i;
                res.emplace_back(new_pt);
            }
            curr_indices[i] = 0;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 异步GPU计算 - 单segment
void PriorityQueue::GenerateGPUAsync_SingleSegment(PT pt, cudaStream_t stream, vector<char>& h_results) {
    const int MAX_GUESS_LENGTH = 128;
    
    // 处理只有一个segment的情况
    segment *a = nullptr;
    if (pt.content[0].type == 1) {
        a = &m.letters[m.FindLetter(pt.content[0])];
    } else if (pt.content[0].type == 2) {
        a = &m.digits[m.FindDigit(pt.content[0])];
    } else if (pt.content[0].type == 3) {
        a = &m.symbols[m.FindSymbol(pt.content[0])];
    }
    
    if (a == nullptr) return;
    
    int num_values = pt.max_indices[0];
    if (num_values == 0) return;
    
    // 准备数据传输到GPU
    vector<char> ordered_values_flat;
    vector<int> value_offsets;
    vector<int> value_lengths;
    
    int offset = 0;
    for (int i = 0; i < num_values; i++) {
        const string& value = a->ordered_values[i];
        value_offsets.push_back(offset);
        value_lengths.push_back(static_cast<int>(value.length()));
        
        for (char c : value) {
            ordered_values_flat.push_back(c);
        }
        offset += static_cast<int>(value.length());
    }
    
    // GPU内存分配
    char* d_ordered_values = nullptr;
    int* d_value_offsets = nullptr;
    int* d_value_lengths = nullptr;
    char* d_results = nullptr;
    
    size_t flat_size = ordered_values_flat.size();
    size_t offsets_size = num_values * sizeof(int);
    size_t lengths_size = num_values * sizeof(int);
    size_t results_size = static_cast<size_t>(num_values) * MAX_GUESS_LENGTH;
    
    cudaError_t err;
    err = cudaMalloc(&d_ordered_values, flat_size);
    if (err != cudaSuccess) return;
    
    err = cudaMalloc(&d_value_offsets, offsets_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        return;
    }
    
    err = cudaMalloc(&d_value_lengths, lengths_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        return;
    }
    
    err = cudaMalloc(&d_results, results_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        return;
    }
    
    // 异步数据传输到GPU
    cudaMemcpyAsync(d_ordered_values, ordered_values_flat.data(), flat_size, 
                  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_value_offsets, value_offsets.data(), offsets_size, 
                  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_value_lengths, value_lengths.data(), lengths_size, 
                  cudaMemcpyHostToDevice, stream);
    
    // 配置kernel启动参数
    int blockSize = 256;
    int numBlocks = (num_values + blockSize - 1) / blockSize;
    
    // 异步启动kernel
    generateGuessesKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_ordered_values,
        d_value_offsets,
        d_value_lengths,
        nullptr,
        0,
        d_results,
        MAX_GUESS_LENGTH,
        num_values,
        true
    );
    
    // 异步将结果传输回CPU
    h_results.resize(results_size);
    cudaMemcpyAsync(h_results.data(), d_results, results_size, 
                  cudaMemcpyDeviceToHost, stream);
    
    // 创建资源数组用于回调函数中释放内存
    void** resources = new void*[5]{d_ordered_values, d_value_offsets, d_value_lengths, d_results, nullptr};
    
    // 注册回调函数来清理GPU内存
    cudaStreamAddCallback(stream, cleanupResources, resources, 0);
}

// 异步GPU计算 - 多segment
void PriorityQueue::GenerateGPUAsync_MultiSegment(PT pt, cudaStream_t stream, vector<char>& h_results) {
    const int MAX_GUESS_LENGTH = 128;
    
    // 处理多个segment的情况
    string guess;
    int seg_idx = 0;
    
    // 构建基础guess字符串（除最后一个segment外）
    for (int idx : pt.curr_indices) {
        if (pt.content[seg_idx].type == 1) {
            guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 2) {
            guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 3) {
            guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        }
        seg_idx += 1;
        if (seg_idx == static_cast<int>(pt.content.size()) - 1) {
            break;
        }
    }
    
    // 获取最后一个segment的指针
    segment *a = nullptr;
    int last_idx = static_cast<int>(pt.content.size()) - 1;
    if (pt.content[last_idx].type == 1) {
        a = &m.letters[m.FindLetter(pt.content[last_idx])];
    } else if (pt.content[last_idx].type == 2) {
        a = &m.digits[m.FindDigit(pt.content[last_idx])];
    } else if (pt.content[last_idx].type == 3) {
        a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
    }
    
    if (a == nullptr) return;
    
    int num_values = pt.max_indices[last_idx];
    if (num_values == 0) return;
    
    // 准备数据传输到GPU
    vector<char> ordered_values_flat;
    vector<int> value_offsets;
    vector<int> value_lengths;
    
    int offset = 0;
    for (int i = 0; i < num_values; i++) {
        const string& value = a->ordered_values[i];
        value_offsets.push_back(offset);
        value_lengths.push_back(static_cast<int>(value.length()));
        
        for (char c : value) {
            ordered_values_flat.push_back(c);
        }
        offset += static_cast<int>(value.length());
    }
    
    // GPU内存分配
    char* d_ordered_values = nullptr;
    int* d_value_offsets = nullptr;
    int* d_value_lengths = nullptr;
    char* d_base_guess = nullptr;
    char* d_results = nullptr;
    
    size_t flat_size = ordered_values_flat.size();
    size_t offsets_size = num_values * sizeof(int);
    size_t lengths_size = num_values * sizeof(int);
    size_t base_size = guess.length() + 1;
    size_t results_size = static_cast<size_t>(num_values) * MAX_GUESS_LENGTH;
    
    cudaError_t err;
    err = cudaMalloc(&d_ordered_values, flat_size);
    if (err != cudaSuccess) return;
    
    err = cudaMalloc(&d_value_offsets, offsets_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        return;
    }
    
    err = cudaMalloc(&d_value_lengths, lengths_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        return;
    }
    
    err = cudaMalloc(&d_base_guess, base_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        return;
    }
    
    err = cudaMalloc(&d_results, results_size);
    if (err != cudaSuccess) {
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        cudaFree(d_base_guess);
        return;
    }
    
    // 异步数据传输到GPU
    cudaMemcpyAsync(d_ordered_values, ordered_values_flat.data(), flat_size, 
                  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_value_offsets, value_offsets.data(), offsets_size, 
                  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_value_lengths, value_lengths.data(), lengths_size, 
                  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_base_guess, guess.c_str(), base_size, 
                  cudaMemcpyHostToDevice, stream);
    
    // 配置kernel启动参数
    int blockSize = 256;
    int numBlocks = (num_values + blockSize - 1) / blockSize;
    
    // 异步启动kernel
    generateGuessesKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_ordered_values,
        d_value_offsets,
        d_value_lengths,
        d_base_guess,
        static_cast<int>(guess.length()),
        d_results,
        MAX_GUESS_LENGTH,
        num_values,
        false
    );
    
    // 异步将结果传输回CPU
    h_results.resize(results_size);
    cudaMemcpyAsync(h_results.data(), d_results, results_size, 
                  cudaMemcpyDeviceToHost, stream);
    
    // 创建资源数组用于回调函数中释放内存
    void** resources = new void*[5]{d_ordered_values, d_value_offsets, d_value_lengths, d_base_guess, d_results};
    
    // 注册回调函数来清理GPU内存
    cudaStreamAddCallback(stream, cleanupResources, resources, 0);
}

// 异步启动GPU计算
void PriorityQueue::GenerateGPUAsync(PT pt, cudaStream_t stream, vector<char>& h_results) {
    // 计算PT的概率
    CalProb(pt);
    
    if (pt.content.size() == 1) {
        GenerateGPUAsync_SingleSegment(pt, stream, h_results);
    } else {
        GenerateGPUAsync_MultiSegment(pt, stream, h_results);
    }
}

// 异步处理单个PT
void PriorityQueue::PopNext() {
    if (priority.empty()) return;
    
    PT current_pt = priority.front();
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 为结果分配内存
    vector<char> h_results;
    
    // 异步启动GPU计算
    GenerateGPUAsync(current_pt, stream, h_results);
    
    // 在GPU计算的同时，CPU处理新的PT
    vector<PT> new_pts = current_pt.NewPTs();
    for (PT& pt : new_pts) {
        CalProb(pt);
        // 将新PT插入优先队列
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    
    // CPU任务完成后，等待GPU计算完成
    cudaStreamSynchronize(stream);
    
    // 处理GPU计算结果
    const int MAX_GUESS_LENGTH = 128;
    int num_values;
    
    if (current_pt.content.size() == 1) {
        num_values = current_pt.max_indices[0];
    } else {
        num_values = current_pt.max_indices[current_pt.content.size() - 1];
    }
    
    for (int i = 0; i < num_values; i++) {
        char* result_start = h_results.data() + i * MAX_GUESS_LENGTH;
        string guess(result_start);
        if (!guess.empty()) {
            guesses.push_back(guess);
            total_guesses += 1;
        }
    }
    
    // 销毁CUDA流
    cudaStreamDestroy(stream);
    
    // 从优先队列中移除已处理的PT
    priority.erase(priority.begin());
}

// 批量处理多个PT
void PriorityQueue::PopNextBatch(int batch_size) {
    if (priority.empty()) return;
    
    // 限制batch_size不超过队列大小
    batch_size = min(batch_size, static_cast<int>(priority.size()));
    
    // 为每个PT创建一个流和结果容器
    vector<cudaStream_t> streams(batch_size);
    vector<vector<char>> results(batch_size);
    vector<PT> batch_pts;
    
    // 创建CUDA流并启动异步计算
    for (int i = 0; i < batch_size; i++) {
        batch_pts.push_back(priority[i]);
        cudaStreamCreate(&streams[i]);
        GenerateGPUAsync(batch_pts[i], streams[i], results[i]);
    }
    
    // CPU处理所有PT生成的新PT
    vector<vector<PT>> all_new_pts(batch_size);
    for (int i = 0; i < batch_size; i++) {
        all_new_pts[i] = batch_pts[i].NewPTs();
    }
    
    // 将所有新PT插入优先队列
    for (auto& new_pts : all_new_pts) {
        for (PT& pt : new_pts) {
            CalProb(pt);
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (iter != priority.end() - 1 && iter != priority.begin()) {
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1) {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob) {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }
    }
    
    // 等待所有GPU计算完成并处理结果
    const int MAX_GUESS_LENGTH = 128;
    for (int i = 0; i < batch_size; i++) {
        cudaStreamSynchronize(streams[i]);
        
        // 处理GPU计算结果
        int num_values;
        if (batch_pts[i].content.size() == 1) {
            num_values = batch_pts[i].max_indices[0];
        } else {
            num_values = batch_pts[i].max_indices[batch_pts[i].content.size() - 1];
        }
        
        for (int j = 0; j < num_values; j++) {
            char* result_start = results[i].data() + j * MAX_GUESS_LENGTH;
            string guess(result_start);
            if (!guess.empty()) {
                guesses.push_back(guess);
                total_guesses += 1;
            }
        }
        
        // 销毁CUDA流
        cudaStreamDestroy(streams[i]);
    }
    
    // 从优先队列中移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + batch_size);
}

void PriorityQueue::GenerateGPU(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    const int MAX_GUESS_LENGTH = 128; // 假设最大密码长度为128

    if (pt.content.size() == 1) {
        // 处理只有一个segment的情况
        segment *a = nullptr;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        if (a == nullptr) return;
        
        int num_values = pt.max_indices[0];
        if (num_values == 0) return;
        
        // 准备数据传输到GPU
        vector<char> ordered_values_flat;
        vector<int> value_offsets;
        vector<int> value_lengths;
        
        int offset = 0;
        for (int i = 0; i < num_values; i++) {
            const string& value = a->ordered_values[i];
            value_offsets.push_back(offset);
            value_lengths.push_back(static_cast<int>(value.length()));
            
            for (char c : value) {
                ordered_values_flat.push_back(c);
            }
            offset += static_cast<int>(value.length());
        }
        
        // GPU内存分配
        char* d_ordered_values = nullptr;
        int* d_value_offsets = nullptr;
        int* d_value_lengths = nullptr;
        char* d_results = nullptr;
        
        size_t flat_size = ordered_values_flat.size();
        size_t offsets_size = num_values * sizeof(int);
        size_t lengths_size = num_values * sizeof(int);
        size_t results_size = static_cast<size_t>(num_values) * MAX_GUESS_LENGTH;
        
        cudaError_t err;
        err = cudaMalloc(&d_ordered_values, flat_size);
        if (err != cudaSuccess) return;
        
        err = cudaMalloc(&d_value_offsets, offsets_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            return;
        }
        
        err = cudaMalloc(&d_value_lengths, lengths_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            cudaFree(d_value_offsets);
            return;
        }
        
        err = cudaMalloc(&d_results, results_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            cudaFree(d_value_offsets);
            cudaFree(d_value_lengths);
            return;
        }
        
        // 数据传输到GPU
        cudaMemcpy(d_ordered_values, ordered_values_flat.data(), flat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_offsets, value_offsets.data(), offsets_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_lengths, value_lengths.data(), lengths_size, cudaMemcpyHostToDevice);
        
        // 配置kernel启动参数
        int blockSize = 256;
        int numBlocks = (num_values + blockSize - 1) / blockSize;
        
        // 启动kernel
        generateGuessesKernel<<<numBlocks, blockSize>>>(
            d_ordered_values,
            d_value_offsets,
            d_value_lengths,
            nullptr,
            0,
            d_results,
            MAX_GUESS_LENGTH,
            num_values,
            true
        );
        
        // 等待GPU计算完成
        cudaDeviceSynchronize();
        
        // 将结果传输回CPU
        vector<char> h_results(results_size);
        cudaMemcpy(h_results.data(), d_results, results_size, cudaMemcpyDeviceToHost);
        
        // 将结果转换为string并添加到guesses
        for (int i = 0; i < num_values; i++) {
            char* result_start = h_results.data() + i * MAX_GUESS_LENGTH;
            string guess(result_start);
            guesses.push_back(guess);
            total_guesses += 1;
        }
        
        // 释放GPU内存
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        cudaFree(d_results);
        
    } else {
        // 处理多个segment的情况
        string guess;
        int seg_idx = 0;
        
        // 构建基础guess字符串（除最后一个segment外）
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == static_cast<int>(pt.content.size()) - 1) {
                break;
            }
        }
        
        // 获取最后一个segment的指针
        segment *a = nullptr;
        int last_idx = static_cast<int>(pt.content.size()) - 1;
        if (pt.content[last_idx].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        } else if (pt.content[last_idx].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        } else if (pt.content[last_idx].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }
        
        if (a == nullptr) return;
        
        int num_values = pt.max_indices[last_idx];
        if (num_values == 0) return;
        
        // 准备数据传输到GPU
        vector<char> ordered_values_flat;
        vector<int> value_offsets;
        vector<int> value_lengths;
        
        int offset = 0;
        for (int i = 0; i < num_values; i++) {
            const string& value = a->ordered_values[i];
            value_offsets.push_back(offset);
            value_lengths.push_back(static_cast<int>(value.length()));
            
            for (char c : value) {
                ordered_values_flat.push_back(c);
            }
            offset += static_cast<int>(value.length());
        }
        
        // GPU内存分配
        char* d_ordered_values = nullptr;
        int* d_value_offsets = nullptr;
        int* d_value_lengths = nullptr;
        char* d_base_guess = nullptr;
        char* d_results = nullptr;
        
        size_t flat_size = ordered_values_flat.size();
        size_t offsets_size = num_values * sizeof(int);
        size_t lengths_size = num_values * sizeof(int);
        size_t base_size = guess.length() + 1;
        size_t results_size = static_cast<size_t>(num_values) * MAX_GUESS_LENGTH;
        
        cudaError_t err;
        err = cudaMalloc(&d_ordered_values, flat_size);
        if (err != cudaSuccess) return;
        
        err = cudaMalloc(&d_value_offsets, offsets_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            return;
        }
        
        err = cudaMalloc(&d_value_lengths, lengths_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            cudaFree(d_value_offsets);
            return;
        }
        
        err = cudaMalloc(&d_base_guess, base_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            cudaFree(d_value_offsets);
            cudaFree(d_value_lengths);
            return;
        }
        
        err = cudaMalloc(&d_results, results_size);
        if (err != cudaSuccess) {
            cudaFree(d_ordered_values);
            cudaFree(d_value_offsets);
            cudaFree(d_value_lengths);
            cudaFree(d_base_guess);
            return;
        }
        
        // 数据传输到GPU
        cudaMemcpy(d_ordered_values, ordered_values_flat.data(), flat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_offsets, value_offsets.data(), offsets_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_lengths, value_lengths.data(), lengths_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_base_guess, guess.c_str(), base_size, cudaMemcpyHostToDevice);
        
        // 配置kernel启动参数
        int blockSize = 256;
        int numBlocks = (num_values + blockSize - 1) / blockSize;
        
        // 启动kernel
        generateGuessesKernel<<<numBlocks, blockSize>>>(
            d_ordered_values,
            d_value_offsets,
            d_value_lengths,
            d_base_guess,
            static_cast<int>(guess.length()),
            d_results,
            MAX_GUESS_LENGTH,
            num_values,
            false
        );
        
        // 等待GPU计算完成
        cudaDeviceSynchronize();
        
        // 将结果传输回CPU
        vector<char> h_results(results_size);
        cudaMemcpy(h_results.data(), d_results, results_size, cudaMemcpyDeviceToHost);
        
        // 将结果转换为string并添加到guesses
        for (int i = 0; i < num_values; i++) {
            char* result_start = h_results.data() + i * MAX_GUESS_LENGTH;
            string temp(result_start);
            guesses.push_back(temp);
            total_guesses += 1;
        }
        
        // 释放GPU内存
        cudaFree(d_ordered_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        cudaFree(d_base_guess);
        cudaFree(d_results);
    }
}