#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <utility> 

using namespace std;

void PriorityQueue::ProcessPTAdaptively(PT pt) {
    int workload = EstimateWorkload(pt);
    
    // 设定阈值，可以通过性能测试动态调整
    const int GPU_THRESHOLD = 5000;  // 示例阈值
    
    if (workload < GPU_THRESHOLD) {
        // 小工作量用CPU处理（避免GPU数据传输开销）
        Generate(pt);
    } else {
        // 大工作量用GPU处理
        GenerateGPU(pt);
    }
}

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

void PriorityQueue::PopNextBatch(int batch_size) {
    if (priority.empty()) return;
    
    // 限制batch_size不超过队列大小
    batch_size = min(batch_size, static_cast<int>(priority.size()));
    
    // 评估每个PT的工作量
    vector<pair<int, int>> workloads; // <工作量, 索引>
    for (int i = 0; i < batch_size; i++) {
        workloads.push_back(make_pair(EstimateWorkload(priority[i]), i));
    }
    
    // 根据工作量对PT进行排序
    sort(workloads.begin(), workloads.end());
    
    // 负载均衡：将工作量分配到多个CUDA流
    const int NUM_STREAMS = 4;  // 流数量根据GPU性能调整
    vector<cudaStream_t> streams(NUM_STREAMS);
    vector<vector<int>> stream_assignments(NUM_STREAMS);
    vector<int> stream_loads(NUM_STREAMS, 0);
    
    // 创建CUDA流
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 使用"贪心"策略分配工作
    for (int i = 0; i < workloads.size(); i++) {
        int load = workloads[i].first;
        int idx = workloads[i].second;
        
        // 找到当前负载最小的流
        int min_load_stream = 0;
        for (int j = 1; j < NUM_STREAMS; j++) {
            if (stream_loads[j] < stream_loads[min_load_stream]) {
                min_load_stream = j;
            }
        }
        
        // 分配工作
        stream_assignments[min_load_stream].push_back(idx);
        stream_loads[min_load_stream] += load;
    }
    
    // 处理各流的PT
    vector<vector<char>> all_results(batch_size);
    for (int i = 0; i < NUM_STREAMS; i++) {
        for (int j = 0; j < stream_assignments[i].size(); j++) {
            int idx = stream_assignments[i][j];
            GenerateGPUAsync(priority[idx], streams[i], all_results[idx]);
        }
    }
    
    // 同步并处理结果
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 处理生成的口令
    const int MAX_GUESS_LENGTH = 128;
    for (int i = 0; i < batch_size; i++) {
        int num_values;
        if (priority[i].content.size() == 1) {
            num_values = priority[i].max_indices[0];
        } else {
            num_values = priority[i].max_indices[priority[i].content.size() - 1];
        }
        
        for (int j = 0; j < num_values; j++) {
            char* result_start = all_results[i].data() + j * MAX_GUESS_LENGTH;
            string guess(result_start);
            if (!guess.empty()) {
                guesses.push_back(guess);
                total_guesses += 1;
            }
        }
    }
    
    // 销毁CUDA流
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    // 从优先队列中移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + batch_size);
}

void PriorityQueue::HybridPopNextBatch(int batch_size) {
    if (priority.empty()) return;
    
    // 限制batch_size
    batch_size = min(batch_size, static_cast<int>(priority.size()));
    
    vector<PT> gpu_pts;
    vector<PT> cpu_pts;
    vector<int> gpu_indices;
    vector<int> cpu_indices;
    
    // 阈值可通过性能测试调整
    const int GPU_THRESHOLD = 5000;
    
    // 分类PT到CPU或GPU队列
    for (int i = 0; i < batch_size; i++) {
        int workload = EstimateWorkload(priority[i]);
        if (workload >= GPU_THRESHOLD) {
            gpu_pts.push_back(priority[i]);
            gpu_indices.push_back(i);
        } else {
            cpu_pts.push_back(priority[i]);
            cpu_indices.push_back(i);
        }
    }
    
    // 创建CUDA流和结果容器
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    vector<vector<char>> gpu_results(gpu_pts.size());
    
    // 启动GPU异步计算
    for (size_t i = 0; i < gpu_pts.size(); i++) {
        GenerateGPUAsync(gpu_pts[i], stream, gpu_results[i]);
    }
    
    // 同时在CPU上处理小型PT
    #pragma omp parallel for
    for (int i = 0; i < cpu_pts.size(); i++) {
        Generate(cpu_pts[i]);
    }
    
    // 等待GPU完成
    cudaStreamSynchronize(stream);
    
    // 处理GPU结果
    const int MAX_GUESS_LENGTH = 128;
    for (size_t i = 0; i < gpu_pts.size(); i++) {
        int num_values;
        if (gpu_pts[i].content.size() == 1) {
            num_values = gpu_pts[i].max_indices[0];
        } else {
            num_values = gpu_pts[i].max_indices[gpu_pts[i].content.size() - 1];
        }
        
        for (int j = 0; j < num_values; j++) {
            char* result_start = gpu_results[i].data() + j * MAX_GUESS_LENGTH;
            string guess(result_start);
            if (!guess.empty()) {
                guesses.push_back(guess);
                total_guesses += 1;
            }
        }
    }
    
    // 清理
    cudaStreamDestroy(stream);
    
    // 从优先队列中移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + batch_size);
}

// 动态调整GPU阈值的方法
int PriorityQueue::AutoTuneThresholds() {
    static int gpu_threshold = 5000;  // 初始阈值
    static int samples = 0;
    static double cpu_time_total = 0;
    static double gpu_time_total = 0;
    static int cpu_guesses_total = 0;
    static int gpu_guesses_total = 0;
    
    // 定期采样性能并调整阈值
    if (++samples % 10 == 0) {
        double cpu_time_per_guess = cpu_guesses_total > 0 ? 
            cpu_time_total / cpu_guesses_total : 0;
        double gpu_time_per_guess = gpu_guesses_total > 0 ? 
            gpu_time_total / gpu_guesses_total : 0;
        
        // 考虑GPU数据传输开销
        double gpu_overhead = 0.0005;  // 估计值，单位：秒
        
        if (cpu_time_per_guess > 0 && gpu_time_per_guess > 0) {
            // 计算平衡点：何时GPU更有优势
            int new_threshold = static_cast<int>(gpu_overhead / 
                (cpu_time_per_guess - gpu_time_per_guess));
                
            // 避免极端值
            if (new_threshold > 100 && new_threshold < 100000) {
                gpu_threshold = new_threshold;
            }
        }
        
        // 重置统计
        cpu_time_total = gpu_time_total = 0;
        cpu_guesses_total = gpu_guesses_total = 0;
    }
    
    return gpu_threshold;
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

void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率
    CalProb(pt);

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
        
        // CPU版本：直接遍历生成所有可能的口令
        for (int i = 0; i < num_values; i++) {
            string guess = a->ordered_values[i];
            guesses.push_back(guess);
            total_guesses += 1;
        }
        
    } else {
        // 处理多个segment的情况
        string base_guess;
        int seg_idx = 0;
        
        // 构建基础guess字符串（除最后一个segment外）
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
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
        
        // CPU版本：遍历最后一个segment的所有值
        for (int i = 0; i < num_values; i++) {
            string complete_guess = base_guess + a->ordered_values[i];
            guesses.push_back(complete_guess);
            total_guesses += 1;
        }
    }
}