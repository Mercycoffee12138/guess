#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
using namespace std;
using namespace chrono;

// 编译指令如下
// mpicxx correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -pthread
// mpirun -np 4 ./main

// 全局变量用于线程间通信
mutex guesses_mutex;
vector<string> pending_hash_guesses;
atomic<int> total_cracked(0);
atomic<int> total_hashed(0);  // 统计实际哈希处理的密码数量
atomic<bool> hash_thread_should_exit(false);

// 哈希计算线程函数
void hash_worker_thread(const unordered_set<string>& test_set, double& time_hash) {
    auto start_hash = system_clock::now();
    
    while (!hash_thread_should_exit) {
        vector<string> local_guesses;
        
        // 从待处理队列中取出猜测进行哈希
        {
            lock_guard<mutex> lock(guesses_mutex);
            if (!pending_hash_guesses.empty()) {
                local_guesses = move(pending_hash_guesses);
                pending_hash_guesses.clear();
            }
        }
        
        // 进行MD5哈希计算
        if (!local_guesses.empty()) {
            bit32 state[4];
            for (const string& pw : local_guesses) {
                if (test_set.find(pw) != test_set.end()) {
                    total_cracked++;
                }
                MD5Hash(pw, state);
                total_hashed++;  // 统计实际处理的密码数量
            }
        } else {
            // 没有工作时短暂休眠
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
    
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

int main(int argc, char** argv)
{
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    
    // 设置MPI参数
    q.mpi_rank = rank;
    q.mpi_size = size;
    
    auto start_train = system_clock::now();
    
    // 只在主进程中进行训练
    if (rank == 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    
    // 简化：所有进程都进行训练（实际应该广播模型数据）
    if (rank != 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 加载测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    
    q.init();
    if (rank == 0) {
        cout << "Starting PT-level parallel processing with overlapped computation..." << endl;
        cout << "Termination condition: 10,000,000 passwords HASHED (not just generated)" << endl;
    }
    
    // 启动哈希计算线程
    thread hash_thread(hash_worker_thread, ref(test_set), ref(time_hash));
    
    int global_curr_num = 0;  
    auto start = system_clock::now();
    int history = 0;
    bool should_continue = true;
    
    // 定义批处理大小（一次处理的PT数量）
    const int BATCH_SIZE = size;  // 与进程数相同
    
    // 新增：跟踪哈希数量，避免重复计数
    int previous_global_hashed = 0;  // 记录上一次的全局哈希数量
    int true_global_hashed = 0;      // 记录真实的累计哈希数量
    
    while (should_continue)
    {
        // 检查是否还有工作要做
        int local_has_work = q.priority.empty() ? 0 : 1;
        int global_has_work;
        MPI_Allreduce(&local_has_work, &global_has_work, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if (global_has_work == 0) {
            should_continue = false;
            break;
        }
        
        // 使用PT层面的并行处理
        if (!q.priority.empty()) {
            q.PopNextBatch(BATCH_SIZE);
        }
        
        q.total_guesses = q.guesses.size();
        
        // 将新生成的猜测添加到哈希队列（重叠计算的关键）
        if (!q.guesses.empty()) {
            lock_guard<mutex> lock(guesses_mutex);
            pending_hash_guesses.insert(pending_hash_guesses.end(), 
                                      q.guesses.begin(), q.guesses.end());
        }
        
        // 收集所有进程的猜测数量（用于显示生成进度）
        int global_guesses;
        MPI_Allreduce(&q.total_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        global_curr_num += global_guesses;
        
        // 收集所有进程的哈希处理数量（用于终止条件）
        int local_hashed_count = total_hashed.load();
        int current_global_hashed;
        MPI_Allreduce(&local_hashed_count, &current_global_hashed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // 计算新增的哈希数量(避免重复计数)
        int new_hashed = current_global_hashed - previous_global_hashed;
        true_global_hashed += new_hashed;
        previous_global_hashed = current_global_hashed;
        
        // 检查是否应该退出（基于正确的哈希处理数量）
        int should_exit = 0;
        if (rank == 0) {
            // 定期显示进度
            if (true_global_hashed >= 100000 && true_global_hashed % 500000 < 100000) {
                cout << "Generated: " << history + global_curr_num << " passwords, ";
                cout << "Hashed: " << true_global_hashed << " passwords" << endl;
            }
            
            // 终止条件：哈希处理达到1000万
            if (true_global_hashed >= 10000000) {
                cout << "Reached 10,000,000 hashed passwords. Terminating..." << endl;
                should_exit = 1;
            }
        }
        
        // 广播退出条件
        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (should_exit) {
            // 等待剩余密码处理完成（可选，确保更准确的结果）
            if (rank == 0) {
                cout << "Waiting for remaining passwords to be processed..." << endl;
            }
            
            // 给哈希线程一些时间处理剩余密码
            this_thread::sleep_for(chrono::milliseconds(500));
            
            // 停止哈希线程
            hash_thread_should_exit = true;
            hash_thread.join();
            
            // 更新最后一次的哈希统计
            int local_hashed_final = total_hashed.load();
            int current_global_hashed_final;
            MPI_Allreduce(&local_hashed_final, &current_global_hashed_final, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            // 加上最后处理的部分
            int final_new_hashed = current_global_hashed_final - previous_global_hashed;
            int final_global_hashed = true_global_hashed + final_new_hashed;
            
            // 收集破解数量
            int local_cracked = total_cracked.load();
            int total_cracked_final = 0;
            MPI_Reduce(&local_cracked, &total_cracked_final, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                cout << "=== Final Results ===" << endl;
                cout << "Total passwords generated: " << history + global_curr_num << endl;
                cout << "Total passwords hashed: " << final_global_hashed << endl;
                cout << "Total passwords cracked: " << total_cracked_final << endl;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                cout << "Crack rate: " << (double)total_cracked_final / final_global_hashed * 100 << "%" << endl;
            }
            
            should_continue = false;
            break;
        }
        
        // 定期清理已生成的猜测（因为已经转交给哈希线程处理）
        if (global_curr_num > 1000000)
        {
            history += global_curr_num;
            global_curr_num = 0;
            q.guesses.clear();  // 清理主线程的猜测缓存
        }
    }
    
    // 确保哈希线程正常结束
    if (!hash_thread_should_exit) {
        hash_thread_should_exit = true;
        hash_thread.join();
    }
    
    // 最终收集所有进程的结果（如果没有通过退出条件收集）
    if (should_continue) {
        // 更新最后一次的哈希统计
        int local_hashed_final = total_hashed.load();
        int current_global_hashed_final;
        MPI_Allreduce(&local_hashed_final, &current_global_hashed_final, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // 加上最后处理的部分
        int final_new_hashed = current_global_hashed_final - previous_global_hashed;
        int final_global_hashed = true_global_hashed + final_new_hashed;
        
        int local_cracked = total_cracked.load();
        int total_cracked_final = 0;
        MPI_Reduce(&local_cracked, &total_cracked_final, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "=== Final Results (alternate exit) ===" << endl;
            cout << "Total passwords generated: " << history + global_curr_num << endl;
            cout << "Total passwords hashed: " << final_global_hashed << endl;
            cout << "Total passwords cracked: " << total_cracked_final << endl;
        }
    }
    
    // 结束MPI
    MPI_Finalize();
    return 0;
}