// filepath: [correctness_guess.cpp](http://_vscodecontentref_/0)
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// mpicxx correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// mpirun -np 4 ./main

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
    
    // 广播训练好的模型到所有进程
    if (rank != 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 加载一些测试数据
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
    int cracked=0;
    int total_cracked=0;

    q.init();
    if (rank == 0) {
        cout << "here" << endl;
    }
    
    int global_curr_num = 0;  
    auto start = system_clock::now();
    int history = 0;
    bool should_continue = true;
    
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
        
        // 只有有工作的进程才执行PopNext
        if (!q.priority.empty()) {
            q.PopNext();
        }
        
        q.total_guesses = q.guesses.size();
        
        // 所有进程都参与收集猜测数量
        int global_guesses;
        MPI_Allreduce(&q.total_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // 更新全局当前数量
        global_curr_num += global_guesses;
        
        // 检查是否应该退出（所有进程都检查）
        int should_exit = 0;
        if (rank == 0 && global_curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + global_curr_num << endl;

            if (history + global_curr_num > 10000000)
            {
                should_exit = 1;
            }
        }
        
        // 广播退出条件给所有进程
        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (should_exit) {
            // 所有进程都需要参与最终的统计收集
            if (rank == 0) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            }
            
            // 收集所有进程的破解数量
            MPI_Reduce(&cracked, &total_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time:" << time_hash << " seconds" << endl;
                cout << "Train time:" << time_train << " seconds" << endl;
                cout << "Cracked:" << total_cracked << endl;
            }
            
            should_continue = false;
            break;
        }
        
        if (global_curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    cracked+=1;
                }
                MD5Hash(pw, state);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += global_curr_num;
            global_curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 最终同步所有进程的结果（防止重复收集）
    if (!should_continue) {
        // 已经在上面收集过了，这里不需要再次收集
    } else {
        MPI_Reduce(&cracked, &total_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            cout << "Final cracked count: " << total_cracked << endl;
        }
    }
    
    // 结束MPI
    MPI_Finalize();
    return 0;
}