#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <windows.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ -mavx2 -O0 main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ -mavx2 -O1 main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ -mavx2 -O2 main.cpp train.cpp guessing.cpp md5.cpp -o main

int main()
{
    SetConsoleOutputCP(CP_UTF8);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    long total_passwords = 0;
    double total_serial_time = 0.0;
    double total_simd2_time = 0.0;
    double total_simd4_time = 0.0;
    double total_simd8_time = 0.0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./output/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            /*auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // cout<<pw<<"\t";
                 //for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     cout << std::setw(8) << std::setfill('0') << hex << state[i1];
                 //}
                // cout << endl;
            }*/
            /*auto start_hash = system_clock::now();
            bit32 state1[4], state2[4];
            vector<string>& guesses = q.guesses;
            // 使用SIMD版本同时处理两个字符串
            for (size_t i = 0; i < guesses.size(); i += 2) {
                if (i + 1 < guesses.size()) {
                   // 有两个字符串可以处理时，使用SIMD版本
                   MD5Hash_SIMD(guesses[i], guesses[i+1], state1, state2);
               } else {
                   // 最后一个落单的字符串，使用普通版本
                MD5Hash(guesses[i], state1);
              }
            }*/
            /*auto start_hash = system_clock::now();
            vector<string>& guesses = q.guesses;

            // 准备处理四路SIMD哈希所需的变量
            bit32 state0[4], state1[4], state2[4], state3[4];
            bit32* states[4] = {state0, state1, state2, state3};
            string batch[4];

            // 每次处理4个密码
            for (size_t i = 0; i < guesses.size(); i += 4) {
               if (i + 3 < guesses.size()) {
                   // 有4个密码可以处理，使用四路SIMD版本
                  batch[0] = guesses[i];
                  batch[1] = guesses[i+1];
                  batch[2] = guesses[i+2];
                  batch[3] = guesses[i+3];
                  MD5Hash_SIMD4(batch, states);
              } else {
                   // 处理剩余不足4个的密码
                   for (size_t j = i; j < guesses.size(); j++) {
                       MD5Hash(guesses[j], state0);
                    }
                }
            }*/
           /* auto start_hash = system_clock::now();
            vector<string>& guesses = q.guesses;
            
            // 准备处理八路SIMD哈希所需的变量
            bit32 state0[4], state1[4], state2[4], state3[4];
            bit32 state4[4], state5[4], state6[4], state7[4];
            bit32* states[8] = {
                state0, state1, state2, state3,
                state4, state5, state6, state7
            };
            string batch[8];
            
            // 每次处理8个密码
            for (size_t i = 0; i < guesses.size(); i += 8) {
                if (i + 7 < guesses.size()) {
                    // 有8个密码可以处理，使用八路SIMD版本
                    batch[0] = guesses[i];
                    batch[1] = guesses[i+1];
                    batch[2] = guesses[i+2];
                    batch[3] = guesses[i+3];
                    batch[4] = guesses[i+4];
                    batch[5] = guesses[i+5];
                    batch[6] = guesses[i+6];
                    batch[7] = guesses[i+7];
                    MD5Hash_SIMD8(batch, states);
                } else {
                    // 处理剩余不足8个的密码
                    for (size_t j = i; j < guesses.size(); j++) {
                        MD5Hash(guesses[j], state0);
                    }
                }
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;*/
            vector<string>& guesses = q.guesses;
            double time_serial = 0.0, time_simd2 = 0.0, time_simd4 = 0.0, time_simd8 = 0.0;
            
            // ============= 串行实现 =============
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : guesses)
                {
                    MD5Hash(pw, state);
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_serial = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            }
            
            // ============= 二路SIMD实现 =============
            {
                auto start_hash = system_clock::now();
                bit32 state1[4], state2[4];
                
                for (size_t i = 0; i < guesses.size(); i += 2) {
                    if (i + 1 < guesses.size()) {
                        MD5Hash_SIMD(guesses[i], guesses[i+1], state1, state2);
                    } else {
                        MD5Hash(guesses[i], state1);
                    }
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_simd2 = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            }
            
            // ============= 四路SIMD实现 =============
            {
                auto start_hash = system_clock::now();
                bit32 state0[4], state1[4], state2[4], state3[4];
                bit32* states[4] = {state0, state1, state2, state3};
                string batch[4];
        
                for (size_t i = 0; i < guesses.size(); i += 4) {
                    if (i + 3 < guesses.size()) {
                        batch[0] = guesses[i];
                        batch[1] = guesses[i+1];
                        batch[2] = guesses[i+2];
                        batch[3] = guesses[i+3];
                        MD5Hash_SIMD4(batch, states);
                    } else {
                        for (size_t j = i; j < guesses.size(); j++) {
                            MD5Hash(guesses[j], state0);
                        }
                    }
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_simd4 = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            }
            
            // ============= 八路SIMD实现 =============
            {
                auto start_hash = system_clock::now();
                bit32 state0[4], state1[4], state2[4], state3[4];
                bit32 state4[4], state5[4], state6[4], state7[4];
                bit32* states[8] = {
                    state0, state1, state2, state3,
                    state4, state5, state6, state7
                };
                string batch[8];
                
                for (size_t i = 0; i < guesses.size(); i += 8) {
                    if (i + 7 < guesses.size()) {
                        batch[0] = guesses[i];
                        batch[1] = guesses[i+1];
                        batch[2] = guesses[i+2];
                        batch[3] = guesses[i+3];
                        batch[4] = guesses[i+4];
                        batch[5] = guesses[i+5];
                        batch[6] = guesses[i+6];
                        batch[7] = guesses[i+7];
                        MD5Hash_SIMD8(batch, states);
                    } else {
                        for (size_t j = i; j < guesses.size(); j++) {
                            MD5Hash(guesses[j], state0);
                        }
                    }
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_simd8 = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            }
            
            // 计算加速比
            double speedup_simd2 = time_serial / time_simd2;
            double speedup_simd4 = time_serial / time_simd4;
            double speedup_simd8 = time_serial / time_simd8;
            
            // 将本批次时间添加到总时间
            time_hash += time_simd8; // 使用八路SIMD的时间作为实际执行时间
            
            // 输出结果
            cout << "\n===== 性能比较 (密码数量: " << guesses.size() << ") =====" << endl;
            cout << "串行实现: " << time_serial << " 秒" << endl;
            cout << "二路SIMD: " << time_simd2 << " 秒 (加速比: " << speedup_simd2 << "x)" << endl;
            cout << "四路SIMD: " << time_simd4 << " 秒 (加速比: " << speedup_simd4 << "x)" << endl;
            cout << "八路SIMD: " << time_simd8 << " 秒 (加速比: " << speedup_simd8 << "x)" << endl;
            
            // 累加总数据
            total_passwords += guesses.size();
            total_serial_time += time_serial;
            total_simd2_time += time_simd2;
            total_simd4_time += time_simd4;
            total_simd8_time += time_simd8;

            // 保存结果到CSV文件
            static bool first_write = true;
            ofstream csv_file;
            if (first_write) {
                csv_file.open("md5_performance_O2.csv");
                csv_file << "批次,密码数量,串行时间(秒),二路SIMD时间(秒),四路SIMD时间(秒),八路SIMD时间(秒),二路加速比,四路加速比,八路加速比\n";
                first_write = false;
            } else {
                csv_file.open("md5_performance_O2.csv", ios::app);
            }
            
            // 写入当前批次数据
            static int batch_count = 0;
            batch_count++;
            csv_file << batch_count << "," 
                     << guesses.size() << ","
                     << time_serial << "," 
                     << time_simd2 << "," 
                     << time_simd4 << "," 
                     << time_simd8 << "," 
                     << speedup_simd2 << "," 
                     << speedup_simd4 << "," 
                     << speedup_simd8 << "\n";
            csv_file.close();
            
            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    // 在程序结束时添加总计汇总数据
    // 计算总体加速比
    double total_speedup_simd2 = total_serial_time / total_simd2_time;
    double total_speedup_simd4 = total_serial_time / total_simd4_time;
    double total_speedup_simd8 = total_serial_time / total_simd8_time;
    
    // 输出总汇总
    cout << "\n===== 总体性能汇总 (总密码数量: " << total_passwords << ") =====" << endl;
    cout << "串行实现总时间: " << total_serial_time << " 秒" << endl;
    cout << "二路SIMD总时间: " << total_simd2_time << " 秒 (总加速比: " << total_speedup_simd2 << "x)" << endl;
    cout << "四路SIMD总时间: " << total_simd4_time << " 秒 (总加速比: " << total_speedup_simd4 << "x)" << endl;
    cout << "八路SIMD总时间: " << total_simd8_time << " 秒 (总加速比: " << total_speedup_simd8 << "x)" << endl;
    
    // 计算每秒处理密码数
    cout << "串行处理速度: " << total_passwords / total_serial_time << " 密码/秒" << endl;
    cout << "二路SIMD处理速度: " << total_passwords / total_simd2_time << " 密码/秒" << endl;
    cout << "四路SIMD处理速度: " << total_passwords / total_simd4_time << " 密码/秒" << endl;
    cout << "八路SIMD处理速度: " << total_passwords / total_simd8_time << " 密码/秒" << endl;
    
    // 将总汇总添加到CSV文件
    ofstream csv_file;
    csv_file.open("md5_performance_O2.csv", ios::app);
    
    // 添加空行和标题行
    csv_file << "\n总计汇总,总密码数量,串行总时间(秒),二路SIMD总时间(秒),四路SIMD总时间(秒),八路SIMD总时间(秒),二路总加速比,四路总加速比,八路总加速比\n";
    
    // 写入总汇总数据
    csv_file << "总计," 
             << total_passwords << ","
             << total_serial_time << "," 
             << total_simd2_time << "," 
             << total_simd4_time << "," 
             << total_simd8_time << "," 
             << total_speedup_simd2 << "," 
             << total_speedup_simd4 << "," 
             << total_speedup_simd8 << "\n";
             
    // 添加处理速度信息
    csv_file << "每秒处理密码数," 
             << "," 
             << total_passwords / total_serial_time << "," 
             << total_passwords / total_simd2_time << "," 
             << total_passwords / total_simd4_time << "," 
             << total_passwords / total_simd8_time << ",,,\n";
    
    csv_file.close();
    
    return 0;
}