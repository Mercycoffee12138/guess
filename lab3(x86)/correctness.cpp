#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <windows.h>
using namespace std;
using namespace chrono;

// SIMD版本的函数声明
void MD5Hash_SIMD(string input1, string input2, bit32 *state1, bit32 *state2);
void MD5Hash_SIMD4(string input[4], bit32 *state[4]);
void MD5Hash_SIMD8(string input[8], bit32 *state[8]);

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性并比较性能
int main()
{
    // 设置控制台代码页为UTF-8
    SetConsoleOutputCP(CP_UTF8);

    // 测试的长字符串
    string testStr = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    
    cout << "===================== 正确性验证 =====================" << endl;
    
    // ================ 八路SIMD验证 ==================
    cout << endl << "---------------------- 八路SIMD测试 ----------------------" << endl;
    
    // 准备八个测试字符串(都相同)
    string testInputs8[8] = {testStr, testStr, testStr, testStr, testStr, testStr, testStr, testStr};
    
    // 使用串行MD5算法计算8个哈希
    bit32 serialStates8[8][4];
    auto start_serial8 = high_resolution_clock::now();
    for (int i = 0; i < 8; i++) {
        MD5Hash(testInputs8[i], serialStates8[i]);
    }
    auto end_serial8 = high_resolution_clock::now();
    
    cout << "串行MD5结果1: ";
    for (int i = 0; i < 4; i++) {
        cout << std::setw(8) << std::setfill('0') << hex << serialStates8[0][i];
    }
    cout << endl;
    
    // 使用八路SIMD版本计算同时计算8个字符串的MD5
    bit32 simd8State0[4], simd8State1[4], simd8State2[4], simd8State3[4];
    bit32 simd8State4[4], simd8State5[4], simd8State6[4], simd8State7[4];
    bit32* simd8States[8] = {
        simd8State0, simd8State1, simd8State2, simd8State3,
        simd8State4, simd8State5, simd8State6, simd8State7
    };
    
    auto start_simd8 = high_resolution_clock::now();
    MD5Hash_SIMD8(testInputs8, simd8States);
    auto end_simd8 = high_resolution_clock::now();
    
    cout << "八路SIMD结果1: ";
    for (int i = 0; i < 4; i++) {
        cout << std::setw(8) << std::setfill('0') << hex << simd8States[0][i];
    }
    cout << endl;
    
    // 验证所有结果是否一致
    bool correct8 = true;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            if (serialStates8[i][j] != simd8States[i][j]) {
                correct8 = false;
                cout << "不匹配: 输入 " << i << ", 位置 " << j << ", 串行值=" 
                     << serialStates8[i][j] << ", SIMD值=" << simd8States[i][j] << endl;
            }
        }
    }
    
    if (correct8) {
        cout << "验证通过：八路SIMD版本与串行版本计算结果完全一致！" << endl;
    } else {
        cout << "验证失败：计算结果不一致！" << endl;
    }
    
    cout << endl << "===================== 八路性能测试 =====================" << endl;
    
    // 计算时间
    auto serial8_duration = duration_cast<microseconds>(end_serial8 - start_serial8);
    auto simd8_duration = duration_cast<microseconds>(end_simd8 - start_simd8);
    
    cout << "串行MD5计算8个哈希用时: " << serial8_duration.count() << " 微秒" << endl;
    cout << "八路SIMD计算8个哈希用时: " << simd8_duration.count() << " 微秒" << endl;
    
    double speedup8 = (double)serial8_duration.count() / simd8_duration.count();
    cout << "加速比: " << speedup8 << "x" << endl;
    
    // 更大批量的性能测试 - 八路SIMD
    const int TEST_SIZE = 10000;
    string* inputs = new string[TEST_SIZE];
    bit32** results_serial = new bit32*[TEST_SIZE];
    bit32** results_simd8 = new bit32*[TEST_SIZE];
    
    // 初始化测试数据
    for (int i = 0; i < TEST_SIZE; i++) {
        inputs[i] = testStr + to_string(i);  // 添加一点变化
        results_serial[i] = new bit32[4];
        results_simd8[i] = new bit32[4];
    }
    
    // 串行计算
    auto start_serial_batch = high_resolution_clock::now();
    for (int i = 0; i < TEST_SIZE; i++) {
        MD5Hash(inputs[i], results_serial[i]);
    }
    auto end_serial_batch = high_resolution_clock::now();
    
    // 八路SIMD计算
    auto start_simd8_batch = high_resolution_clock::now();
    for (int i = 0; i < TEST_SIZE; i += 8) {
        if (i + 7 < TEST_SIZE) {
            string batch[8] = {
                inputs[i], inputs[i+1], inputs[i+2], inputs[i+3],
                inputs[i+4], inputs[i+5], inputs[i+6], inputs[i+7]
            };
            bit32* states[8] = {
                results_simd8[i], results_simd8[i+1], results_simd8[i+2], results_simd8[i+3],
                results_simd8[i+4], results_simd8[i+5], results_simd8[i+6], results_simd8[i+7]
            };
            MD5Hash_SIMD8(batch, states);
        } else {
            // 处理剩余的不足8个的输入
            for (int j = i; j < TEST_SIZE; j++) {
                MD5Hash(inputs[j], results_simd8[j]);
            }
        }
    }
    auto end_simd8_batch = high_resolution_clock::now();
    
    // 验证大批量结果
    bool correct_batch = true;
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < 4; j++) {
            if (results_serial[i][j] != results_simd8[i][j]) {
                correct_batch = false;
                cout << "批量不匹配: 输入 " << i << ", 位置 " << j << endl;
                break;
            }
        }
        if (!correct_batch) break;
    }
    
    auto serial_batch_duration = duration_cast<milliseconds>(end_serial_batch - start_serial_batch);
    auto simd8_batch_duration = duration_cast<milliseconds>(end_simd8_batch - start_simd8_batch);
    
    cout << endl << "大批量测试(" << TEST_SIZE << "个哈希)结果 - 八路SIMD:" << endl;
    cout << "串行计算用时: " << serial_batch_duration.count() << " 毫秒" << endl;
    cout << "八路SIMD计算用时: " << simd8_batch_duration.count() << " 毫秒" << endl;
    
    double speedup_batch = (double)serial_batch_duration.count() / simd8_batch_duration.count();
    cout << "大批量加速比: " << speedup_batch << "x" << endl;
    
    if (correct_batch) {
        cout << "大批量验证通过：八路SIMD版本与串行版本计算结果完全一致！" << endl;
    } else {
        cout << "大批量验证失败：计算结果不一致！" << endl;
    }
    
    // 释放内存
    for (int i = 0; i < TEST_SIZE; i++) {
        delete[] results_serial[i];
        delete[] results_simd8[i];
    }
    delete[] inputs;
    delete[] results_serial;
    delete[] results_simd8;
    
    return 0;
}