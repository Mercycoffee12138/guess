#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    // 原始输入字符串
    string input = "dbwuidwuihncuiwncuiwhnduilhweuikfhewilidhqihdjqwhwdhqidwhqihwudhuwhslqhwchucedhbuehhwlhwujchwuikhdlwhsqioluuhdwilhncwlscnsqjkwbcyukhwbshdkqvbwdywukdbyhukwvdsgvhjdhwvdhjwvejwhvgdxgqvwdyiwkgfcyeidgdowddgebqkcvolqygsbkcqvxukvxyqhkqbxqvwykd";
    
    // 为SIMD版本准备输入数组和状态数组
    string inputs[4] = {input, input, input, input};
    bit32 states[4][4];
    
    // 调用SIMD版本的MD5哈希函数
    MD5Hash_SIMD(inputs, states);
    
    // 输出第一个哈希结果（应与原始MD5Hash相同）
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][0];
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][1];
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][2];
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][3];
        cout << endl;
    }
    cout << endl;
    
    // 可选：验证与原始函数结果相同
    bit32 state[4];
    MD5Hash(input, state);
    
    cout << "原始MD5Hash结果: ";
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;
    
    // 验证两个结果是否相同
    bool match = true;
    for (int i = 0; i < 4; i++) {
        if (state[i] != states[0][i]) {
            match = false;
            break;
        }
    }
    
    cout << "验证结果: " << (match ? "相同" : "不同") << endl;
    cout << endl;

    // 验证八线程版本的正确性
    cout << "验证八线程版本的正确性:" << endl;
    
    // 为八线程版本准备输入数组和状态数组
    string inputs8[8] = {input, input, input, input, input, input, input, input};
    bit32 states8[8][4];
    
    // 调用八线程版本的MD5哈希函数
    MD5Hash_SIMD8(inputs8, states8);
    
    // 输出八线程版本的哈希结果
    for (int i1 = 0; i1 < 8; i1 += 1)
    {
        cout << "线程 " << i1 << " 结果: ";
        cout << std::setw(8) << std::setfill('0') << hex << states8[i1][0];
        cout << std::setw(8) << std::setfill('0') << hex << states8[i1][1];
        cout << std::setw(8) << std::setfill('0') << hex << states8[i1][2];
        cout << std::setw(8) << std::setfill('0') << hex << states8[i1][3];
        cout << endl;
    }
    cout << endl;
    
    // 验证八线程结果是否与原始MD5Hash相同
    bool match8 = true;
    for (int i = 0; i < 8; i++) {
        bool threadMatch = true;
        for (int j = 0; j < 4; j++) {
            if (state[j] != states8[i][j]) {
                threadMatch = false;
                break;
            }
        }
        if (!threadMatch) {
            match8 = false;
            cout << "线程 " << i << " 结果不匹配" << endl;
        }
    }
    
    cout << "八线程验证结果: " << (match8 ? "全部相同" : "存在不同") << endl;

    CleanupMD5Resources();
    return 0;
}