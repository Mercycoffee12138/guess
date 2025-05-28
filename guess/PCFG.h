#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include<algorithm>
// #include <chrono>   
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;
    int start_index;        // 在全局字典中的起始位置
    int container_size;     // 容器大小（经过平滑处理后的大小）
    int effective_container_size = 0; // 新增：平滑后的有效容器大小, 例如 2^n

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order(int n_smoothing_exponent = 0);// 修改：增加平滑参数，0表示不平滑
    void PrintValues();
};

//PT存储结构优化
struct OptimizedPT {
    // Header data (24 bytes)
    uint64_t offset;           // 8 bytes - 线程偏移量
    uint64_t previous_guesses; // 8 bytes - 之前的猜测数量
    uint8_t num_containers;    // 1 byte - 容器数量
    uint64_t total_guesses;    // 7 bytes - 总猜测数量（使用7字节）
    
    // Body data (最多29个容器，每个8字节)
    struct Container {
        uint8_t type;          // 1 byte - 类型
        uint8_t length;        // 1 byte - 长度  
        uint8_t start_index;   // 1 byte - 起始索引
        uint64_t container_size; // 5 bytes - 容器大小（使用5字节）
    } containers[29];
    
    // 确保总大小为256字节
    uint8_t padding[256 - 24 - 29*8];
};


class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;
};

// 全局字典结构
class GlobalDictionary {
public:
    vector<string> dictionary;              // 一维数组存储所有字符串
    vector<vector<int>> starting_positions; // T×L数组存储起始位置
    int max_types = 3;                      // 字母、数字、符号
    int max_length = 32;                    // 最大长度限制
    
    void BuildDictionary(const model& model);
    int GetPosition(int type, int length, int rank);
    string GetString(int position);
};

// 函数声明
void ConvertToOptimizedPT(const PT& pt, OptimizedPT& opt_pt);
void ApplySmoothingToSegment(segment& seg, int container_size);
void ApplySmoothingTechnique(model& model, int n = 3);

// 全局字典实例
extern GlobalDictionary global_dict;