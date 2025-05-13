#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include<pthread.h>
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

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    //表示更新segments的值到哪里了，往后只更新pivot之后的值
    // 例如，L6D1的content大小为2，pivot=0表示更新L6的值，pivot=1表示更新D1的值
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    // 作用：根据当前的 PT 实例，生成一组新的、可能概率略低的 PT 实例。
    // 这通常用于在密码猜测过程中，当一个 PT 的所有高概率具体值被探索完后，
    // 衍生出新的 PT 结构或值的组合进行进一步探索。
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    // 这个向量记录了除了最后一个分段之外的每个分段所选取的具体值，
    // 在其对应类型（如字母、数字）的值列表（通常按概率排序）中的索引。
    // 例如，如果 L6S1 中的 L6 被实例化为 "abcdef"，并且 "abcdef" 在所有 L6 可能值中的索引是 2，
    // 那么 curr_indices 中对应 L6 的条目就是 2。
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    // 作用：存储 PT 中每个分段（通常除了最后一个）在其对应类型的值列表中有多少个可能的值。
    // 例如，如果模型中有 100 种不同的 L6 类型的具体值，那么 L6 对应的 max_indices 条目就是 100。
    // 这用于确定 curr_indices 中索引的有效范围（0 到 max_indices[x] - 1）。
    vector<int> max_indices;
    // void init();

    // 作用：存储这个 PT 结构本身（例如 L6S1 这种模式）在模型中出现的原始概率或频率派生的概率，
    // 在任何具体值被实例化之前。
    float preterm_prob;

    // 作用：存储当前 PT 实例的计算概率。这个概率是 preterm_prob 与其所有已实例化分段的具体值的概率的乘积。
    // 这个值会随着 PT 中分段的具体值变化而更新。
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    // 作用：这些变量用作计数器，为模型中新遇到的不同类型的结构（PT - 密码模板，LDS - 可能指字母/数字/符号分段类型）分配唯一的ID。
    // 每次遇到一个新的、之前未见过的结构时，对应的ID会递增。
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
    // 作用：记录模型中观察到的所有 PT（密码模板）实例的总数或总频率。
    int total_preterm = 0;

    // 作用：存储模型中所有独特的 PT 结构（例如 L6S1, D8 等）。
    vector<PT> preterminals;

     // 作用：在 `preterminals` 向量中查找给定的 PT 结构，并返回其索引（或ID）。如果未找到，可能会添加新的 PT 并返回其新分配的 ID。
    int FindPT(PT pt);

    // 作用：分别存储模型中所有独特的字母类型分段（如 L1, L2, ...）、数字类型分段（如 D1, D2, ...）和符号类型分段（如 S1, S2, ...）的定义和统计信息。
    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;

    // 作用：分别在 `letters`, `digits`, `symbols` 向量中查找给定的分段结构，并返回其索引（或ID）。
    // 如果未找到，可能会添加新的分段并返回其新分配的 ID。
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    // 作用：这些哈希表用于存储各种结构类型（PT、字母分段、数字分段、符号分段）的频率。
    // 键通常是对应结构类型的 ID（由 GetNext...ID 或 Find... 函数返回），值是该结构在训练数据中出现的次数（频率）。
    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    // 作用：存储模型中所有的 PT 结构，并且这些 PT 结构可能是按照某种顺序（例如，按概率或频率降序）排列的。
    // 这可以用于优先队列的初始化，以便从最可能的密码模板开始生成猜测。
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

// 线程参数结构体
struct ThreadGenerateArgs {
    segment* segment_data;      // 指向 segment 对象的指针
    int start_index;            // 当前线程处理的起始索引
    int end_index;              // 当前线程处理的结束索引 (不包含)
    std::string prefix_str;     // 字符串前缀
    std::vector<std::string> thread_local_guesses; // 线程本地存储生成的猜测
    int thread_local_guess_count; // 线程本地记录生成的猜测数量
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    // 虽然名为 "priority queue"，但这里是使用 std::vector 手动实现的，
    // 这意味着元素的插入和删除（特别是保持顺序）需要自定义逻辑，
    // 而不是直接使用 std::priority_queue。队列中的 PT 对象可能根据其 `prob` 成员排序。
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    // 作用：持有一个 PCFG 模型的实例。这个模型包含了生成密码猜测所需的统计信息和结构定义（如各种分段的概率、具体值的频率等）。
    // PriorityQueue 类的方法会使用这个 `m` 对象来计算 PT 的概率、获取分段的具体值等。
    model m;

    // 计算一个pt的概率
    // 作用：计算给定 PT 实例的概率。这个计算通常会考虑 PT 结构本身的概率 (preterm_prob)
    // 以及其已实例化分段的具体值的概率（基于模型 `m` 中的统计数据）。
    // 计算结果会更新 pt 对象的 `prob` 成员。
    void CalProb(PT &pt);

    // 优先队列的初始化
    // 作用：初始化优先队列。这可能包括从模型 `m` 中获取初始的一批 PT（例如，`m.ordered_pts`），
    // 为它们计算初始概率，并将它们添加到 `priority` 向量中，可能会进行初步的排序。
    void init();

    // 对优先队列的一个PT，生成所有guesses
    // 作用：根据给定的 PT 实例生成具体的密码猜测。
    // 如果 PT 只有一个分段，它会遍历该分段所有可能的值。
    // 如果 PT 有多个分段，它通常会实例化最后一个未实例化的分段的所有可能值，
    // 而前面的分段的值已经由 pt.curr_indices 确定。
    // 生成的猜测会存储在 `guesses` 成员变量中，并且 `total_guesses` 会相应增加。
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    // 作用：处理优先队列中的下一个（通常是概率最高的）PT。
    // 这个过程一般包括：
    // 1. 从 `priority` 队列的前端取出一个 PT。
    // 2. 调用 `Generate` 函数，用这个 PT 生成一批密码猜测。
    // 3. 调用该 PT 的 `NewPTs()` 方法，生成一组新的、可能概率稍低的 PT 变体。
    // 4. 为这些新的 PT 计算概率（使用 `CalProb`）。
    // 5. 将这些新的 PT 按照其概率插入到 `priority` 队列的合适位置，以维持队列的有序性。
    // 6. 最后，从 `priority` 队列中移除已经处理过的那个 PT。
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;

    // 用于保护 guesses 和 total_guesses 的互斥锁
    pthread_mutex_t guesses_mutex; 

    /*// 构造函数，用于初始化互斥锁
    PriorityQueue() {
        pthread_mutex_init(&guesses_mutex, NULL);
    }

    // 析构函数，用于销毁互斥锁
    ~PriorityQueue() {
        pthread_mutex_destroy(&guesses_mutex);
    }*/

     // --- 线程池成员 ---
    pthread_t* worker_threads_pool;
    int pool_size;
    std::queue<ThreadGenerateArgs*> task_queue_pool; // 存储指向任务参数的指针
    pthread_mutex_t pool_queue_mutex;          // 保护任务队列和活动任务计数器
    pthread_cond_t pool_cond_task_available;   // 当任务队列非空时通知工作线程
    pthread_cond_t pool_cond_tasks_completed;  // 当一批任务完成时通知 Generate 函数
    int pool_tasks_in_progress;                // 当前在队列中或正在被处理的任务数量
    bool pool_shutdown_flag;                   // 通知工作线程关闭
    
    // 任务参数缓冲区，由 Generate 函数填充，供工作线程使用
    // 大小将在 InitializeThreadPool 中根据 pool_size 确定并分配
    ThreadGenerateArgs* task_args_buffer_for_generate; 

    // 工作线程函数
    static void* WorkerThreadFunction(void* arg);

    // 线程池管理
    void InitializeThreadPool(int num_threads);
    void DestroyThreadPool();

    // 构造函数，用于初始化互斥锁和线程池
    PriorityQueue(int num_pool_threads = 4) { // 默认线程池大小为4
        pthread_mutex_init(&guesses_mutex, NULL);
        pthread_mutex_init(&pool_queue_mutex, NULL);
        pthread_cond_init(&pool_cond_task_available, NULL);
        pthread_cond_init(&pool_cond_tasks_completed, NULL);
        worker_threads_pool = nullptr;
        task_args_buffer_for_generate = nullptr;
        pool_tasks_in_progress = 0;
        pool_shutdown_flag = false;
        InitializeThreadPool(num_pool_threads > 0 ? num_pool_threads : 1); // 确保至少1个线程
    }

    // 析构函数，用于销毁互斥锁和线程池
    ~PriorityQueue() {
        DestroyThreadPool(); // 确保线程池先于其他成员销毁
        pthread_mutex_destroy(&guesses_mutex);
        pthread_mutex_destroy(&pool_queue_mutex);
        pthread_cond_destroy(&pool_cond_task_available);
        pthread_cond_destroy(&pool_cond_tasks_completed);
    }
};
