#include "PCFG.h"
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    // 将 pt 对象的 prob 成员（用于存储最终计算出的概率）初始化为 pt 对象的 preterm_prob 成员的值。
    // preterm_prob 代表了这个 PT 结构（如 L6S1）本身出现的概率，这是概率计算的基础。
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            //乘频数除总数：即乘该segment的概率
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
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
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 线程参数结构体
struct ThreadGenerateArgs {
    segment* segment_data;      // 指向 segment 对象的指针
    int start_index;            // 当前线程处理的起始索引
    int end_index;              // 当前线程处理的结束索引 (不包含)
    std::string prefix_str;     // 字符串前缀
    std::vector<std::string> thread_local_guesses; // 线程本地存储生成的猜测
    int thread_local_guess_count; // 线程本地记录生成的猜测数量
};

// 线程工作函数
void* process_segment_range(void* args_ptr) {
    ThreadGenerateArgs* args = (ThreadGenerateArgs*)args_ptr;
    args->thread_local_guess_count = 0;
    // args->thread_local_guesses.clear(); // 如果 ThreadGenerateArgs 对象被复用，确保清空

    for (int i = args->start_index; i < args->end_index; ++i) {
        args->thread_local_guesses.emplace_back(args->prefix_str + args->segment_data->ordered_values[i]);
        args->thread_local_guess_count++;
    }

    // 不再在此处加锁和更新共享数据
    pthread_exit(NULL);
}

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// 该函数接收一个 PT 对象 pt（按值传递，意味着函数操作的是 pt 的一个副本）。
// 这个函数是 PCFG 并行化算法的主要载体，主要负责根据一个 PT 生成具体的密码猜测。
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    // 调用 CalProb 函数，计算传入的 pt 对象的概率。
    // 注意：由于 pt 是按值传递的，这里的 CalProb(pt) 修改的是 pt 这个副本的 prob 成员，
    // 而不是优先队列中原始 PT 对象的 prob。
    // 这通常意味着 Generate 函数期望接收一个已经设置好 preterm_prob 和 curr_indices 的 PT。
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        if (!a || pt.max_indices.empty()) { // 安全检查
            return;
        }

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        /*int num_total_values = pt.max_indices[0];
        if (num_total_values == 0) {
            return;
        }

        const int NUM_THREADS = 4; // 线程数，先尝试四线程的并行化
        pthread_t threads[NUM_THREADS];
        ThreadGenerateArgs thread_args[NUM_THREADS];
        
        int items_per_thread = num_total_values / NUM_THREADS;
        int remaining_items = num_total_values % NUM_THREADS;
        int current_start_idx = 0;

        for (int i = 0; i < NUM_THREADS; ++i) {
            int items_for_this_thread = items_per_thread + (i < remaining_items ? 1 : 0);
            if (items_for_this_thread == 0 && current_start_idx >= num_total_values) { // 如果没有更多任务，则不创建线程
                 threads[i] = 0; // 标记为无效/未创建
                 continue;
            }
            
            thread_args[i].segment_data = a;
            thread_args[i].start_index = current_start_idx;
            thread_args[i].end_index = current_start_idx + items_for_this_thread;
            thread_args[i].prefix_str = ""; // 单 segment 情况，前缀为空

            if (pthread_create(&threads[i], NULL, process_segment_range, &thread_args[i]) != 0) {
                // 处理线程创建错误，例如打印错误信息
                perror("Error creating thread");
                threads[i] = 0; // 标记为无效
            }
            current_start_idx += items_for_this_thread;
        }

        for (int i = 0; i < NUM_THREADS; ++i) {
            if (threads[i] != 0) { // 仅 join 已成功创建的线程
                pthread_join(threads[i], NULL);
            }
        }
        
         // 所有线程完成后，统一收集结果
        pthread_mutex_lock(&guesses_mutex); // 假设 guesses_mutex 是 PriorityQueue 的成员并已初始化
        for (int i = 0; i < NUM_THREADS; ++i) {
            if (threads[i] != 0) { // 仅处理成功创建并完成的线程的结果
                guesses.insert(guesses.end(), thread_args[i].thread_local_guesses.begin(), thread_args[i].thread_local_guesses.end());
                total_guesses += thread_args[i].thread_local_guess_count;
            }
        }
        pthread_mutex_unlock(&guesses_mutex);*/

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // 安全检查: 确保 last_segment_data 有效，并且 pt.max_indices 对最后一个分段有有效条目
        if (!a || 
            pt.max_indices.size() <= static_cast<size_t>(pt.content.size() - 1) || 
            pt.max_indices[pt.content.size() - 1] == 0) {
             return;
        }

        /*int num_total_values_last_segment = pt.max_indices[pt.content.size() - 1];

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        const int NUM_THREADS_MULTI = 4; // 示例线程数
        pthread_t threads_multi[NUM_THREADS_MULTI];
        ThreadGenerateArgs thread_args_multi[NUM_THREADS_MULTI]; // 为多分段情况使用独立的参数数组

        int items_per_thread_multi = num_total_values_last_segment / NUM_THREADS_MULTI;
        int remaining_items_multi = num_total_values_last_segment % NUM_THREADS_MULTI;
        int current_start_idx_multi = 0;

        for (int i = 0; i < NUM_THREADS_MULTI; ++i) {
            int items_for_this_thread = items_per_thread_multi + (i < remaining_items_multi ? 1 : 0);
            if (items_for_this_thread == 0 && current_start_idx_multi >= num_total_values_last_segment) {
                threads_multi[i] = 0; // 标记为无效/未创建
                continue;
            }

            thread_args_multi[i].segment_data = a;
            thread_args_multi[i].start_index = current_start_idx_multi;
            thread_args_multi[i].end_index = current_start_idx_multi + items_for_this_thread;
             thread_args_multi[i].prefix_str = guess; // 使用之前构建的前缀

            if (pthread_create(&threads_multi[i], NULL, process_segment_range, &thread_args_multi[i]) != 0) {
                perror("Error creating thread for multi-segment");
                threads_multi[i] = 0; // 标记为无效
            }
            current_start_idx_multi += items_for_this_thread;
        }

        for (int i = 0; i < NUM_THREADS_MULTI; ++i) {
            if (threads_multi[i] != 0) { // 仅 join 已成功创建的线程
                pthread_join(threads_multi[i], NULL);
            }
        }

        // 所有线程完成后，统一收集结果
        pthread_mutex_lock(&guesses_mutex); // 假设 guesses_mutex 是 PriorityQueue 的成员并已初始化
        for (int i = 0; i < NUM_THREADS_MULTI; ++i) {
            if (threads_multi[i] != 0) { // 仅处理成功创建并完成的线程的结果
                guesses.insert(guesses.end(), thread_args_multi[i].thread_local_guesses.begin(), thread_args_multi[i].thread_local_guesses.end());
                total_guesses += thread_args_multi[i].thread_local_guess_count;
            }
        }
        pthread_mutex_unlock(&guesses_mutex);*/
        
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}