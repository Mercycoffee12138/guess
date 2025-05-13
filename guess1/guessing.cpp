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

// --- 线程池实现 ---

void PriorityQueue::InitializeThreadPool(int num_threads) {
    if (num_threads <= 0) {
        // std::cerr << "Warning: Thread pool size must be positive. Defaulting to 1 thread." << std::endl;
        num_threads = 1;
    }
    this->pool_size = num_threads;
    this->pool_shutdown_flag = false;
    this->pool_tasks_in_progress = 0;

    // 为任务参数分配缓冲区
    // Generate 函数将填充这些结构体，然后将其指针添加到任务队列中
    this->task_args_buffer_for_generate = new ThreadGenerateArgs[pool_size];
    for(int i=0; i<pool_size; ++i) {
        //确保每个vector是空的，以防万一
        this->task_args_buffer_for_generate[i].thread_local_guesses.reserve(100); // 预分配一些空间
    }


    this->worker_threads_pool = new pthread_t[pool_size];
    for (int i = 0; i < pool_size; ++i) {
        if (pthread_create(&worker_threads_pool[i], NULL, WorkerThreadFunction, this) != 0) {
            std::cerr << "Error creating thread " << i << std::endl;
            // 错误处理：可能需要关闭已创建的线程并抛出异常
            this->pool_size = i; // 记录实际创建的线程数
            // 在这种情况下，析构函数最终会调用 DestroyThreadPool
            // 或者我们可以立即尝试清理
            // For simplicity here, we let destructor handle partial cleanup if constructor fails mid-way.
            // However, a more robust approach would be to clean up here.
            delete[] this->task_args_buffer_for_generate;
            this->task_args_buffer_for_generate = nullptr;
            delete[] this->worker_threads_pool;
            this->worker_threads_pool = nullptr;
            throw std::runtime_error("Failed to create worker thread in InitializeThreadPool");
        }
    }
}

void PriorityQueue::DestroyThreadPool() {
    if (worker_threads_pool == nullptr && task_args_buffer_for_generate == nullptr) { // 已经销毁或未初始化
        return;
    }

    pthread_mutex_lock(&pool_queue_mutex);
    pool_shutdown_flag = true;
    // 清理任务队列中的任何挂起任务
    while(!task_queue_pool.empty()) {
        task_queue_pool.pop();
    }
    // pool_tasks_in_progress = 0; // 重置进行中的任务，确保等待条件能被满足
    pthread_cond_broadcast(&pool_cond_task_available); // 唤醒所有等待的线程，以便它们检查关闭标志
    pthread_mutex_unlock(&pool_queue_mutex);

    if (worker_threads_pool != nullptr) {
        for (int i = 0; i < pool_size; ++i) {
            // 检查线程是否已成功创建并记录（例如，通过检查非零的 pthread_t 值，但这不可靠）
            // 假设所有在 InitializeThreadPool 中尝试创建的线程都需要 join
            // 如果 InitializeThreadPool 中途失败，pool_size 可能小于原始请求值
            if (worker_threads_pool[i] != 0) { // 简单的检查，假设0表示未使用的槽位或失败
                 pthread_join(worker_threads_pool[i], NULL);
            }
        }
        delete[] worker_threads_pool;
        worker_threads_pool = nullptr;
    }

    if (task_args_buffer_for_generate != nullptr) {
        delete[] task_args_buffer_for_generate; // 释放任务参数缓冲区
        task_args_buffer_for_generate = nullptr;
    }
    // 互斥锁和条件变量在 PriorityQueue 的析构函数中销毁
}

// static
void* PriorityQueue::WorkerThreadFunction(void* arg) {
    PriorityQueue* pq = static_cast<PriorityQueue*>(arg);

    while (true) {
        ThreadGenerateArgs* task_args = nullptr;

        pthread_mutex_lock(&pq->pool_queue_mutex);
        while (pq->task_queue_pool.empty() && !pq->pool_shutdown_flag) {
            pthread_cond_wait(&pq->pool_cond_task_available, &pq->pool_queue_mutex);
        }

        if (pq->pool_shutdown_flag && pq->task_queue_pool.empty()) {
            pthread_mutex_unlock(&pq->pool_queue_mutex);
            pthread_exit(NULL);
        }

        if (!pq->task_queue_pool.empty()) {
            task_args = pq->task_queue_pool.front();
            pq->task_queue_pool.pop();
        } else { // 如果因为 shutdown 标志而退出循环，但队列仍可能为空
            pthread_mutex_unlock(&pq->pool_queue_mutex);
            continue; // 重新检查 shutdown 标志
        }
        pthread_mutex_unlock(&pq->pool_queue_mutex);

        if (task_args) {
            task_args->thread_local_guesses.clear(); // 为新结果清空
            task_args->thread_local_guess_count = 0;

            if (task_args->segment_data != nullptr) { // 安全检查
                for (int i = task_args->start_index; i < task_args->end_index; ++i) {
                    // 确保 i 在 ordered_values 的有效范围内
                    if (i < task_args->segment_data->ordered_values.size()) {
                        std::string current_guess = task_args->prefix_str + task_args->segment_data->ordered_values[i];
                        task_args->thread_local_guesses.push_back(current_guess);
                        task_args->thread_local_guess_count++;
                    } else {
                        // 如果索引超出范围，这可能表示 pt.max_indices 与实际 segment 大小不匹配
                        // 或者任务分配逻辑有误
                        std::cerr << "Warning: Worker thread index " << i << " out of bounds for segment." << std::endl;
                        break; 
                    }
                }
            }

            pthread_mutex_lock(&pq->pool_queue_mutex);
            pq->pool_tasks_in_progress--;
            if (pq->pool_tasks_in_progress == 0) {
                pthread_cond_signal(&pq->pool_cond_tasks_completed);
            }
            pthread_mutex_unlock(&pq->pool_queue_mutex);
        }
    }
    return nullptr;
}

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// 该函数接收一个 PT 对象 pt（按值传递，意味着函数操作的是 pt 的一个副本）。
// 这个函数是 PCFG 并行化算法的主要载体，主要负责根据一个 PT 生成具体的密码猜测。
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    segment *target_segment_ptr = nullptr;
    int total_values_in_segment = 0;
    string current_prefix_str = ""; // 使用局部变量以避免多线程问题（如果Generate本身可重入）

    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        const auto& seg_content = pt.content[0];
        if (seg_content.type == 1) target_segment_ptr = &m.letters[m.FindLetter(seg_content)];
        else if (seg_content.type == 2) target_segment_ptr = &m.digits[m.FindDigit(seg_content)];
        else if (seg_content.type == 3) target_segment_ptr = &m.symbols[m.FindSymbol(seg_content)];
        
        if (!pt.max_indices.empty()) {
            total_values_in_segment = pt.max_indices[0];
        } else {
            std::cerr << "Warning: pt.max_indices is empty for single segment PT. Falling back to segment's value count." << std::endl;
            if (target_segment_ptr) total_values_in_segment = target_segment_ptr->ordered_values.size();
            else total_values_in_segment = 0;
        }
        // current_prefix_str 保持为空

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
             string guess = a->ordered_values[i];
            // cout << guess << endl;
             guesses.emplace_back(guess);
             total_guesses += 1;
        }
    }
    else // pt.content.size() > 1
    {
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        for (int idx : pt.curr_indices)
        {
            // 确保 idx 在有效范围内
            if (pt.content[seg_idx].type == 1) {
                const auto& letter_seg_values = m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values;
                if (idx < letter_seg_values.size()) current_prefix_str += letter_seg_values[idx];
                else { /* Handle error or inconsistent data */ std::cerr << "Index out of bounds for letter segment in prefix construction." << std::endl; return; }
            } else if (pt.content[seg_idx].type == 2) {
                const auto& digit_seg_values = m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values;
                if (idx < digit_seg_values.size()) current_prefix_str += digit_seg_values[idx];
                else { /* Handle error */ std::cerr << "Index out of bounds for digit segment in prefix construction." << std::endl; return; }
            } else if (pt.content[seg_idx].type == 3) {
                const auto& symbol_seg_values = m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values;
                if (idx < symbol_seg_values.size()) current_prefix_str += symbol_seg_values[idx];
                else { /* Handle error */ std::cerr << "Index out of bounds for symbol segment in prefix construction." << std::endl; return; }
            }
            
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) { // 已经处理了倒数第二个 segment
                break;
            }
        }

        // 指向最后一个segment的指针
        const auto& last_segment_content = pt.content.back();
        if (last_segment_content.type == 1) target_segment_ptr = &m.letters[m.FindLetter(last_segment_content)];
        else if (last_segment_content.type == 2) target_segment_ptr = &m.digits[m.FindDigit(last_segment_content)];
        else if (last_segment_content.type == 3) target_segment_ptr = &m.symbols[m.FindSymbol(last_segment_content)];

        if (pt.max_indices.size() == pt.content.size()) {
             total_values_in_segment = pt.max_indices.back();
        } else {
            std::cerr << "Warning: pt.max_indices size mismatch or empty for multi-segment PT. Falling back to segment's value count." << std::endl;
            if (target_segment_ptr) total_values_in_segment = target_segment_ptr->ordered_values.size();
            else total_values_in_segment = 0;
        }
    }

    if (!target_segment_ptr || total_values_in_segment == 0) {
        // 没有工作可做或状态无效
        return;
    }
    
    // --- 线程池任务提交 ---
    pthread_mutex_lock(&pool_queue_mutex);
    // 确保 Generate 的此调用没有正在进行的先前任务
    // 假设 Generate 由 PopNext 顺序调用，因此 pool_tasks_in_progress 此时应为0
    // 如果 Generate 可以并发调用，则需要更强大的状态管理

    int num_actual_threads_to_use = std::min(this->pool_size, total_values_in_segment);
    // 如果有工作但池大小为0（不应发生，因为构造函数确保至少1个）或 total_values_in_segment < pool_size
    if (num_actual_threads_to_use == 0 && total_values_in_segment > 0) {
        num_actual_threads_to_use = 1; // 如果有工作，至少使用一个“任务槽”
    }
    
    if (num_actual_threads_to_use == 0) { // 没有工作或没有线程可用
        pthread_mutex_unlock(&pool_queue_mutex);
        return;
    }

    int items_per_thread_base = total_values_in_segment / num_actual_threads_to_use;
    int remaining_items_distribute = total_values_in_segment % num_actual_threads_to_use;
    int current_item_start_index = 0;
    int tasks_submitted_count = 0;

    for (int i = 0; i < num_actual_threads_to_use; ++i) {
        if (current_item_start_index >= total_values_in_segment) {
            break; // 没有更多项目可分配
        }

        ThreadGenerateArgs* current_task_args_ptr = &task_args_buffer_for_generate[i];
        
        current_task_args_ptr->segment_data = target_segment_ptr;
        current_task_args_ptr->prefix_str = current_prefix_str; // 复制前缀
        current_task_args_ptr->start_index = current_item_start_index;
        
        int items_for_this_task = items_per_thread_base + (i < remaining_items_distribute ? 1 : 0);
        current_task_args_ptr->end_index = current_item_start_index + items_for_this_task;
        
        // 确保 end_index 不超过 total_values_in_segment
        if (current_task_args_ptr->end_index > total_values_in_segment) {
            current_task_args_ptr->end_index = total_values_in_segment;
        }
        
        // 如果 start_index 不小于 end_index（例如，items_for_this_task 为0），则跳过此任务
        if (current_task_args_ptr->start_index >= current_task_args_ptr->end_index) {
            continue;
        }

        // current_task_args_ptr->thread_local_guesses.clear(); // 在 WorkerThreadFunction 中完成
        // current_task_args_ptr->thread_local_guess_count = 0; // 在 WorkerThreadFunction 中完成

        task_queue_pool.push(current_task_args_ptr);
        pool_tasks_in_progress++;
        tasks_submitted_count++;
        current_item_start_index = current_task_args_ptr->end_index;
    }
    
    if (tasks_submitted_count > 0) {
        pthread_cond_broadcast(&pool_cond_task_available); // 通知所有工作线程任务已分发
    }
    pthread_mutex_unlock(&pool_queue_mutex);

    // --- 等待任务完成 ---
    if (tasks_submitted_count > 0) {
        pthread_mutex_lock(&pool_queue_mutex);
        // 等待由此 Generate 调用提交的任务完成
        while (pool_tasks_in_progress > 0) { 
            pthread_cond_wait(&pool_cond_tasks_completed, &pool_queue_mutex);
        }
        pthread_mutex_unlock(&pool_queue_mutex);
    }

    // --- 收集结果 ---
    // 此锁保护 'guesses' 和 'total_guesses'
    pthread_mutex_lock(&guesses_mutex);
    for (int i = 0; i < tasks_submitted_count; ++i) { // 仅从本次调用提交的任务中收集
        ThreadGenerateArgs* completed_task_args = &task_args_buffer_for_generate[i];
        if (!completed_task_args->thread_local_guesses.empty()) {
            // 为了效率，可以考虑使用 std::move 或者直接遍历添加
            guesses.insert(guesses.end(), 
                           std::make_move_iterator(completed_task_args->thread_local_guesses.begin()),
                           std::make_move_iterator(completed_task_args->thread_local_guesses.end()));
            completed_task_args->thread_local_guesses.clear(); // 移动后清空
        }
        total_guesses += completed_task_args->thread_local_guess_count;
    }
    pthread_mutex_unlock(&guesses_mutex);

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
             string guess = a->ordered_values[i];
            // cout << guess << endl;
             guesses.emplace_back(guess);
             total_guesses += 1;
        }
}