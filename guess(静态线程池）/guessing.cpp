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

// 线程工作函数
void* process_segment_range(void* args_ptr) {
    ThreadGenerateArgs* args = static_cast<ThreadGenerateArgs*>(args_ptr);
    if (!args || !args->segment_data) {
        // This case should ideally be prevented by Generate's logic
        return NULL;
    }

    // Ensure thread_local_guesses is clean (though Generate should also clear it before task submission)
    args->thread_local_guesses.clear(); 
    args->thread_local_guess_count = 0;

    // Determine the actual number of values in the segment's ordered_values
    int actual_segment_values_count = args->segment_data->ordered_values.size();
    
    // Calculate the effective end index for the loop, ensuring it doesn't exceed available values
    int loop_end_index = std::min(args->end_index, actual_segment_values_count);

    // Ensure start_index is valid and less than loop_end_index
    if (args->start_index >= loop_end_index) {
        return NULL; // No work to do for this task or invalid range
    }

    for (int i = args->start_index; i < loop_end_index; ++i) {
        args->thread_local_guesses.emplace_back(args->prefix_str + args->segment_data->ordered_values[i]);
    }
    args->thread_local_guess_count = args->thread_local_guesses.size();
    
    return NULL; // The task is done; worker_thread_function handles pthread_exit logic
}

void* PriorityQueue::worker_thread_function(void* arg) {
    PriorityQueue* pq_instance = static_cast<PriorityQueue*>(arg);

    while (true) {
        ThreadGenerateArgs* task_args = nullptr;

        pthread_mutex_lock(&pq_instance->pool_mutex);
        while (pq_instance->task_queue.empty() && !pq_instance->pool_shutdown_flag) {
            pthread_cond_wait(&pq_instance->pool_task_available_cond, &pq_instance->pool_mutex);
        }

        if (pq_instance->pool_shutdown_flag && pq_instance->task_queue.empty()) {
            pthread_mutex_unlock(&pq_instance->pool_mutex);
            pthread_exit(NULL);
        }

        if (!pq_instance->task_queue.empty()) {
            task_args = pq_instance->task_queue.front();
            pq_instance->task_queue.pop();
        }
        pthread_mutex_unlock(&pq_instance->pool_mutex);

        if (task_args) {
            // Execute the task
            process_segment_range(task_args); // This function is defined elsewhere and populates task_args->thread_local_guesses

            // Notify that a task is done
            pthread_mutex_lock(&pq_instance->pool_mutex);
            pq_instance->pool_pending_tasks_count--;
            if (pq_instance->pool_pending_tasks_count == 0) {
                pthread_cond_signal(&pq_instance->pool_tasks_all_done_cond);
            }
            pthread_mutex_unlock(&pq_instance->pool_mutex);
        }
    }
    return NULL;
}

void PriorityQueue::init_thread_pool(int num_threads) {
    if (num_threads <= 0) num_threads = 1; // Ensure at least one thread

    pthread_mutex_init(&pool_mutex, NULL);
    pthread_cond_init(&pool_task_available_cond, NULL);
    pthread_cond_init(&pool_tasks_all_done_cond, NULL);
    pool_shutdown_flag = false;
    pool_pending_tasks_count = 0;

    pool_threads.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        if (pthread_create(&pool_threads[i], NULL, PriorityQueue::worker_thread_function, this) != 0) {
            perror("Error creating worker thread");
            // Consider more robust error handling: stop already created threads, throw exception, etc.
            pool_threads.resize(i); // Record only successfully created threads
            shutdown_thread_pool(); // Attempt to clean up
            throw std::runtime_error("Failed to create worker thread pool");
        }
    }
}

void PriorityQueue::shutdown_thread_pool() {
    pthread_mutex_lock(&pool_mutex);
    pool_shutdown_flag = true;
    pthread_mutex_unlock(&pool_mutex);

    pthread_cond_broadcast(&pool_task_available_cond); // Wake up all waiting threads

    for (size_t i = 0; i < pool_threads.size(); ++i) {
        if (pool_threads[i] != 0) { // Check if thread was successfully created
             pthread_join(pool_threads[i], NULL);
        }
    }
    pool_threads.clear();

    pthread_mutex_destroy(&pool_mutex);
    pthread_cond_destroy(&pool_task_available_cond);
    pthread_cond_destroy(&pool_tasks_all_done_cond);
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// 该函数接收一个 PT 对象 pt（按值传递，意味着函数操作的是 pt 的一个副本）。
// 这个函数是 PCFG 并行化算法的主要载体，主要负责根据一个 PT 生成具体的密码猜测。
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);
    
    segment *target_segment_data = nullptr;
    string current_prefix_str = "";
    int num_values_to_process = 0;

    if (pt.content.size() == 1)
    {
        if (pt.content[0].type == 1) target_segment_data = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) target_segment_data = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3) target_segment_data = &m.symbols[m.FindSymbol(pt.content[0])];
        
        if (!pt.max_indices.empty()) {
            num_values_to_process = pt.max_indices[0];
        } else if (target_segment_data) {
            num_values_to_process = target_segment_data->ordered_values.size();
        }
    }
    else // pt.content.size() > 1
    {
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            // Basic bounds check for pt.content and its type mapping
            if (seg_idx >= pt.content.size()) break; // Should not happen with valid PT

            const segment& current_seg_template = pt.content[seg_idx];
            const vector<string>* values_source = nullptr;
            const segment* concrete_segment_data = nullptr;

            if (current_seg_template.type == 1) concrete_segment_data = &m.letters[m.FindLetter(current_seg_template)];
            else if (current_seg_template.type == 2) concrete_segment_data = &m.digits[m.FindDigit(current_seg_template)];
            else if (current_seg_template.type == 3) concrete_segment_data = &m.symbols[m.FindSymbol(current_seg_template)];

            if (concrete_segment_data && idx < concrete_segment_data->ordered_values.size()) {
                current_prefix_str += concrete_segment_data->ordered_values[idx];
            } else {
                // Error: index out of bounds for segment values, PT might be malformed or data inconsistent
                // For robustness, one might log this and skip this PT.
                return; 
            }
            
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) break;
        }

        int last_seg_idx = pt.content.size() - 1;
        if (last_seg_idx < 0) return; // Should not happen



        const segment& last_seg_template = pt.content[last_seg_idx];
        if (last_seg_template.type == 1) target_segment_data = &m.letters[m.FindLetter(last_seg_template)];
        else if (last_seg_template.type == 2) target_segment_data = &m.digits[m.FindDigit(last_seg_template)];
        else if (last_seg_template.type == 3) target_segment_data = &m.symbols[m.FindSymbol(last_seg_template)];
        
        if (last_seg_idx < pt.max_indices.size()) {
            num_values_to_process = pt.max_indices[last_seg_idx];
        } else if (target_segment_data) {
            num_values_to_process = target_segment_data->ordered_values.size();
        }
    }

    if (!target_segment_data || num_values_to_process == 0) {
        return; // Nothing to generate
    }

    const int PARALLEL_EXECUTION_THRESHOLD = 1000; // 示例阈值

    // Fallback to serial execution if thread pool is not available/initialized or for very small tasks
    if (pool_threads.empty() || num_values_to_process < 2 ) { // Simple heuristic for serial
        for (int i = 0; i < num_values_to_process; ++i) {
            if (i < target_segment_data->ordered_values.size()) { // Bounds check
                string final_guess = current_prefix_str + target_segment_data->ordered_values[i];
                guesses.emplace_back(final_guess);
            } else {
                break; // Index out of bounds
            }
        }
        return;
    }
if (pool_threads.empty() || num_values_to_process < PARALLEL_EXECUTION_THRESHOLD ) {
        // 串行执行路径
        // (如果需要，可以为串行路径预分配guesses空间以提高效率，但此处保持与原逻辑一致)
        for (int i = 0; i < num_values_to_process; ++i) {
            if (i < target_segment_data->ordered_values.size()) { // Bounds check
                string final_guess = current_prefix_str + target_segment_data->ordered_values[i];
                guesses.emplace_back(final_guess);
            } else {
                break; // Index out of bounds
            }
        }
        return; // 串行执行完毕，直接返回
    }

    int num_actual_threads_to_use = std::min((int)pool_threads.size(), num_values_to_process);
    if (num_actual_threads_to_use <=0) num_actual_threads_to_use = 1; // Ensure at least one if num_values_to_process > 0

    std::vector<ThreadGenerateArgs> local_task_args_list(num_actual_threads_to_use);
    int items_per_thread = num_values_to_process / num_actual_threads_to_use;
    int remaining_items = num_values_to_process % num_actual_threads_to_use;
    int current_start_index = 0;
    int tasks_created = 0;

    pthread_mutex_lock(&pool_mutex);
    pool_pending_tasks_count = 0; // Reset for this Generate call's batch of tasks
    pthread_mutex_unlock(&pool_mutex);

    for (int i = 0; i < num_actual_threads_to_use; ++i) {
        int items_for_this_thread_task = items_per_thread + (i < remaining_items ? 1 : 0);

        if (items_for_this_thread_task == 0) {
            continue; // No work for this thread slot
        }
        if (current_start_index >= num_values_to_process) {
            break; // All work assigned
        }
        
        local_task_args_list[tasks_created].segment_data = target_segment_data;
        local_task_args_list[tasks_created].start_index = current_start_index;
        local_task_args_list[tasks_created].end_index = current_start_index + items_for_this_thread_task;
        local_task_args_list[tasks_created].prefix_str = current_prefix_str;
        local_task_args_list[tasks_created].thread_local_guesses.clear(); 
        local_task_args_list[tasks_created].thread_local_guess_count = 0;

        pthread_mutex_lock(&pool_mutex);
        task_queue.push(&local_task_args_list[tasks_created]);
        pool_pending_tasks_count++;
        pthread_cond_signal(&pool_task_available_cond);
        pthread_mutex_unlock(&pool_mutex);

        current_start_index += items_for_this_thread_task;
        tasks_created++;
    }
    
    if (tasks_created > 0) {
        pthread_mutex_lock(&pool_mutex);
        while (pool_pending_tasks_count > 0) {
            pthread_cond_wait(&pool_tasks_all_done_cond, &pool_mutex);
        }
        pthread_mutex_unlock(&pool_mutex);
    }

    // Collect results
    for (int i = 0; i < tasks_created; ++i) {
        guesses.insert(guesses.end(), 
                       local_task_args_list[i].thread_local_guesses.begin(), 
                       local_task_args_list[i].thread_local_guesses.end());
        // total_guesses is updated in correctness_guess.cpp based on guesses.size()
    }
}