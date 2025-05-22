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
    priority.clear(); 
    for (const PT& pt_template_const : m.ordered_pts)
    {
        PT pt = pt_template_const; 
        pt.max_indices.clear();
        pt.curr_indices.clear();

        for (size_t i = 0; i < pt.content.size(); ++i)
        {
            const segment& seg_in_pt = pt.content[i]; 
            const segment* model_seg_ptr = nullptr;   

            if (seg_in_pt.type == 1) {
                int model_idx = m.FindLetter(seg_in_pt);
                if (model_idx != -1) model_seg_ptr = &m.letters[model_idx];
            } else if (seg_in_pt.type == 2) {
                int model_idx = m.FindDigit(seg_in_pt);
                if (model_idx != -1) model_seg_ptr = &m.digits[model_idx];
            } else if (seg_in_pt.type == 3) {
                int model_idx = m.FindSymbol(seg_in_pt);
                if (model_idx != -1) model_seg_ptr = &m.symbols[model_idx];
            }

            if (model_seg_ptr) {
                // max_indices 存储模型中该 segment 的 ordered_values 的实际数量
                pt.max_indices.emplace_back(model_seg_ptr->ordered_values.size());
            } else {
                pt.max_indices.emplace_back(0); 
            }
            
            if (i < pt.content.size() - 1) {
                pt.curr_indices.emplace_back(0);
            }
        }
        
        CalProb(pt); 
        
        // 保持优先队列有序（按概率降序）
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); ++iter) {
            if (pt.prob > iter->prob) { 
                priority.insert(iter, pt);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            priority.emplace_back(pt);
        }
    }
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


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    if (pt.content.empty() || pt.max_indices.empty()) return;

    int num_threads_to_target = 32; // 例如，一个 warp 的大小

    if (pt.content.size() == 1)
    {
        const segment& single_seg_template = pt.content[0];
        segment* model_segment_ptr = nullptr; // 使用非 const 指针，因为 segment 成员不是都为 const

        if (single_seg_template.type == 1) {
            int model_idx = m.FindLetter(single_seg_template);
            if (model_idx != -1) model_segment_ptr = &m.letters[model_idx];
        } else if (single_seg_template.type == 2) {
            int model_idx = m.FindDigit(single_seg_template);
            if (model_idx != -1) model_segment_ptr = &m.digits[model_idx];
        } else if (single_seg_template.type == 3) {
            int model_idx = m.FindSymbol(single_seg_template);
            if (model_idx != -1) model_segment_ptr = &m.symbols[model_idx];
        }

        if (!model_segment_ptr || model_segment_ptr->ordered_values.empty()) return;

        size_t actual_values_count = model_segment_ptr->ordered_values.size();
        if (actual_values_count == 0) return;

        size_t iterations_for_loop = actual_values_count;
        bool repeat_values = false;
        // 如果实际值数量小于目标线程数，并且我们希望每个线程都有工作（通过重复值）
        if (actual_values_count < num_threads_to_target) {
             iterations_for_loop = num_threads_to_target; // 循环这么多次以生成足够猜测
             repeat_values = true;
        }
        
        #pragma omp parallel
        {
            vector<string> thread_local_guesses;
            #pragma omp for schedule(static)
            for (int i = 0; i < iterations_for_loop; ++i) {
                string guess;
                if (repeat_values) {
                    guess = model_segment_ptr->ordered_values[i % actual_values_count];
                } else {
                    // iterations_for_loop == actual_values_count
                    guess = model_segment_ptr->ordered_values[i];
                }
                thread_local_guesses.emplace_back(guess);
            }
            
            #pragma omp critical
            {
                guesses.insert(guesses.end(), make_move_iterator(thread_local_guesses.begin()), make_move_iterator(thread_local_guesses.end()));
            }
        }
    }
    else // pt.content.size() > 1
    {
        string prefix_guess;
        int seg_idx = 0;
        for (int val_idx : pt.curr_indices) {
            if (seg_idx >= pt.content.size() -1) break; // curr_indices 是为前缀服务的

            const segment& current_seg_template = pt.content[seg_idx];
            const segment* model_seg_ptr = nullptr;

            if (current_seg_template.type == 1) {
                int model_idx = m.FindLetter(current_seg_template);
                if (model_idx != -1) model_seg_ptr = &m.letters[model_idx];
            } else if (current_seg_template.type == 2) {
                int model_idx = m.FindDigit(current_seg_template);
                if (model_idx != -1) model_seg_ptr = &m.digits[model_idx];
            } else if (current_seg_template.type == 3) {
                int model_idx = m.FindSymbol(current_seg_template);
                if (model_idx != -1) model_seg_ptr = &m.symbols[model_idx];
            }
            
            if (model_seg_ptr && val_idx < model_seg_ptr->ordered_values.size()) {
                prefix_guess += model_seg_ptr->ordered_values[val_idx];
            } else {
                // 不一致的状态，提前返回或记录错误
                return; 
            }
            seg_idx++;
        }

        const segment& last_seg_template = pt.content.back();
        segment* final_model_segment_ptr = nullptr; // 非 const 指针

        if (last_seg_template.type == 1) {
            int model_idx = m.FindLetter(last_seg_template);
            if (model_idx != -1) final_model_segment_ptr = &m.letters[model_idx];
        } else if (last_seg_template.type == 2) {
            int model_idx = m.FindDigit(last_seg_template);
            if (model_idx != -1) final_model_segment_ptr = &m.digits[model_idx];
        } else if (last_seg_template.type == 3) {
            int model_idx = m.FindSymbol(last_seg_template);
            if (model_idx != -1) final_model_segment_ptr = &m.symbols[model_idx];
        }

        if (!final_model_segment_ptr || final_model_segment_ptr->ordered_values.empty()) return;
        
        size_t last_segment_actual_values_count = final_model_segment_ptr->ordered_values.size();
        if (last_segment_actual_values_count == 0) return;

        #pragma omp parallel
        {
            vector<string> thread_local_guesses;
            #pragma omp for schedule(static)
            for (int i = 0; i < last_segment_actual_values_count; ++i) {
                string temp_guess = prefix_guess + final_model_segment_ptr->ordered_values[i];
                thread_local_guesses.emplace_back(temp_guess);
            }
            
            #pragma omp critical
            {
                guesses.insert(guesses.end(), make_move_iterator(thread_local_guesses.begin()), make_move_iterator(thread_local_guesses.end()));
            }
        }
    }
}