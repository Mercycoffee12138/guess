#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <cmath>

// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 * 
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 * 
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 * 
 */

// 训练的wrapper，实际上就是读取训练集
void model::train(string path)
{
    string pw;
    ifstream train_set(path);
    int lines = 0;
    cout<<"Training..."<<endl;
    cout<<"Training phase 1: reading and parsing passwords..."<<endl;
    while (train_set >> pw)
    {
        lines += 1;
        if (lines % 10000 == 0)
        {
            cout <<"Lines processed: "<< lines << endl;
            // 在这里更改读取的训练集口令上限
            if (lines > 3000000)
            {
                break;
            }
        }
        // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
        parse(pw);
    }
}

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(PT pt)
{
    for (int id = 0; id < preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }
        else
        {
            bool equal_flag = true;
            for (int idx = 0; idx < preterminals[id].content.size(); idx += 1)
            {
                if (preterminals[id].content[idx].type != pt.content[idx].type || preterminals[id].content[idx].length != pt.content[idx].length)
                {
                    equal_flag = false;
                    break;
                }
            }
            if (equal_flag == true)
            {
                return id;
            }
        }
    }
    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(segment seg)
{
    for (int id = 0; id < letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(segment seg)
{
    for (int id = 0; id < digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}


void segment::order(int n_smoothing_exponent)
{
    ordered_values.clear();
    for (pair<string, int> value_pair : values) // 使用 value_pair 避免与成员 values 混淆
    {
        ordered_values.emplace_back(value_pair.first);
    }

    if (ordered_values.empty()) {
        total_freq = 0;
        effective_container_size = 0;
        ordered_freqs.clear();
        return;
    }

    std::sort(ordered_values.begin(), ordered_values.end(),
              [this](const std::string &a, const std::string &b)
              {
                  // 确保 freqs 和 values 包含对应的键
                  auto it_a = values.find(a);
                  auto it_b = values.find(b);
                  if (it_a == values.end() || it_b == values.end()) {
                      // 处理错误或返回默认排序
                      return false; 
                  }
                  auto freq_it_a = freqs.find(it_a->second);
                  auto freq_it_b = freqs.find(it_b->second);
                  if (freq_it_a == freqs.end() || freq_it_b == freqs.end()) {
                      // 处理错误或返回默认排序
                      return false;
                  }
                  return freq_it_a->second > freq_it_b->second;
              });

    ordered_freqs.clear();
    total_freq = 0; // total_freq 将是原始频率的总和

    for (const std::string &val : ordered_values)
    {
        // ordered_freqs 初始填充原始频率，后续可能被平滑后的频率替换
        ordered_freqs.emplace_back(freqs.at(values.at(val)));
        total_freq += freqs.at(values.at(val));
    }
    // 移除之前代码中的重复循环

    if (n_smoothing_exponent > 0 && !ordered_freqs.empty())
    {
        int group_size = 1 << n_smoothing_exponent;
        this->effective_container_size = group_size;

        std::vector<int> smoothed_freqs;
        smoothed_freqs.reserve(ordered_freqs.size());
        
        for (size_t i = 0; i < ordered_freqs.size(); i += group_size)
        {
            long long current_group_freq_sum = 0;
            size_t current_group_actual_size = 0;
            for (size_t j = 0; j < group_size && (i + j) < ordered_freqs.size(); ++j)
            {
                current_group_freq_sum += ordered_freqs[i + j];
                current_group_actual_size++;
            }

            if (current_group_actual_size > 0)
            {
                // 论文提到“重新分配概率为组的平均概率”
                // 这里我们存储平均频率；概率计算时 P = smoothed_freq / original_total_freq
                int average_freq = static_cast<int>(round(static_cast<double>(current_group_freq_sum) / current_group_actual_size));
                for (size_t j = 0; j < current_group_actual_size; ++j)
                {
                    // smoothed_freqs.push_back(average_freq); // 这会改变原始排序的频率对应关系
                    // 应该直接修改 ordered_freqs 中对应位置的值
                    if ((i + j) < ordered_freqs.size()) { // 确保不越界
                        ordered_freqs[i+j] = average_freq;
                    }
                }
            }
        }
        // ordered_freqs 现在包含了平滑后的频率
        // total_freq 保持为原始总频率，用于概率计算 pt.prob /= segment.total_freq
    } else if (!ordered_freqs.empty()) {
        this->effective_container_size = ordered_values.size(); 
    } else {
        this->effective_container_size = 0;
    }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }
        int id = GetNextPretermID();
        // cout << id << endl;
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        // cout << id << endl;
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT& a, const PT& b) {
    return a.preterm_prob > b.preterm_prob;  // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT& pt : preterminals) // 使用引用以修改原始数据
    {
        // FindPT 可能需要 const PT&，如果 preterminals 是 const vector<PT>
        // 或者 FindPT(const_cast<const PT&>(pt))
        // 假设 FindPT 可以接受 PT& 或者 preterminals 是 vector<PT>
        int pt_idx = FindPT(pt);
        if (pt_idx != -1 && preterm_freq.count(pt_idx) && total_preterm > 0) {
             pt.preterm_prob = static_cast<float>(preterm_freq[pt_idx]) / total_preterm;
        } else {
             pt.preterm_prob = 0.0f;
        }
        ordered_pts.emplace_back(pt);
    }
    // bool swapped; // 未使用
    cout << "total pts before sort: " << ordered_pts.size() << endl; // 修改日志
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    
    int smoothing_n = 3; // 例如 n=3, 组大小为 8. 可以设为可配置参数
    // 如果 smoothing_n = 0, 则不进行平滑

    cout << "Ordering letters" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].order(smoothing_n);
    }
    cout << "Ordering digits" << endl;
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].order(smoothing_n);
    }
    cout << "ordering symbols" << endl;
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].order(smoothing_n);
    }
    cout << "Training phase 2 finished." << endl; // 添加结束日志
}