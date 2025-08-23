// 3. 无重复字符的最长子串
pub mod n3 {
    pub fn length_of_longest_substring(s: String) -> i32 {
        let s = s.as_bytes();
        let mut ans = 0;
        let mut left = 0;
        let mut window = [false; 128]; // 也可以用哈希集合，这里为了效率用的数组

        for (right, &c) in s.iter().enumerate() {
            let c = c as usize;

            // 如果窗口内已经包含 c，那么再加入一个 c 会导致窗口内有重复元素
            // 所以要在加入 c 之前，先移出窗口内的 c
            while window[c] {
                // 窗口内有 c
                window[s[left] as usize] = false;
                left += 1; // 缩小窗口
            }

            window[c] = true; // 加入 c

            ans = ans.max(right - left + 1); // 更新窗口长度最大值
        }

        ans as _
    }
}

// 3090. 每个字符最多出现两次的最长子字符串
pub mod n3090 {
    use std::collections::HashMap;

    pub fn maximum_length_substring(s: String) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();
        let chars: Vec<char> = s.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            // 增加当前字符计数
            *cnt.entry(c).or_insert(0) += 1;

            // 如果当前字符出现次数超过2，移动左指针
            while *cnt.get(&c).unwrap() > 2 {
                let left_char = chars[left];
                *cnt.get_mut(&left_char).unwrap() -= 1;
                left += 1;
            }

            // 更新最大长度
            ans = ans.max(i - left + 1);
        }

        ans as i32
    }
}

// 1493. 删掉一个元素以后全为 1 的最长子数组
pub mod n1493 {
    pub fn longest_subarray(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut cnt0 = 0;
        let mut left = 0;
        for (right, &x) in nums.iter().enumerate() {
            // 1. 入，nums[right] 进入窗口
            cnt0 += 1 - x; // 维护窗口中的 0 的个数
            while cnt0 > 1 {
                // 2. 出，nums[left] 离开窗口
                cnt0 -= 1 - nums[left]; // 维护窗口中的 0 的个数
                left += 1;
            }
            // 3. 更新答案，注意不是 right-left+1，因为我们要删掉一个数
            ans = ans.max(right - left);
        }
        ans as _
    }
}

// 3634. 使数组平衡的最少移除数目
pub mod n3634 {
    pub fn min_removal(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable();
        let mut max_save = 0;
        let mut left = 0;
        let k = k as f64;

        for (i, &mx) in nums.iter().enumerate() {
            while nums[left] as f64 * k < mx as f64 {
                left += 1;
            }
            max_save = max_save.max(i - left + 1);
        }

        (nums.len() - max_save) as i32
    }
}

// 1208. 尽可能使字符串相等
pub mod n1208 {
    pub fn equal_substring(s: String, t: String, max_cost: i32) -> i32 {
        let s_bytes = s.as_bytes();
        let t_bytes = t.as_bytes();
        let n = s_bytes.len();

        // 计算每个位置的字节差值绝对值
        let diff: Vec<i32> = s_bytes
            .iter()
            .zip(t_bytes.iter())
            .map(|(sc, tc)| (*sc as i32 - *tc as i32).abs())
            .collect();

        let mut max_length = 0;
        let mut start = 0;
        let mut total = 0;

        // 滑动窗口逻辑
        for end in 0..n {
            total += diff[end];

            // 超出最大成本时移动左指针
            while total > max_cost {
                total -= diff[start];
                start += 1;
            }

            // 更新最大长度
            max_length = max_length.max((end - start + 1) as i32);
        }

        max_length
    }
}

// 904. 水果成篮
pub mod n904 {
    use std::collections::HashMap;

    pub fn total_fruit(fruits: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();
        for (right, &x) in fruits.iter().enumerate() {
            *cnt.entry(x).or_insert(0) += 1; // fruits[right] 进入窗口
            while cnt.len() > 2 {
                // 不满足要求
                let out = fruits[left];
                *cnt.entry(out).or_insert(0) -= 1; // fruits[left] 离开窗口
                if cnt[&out] == 0 {
                    cnt.remove(&out);
                }
                left += 1;
            }
            ans = ans.max(right - left + 1);
        }
        ans as _
    }
}

// 1695. 删除子数组的最大得分
pub mod n1695 {
    pub fn maximum_unique_subarray(nums: Vec<i32>) -> i32 {
        let mx = *nums.iter().max().unwrap();
        let mut has = vec![false; (mx + 1) as usize];
        let mut ans = 0;
        let mut sum = 0;
        let mut left = 0;
        for &x in &nums {
            while has[x as usize] {
                has[nums[left] as usize] = false;
                sum -= nums[left];
                left += 1;
            }
            has[x as usize] = true;
            sum += x;
            ans = ans.max(sum);
        }
        ans
    }
}

// 2958. 最多 K 个重复元素的最长子数组
pub mod n2958 {
    use std::collections::HashMap;

    pub fn max_subarray_length(nums: Vec<i32>, k: i32) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();

        for (right, &x) in nums.iter().enumerate() {
            // 增加当前元素的计数
            *cnt.entry(x).or_insert(0) += 1;

            // 当当前元素的计数超过k时，移动左指针
            while *cnt.get(&x).unwrap() > k {
                let left_num = nums[left];
                *cnt.get_mut(&left_num).unwrap() -= 1;
                left += 1;
            }

            // 更新最大长度
            ans = ans.max((right - left + 1) as i32);
        }

        ans
    }
}

// 2024. 考试的最大困扰度
pub mod n2024 {
    pub fn max_consecutive_answers(answer_key: String, k: i32) -> i32 {
        let s = answer_key.as_bytes();
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = [0, 0];
        for (right, &ch) in s.iter().enumerate() {
            cnt[(ch >> 1 & 1) as usize] += 1;
            while cnt[0] > k && cnt[1] > k {
                cnt[(s[left] >> 1 & 1) as usize] -= 1;
                left += 1;
            }
            ans = ans.max(right - left + 1);
        }
        ans as _
    }
}

// 1004. 最大连续1的个数 III
pub mod n1004 {
    pub fn longest_ones(nums: Vec<i32>, k: i32) -> i32 {
        let mut max_len = 0;
        let mut left = 0;
        let mut zero_count = 0;

        for (right, &num) in nums.iter().enumerate() {
            // 统计窗口中0的数量（1 - num：0转为1，1转为0）
            zero_count += 1 - num;

            // 当0的数量超过k时，移动左指针缩小窗口
            while zero_count > k {
                zero_count -= 1 - nums[left];
                left += 1;
            }

            // 更新当前窗口的最大长度
            max_len = max_len.max((right - left + 1) as i32);
        }

        max_len
    }
}

// 1658. 将 x 减到 0 的最小操作数
pub mod n1658 {
    pub fn min_operations(nums: Vec<i32>, x: i32) -> i32 {
        let total: i32 = nums.iter().sum();
        let target = total - x;

        if target < 0 {
            return -1;
        }

        let mut ans = -1;
        let mut s = 0;
        let mut left = 0;

        for (right, &num) in nums.iter().enumerate() {
            s += num;

            // 当当前和超过目标时，移动左指针缩小窗口
            while s > target {
                s -= nums[left];
                left += 1;
            }

            // 找到符合条件的子数组，更新最大长度
            if s == target {
                ans = ans.max((right - left + 1) as i32);
            }
        }

        if ans < 0 { -1 } else { nums.len() as i32 - ans }
    }
}

// 209. 长度最小的子数组
pub mod n209 {
    pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut ans = n + 1;
        let mut sum = 0; // 子数组元素和
        let mut left = 0; // 子数组左端点
        for (right, &x) in nums.iter().enumerate() {
            // 枚举子数组右端点
            sum += x;
            while sum >= target {
                // 满足要求
                ans = ans.min(right - left + 1);
                sum -= nums[left];
                left += 1; // 左端点右移
            }
        }
        if ans <= n { ans as i32 } else { 0 }
    }
}

// 2904. 最短且字典序最小的美丽子字符串
pub mod n2904 {
    pub fn shortest_beautiful_substring(s: String, k: i32) -> String {
        // 统计字符串中'1'的总数，若不足k则直接返回空字符串
        let count_ones = s.chars().filter(|&c| c == '1').count() as i32;
        if count_ones < k {
            return String::new();
        }

        let s_chars: Vec<char> = s.chars().collect();
        let mut ans = s.clone();
        let mut cnt1 = 0;
        let mut left = 0;

        for right in 0..s_chars.len() {
            // 累加当前位置的'1'（若为'1'则加1，'0'则加0）
            cnt1 += s_chars[right].to_digit(10).unwrap() as i32;

            // 收缩左指针：当1的数量超过k，或左指针指向'0'时（可优化窗口）
            while cnt1 > k || s_chars[left] == '0' {
                cnt1 -= s_chars[left].to_digit(10).unwrap() as i32;
                left += 1;
            }

            // 当窗口内1的数量恰好为k时，更新答案
            if cnt1 == k {
                let current = &s[left..=right];
                // 优先选长度更短的，长度相同则选字典序更小的
                if current.len() < ans.len() || (current.len() == ans.len() && current < &ans) {
                    ans = current.to_string();
                }
            }
        }

        ans
    }
}

// 1234. 替换子串得到平衡字符串
pub mod n1234 {
    use std::collections::HashMap;

    pub fn balanced_string(s: String) -> i32 {
        let m = s.len() / 4;
        let mut cnt: HashMap<char, usize> = HashMap::new();

        // 统计各字符出现次数
        for c in s.chars() {
            *cnt.entry(c).or_insert(0) += 1;
        }

        // 检查是否已平衡
        if cnt.len() == 4 {
            let min_val = *cnt.values().min().unwrap();
            if min_val == m {
                return 0;
            }
        }

        let s_chars: Vec<char> = s.chars().collect();
        let mut ans = usize::MAX;
        let mut left = 0;
        let mut current_cnt = cnt.clone();

        // 滑动窗口寻找最小替换子串
        for right in 0..s_chars.len() {
            let c = s_chars[right];
            *current_cnt.get_mut(&c).unwrap() -= 1;

            // 当窗口外字符均符合要求时，尝试缩小窗口
            while current_cnt.values().max().unwrap_or(&0) <= &m {
                ans = ans.min(right - left + 1);
                let left_c = s_chars[left];
                *current_cnt.get_mut(&left_c).unwrap() += 1;
                left += 1;
                if left > right {
                    break;
                }
            }
        }

        ans as i32
    }
}
