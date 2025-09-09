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

// 2875. 无限数组的最短子数组
pub mod n2875 {
    pub fn min_size_subarray(nums: Vec<i32>, target: i32) -> i32 {
        let target = target as i64;

        let total = nums.iter().map(|&x| x as i64).sum::<i64>();

        let n = nums.len();

        let mut ans = usize::MAX;

        let mut sum = 0;

        let mut left = 0;

        for right in 0..n * 2 {
            sum += nums[right % n];

            while sum > (target % total) as i32 {
                sum -= nums[left % n];

                left += 1;
            }

            if sum == (target % total) as i32 {
                ans = ans.min(right - left + 1);
            }
        }

        if ans < usize::MAX {
            ans as i32 + (target / total) as i32 * n as i32
        } else {
            -1
        }
    }
}

// 76. 最小覆盖子串
#[allow(non_snake_case)]
pub mod n76 {
    pub fn min_window(S: String, t: String) -> String {
        fn is_covered(cnt_s: &[i32; 128], cnt_t: &[i32; 128]) -> bool {
            for i in b'A'..=b'Z' {
                if cnt_s[i as usize] < cnt_t[i as usize] {
                    return false;
                }
            }
            for i in b'a'..=b'z' {
                if cnt_s[i as usize] < cnt_t[i as usize] {
                    return false;
                }
            }
            true
        }

        let mut cnt_s = [0; 128]; // s 子串字母的出现次数
        let mut cnt_t = [0; 128]; // t 中字母的出现次数
        for c in t.bytes() {
            cnt_t[c as usize] += 1;
        }

        let s = S.as_bytes();
        let m = s.len();
        let mut ans_left = 0;
        let mut ans_right = m;

        let mut left = 0;
        for (right, &c) in s.iter().enumerate() {
            // 移动子串右端点
            cnt_s[c as usize] += 1; // 右端点字母移入子串
            while is_covered(&cnt_s, &cnt_t) {
                // 涵盖
                if right - left < ans_right - ans_left {
                    // 找到更短的子串
                    ans_left = left; // 记录此时的左右端点
                    ans_right = right;
                }
                cnt_s[s[left] as usize] -= 1; // 左端点字母移出子串
                left += 1;
            }
        }

        if ans_right < m {
            S[ans_left..=ans_right].to_string()
        } else {
            String::new()
        }
    }

    pub fn min_window1(S: String, t: String) -> String {
        let mut cnt = [0; 128];
        let mut less = 0;
        for c in t.bytes() {
            let c = c as usize;
            if cnt[c] == 0 {
                less += 1; // 有 less 种字母的出现次数 < t 中的字母出现次数
            }
            cnt[c] += 1;
        }

        let s = S.as_bytes();
        let m = s.len();
        let mut ans_left = 0;
        let mut ans_right = m;

        let mut left = 0;
        for (right, &c) in s.iter().enumerate() {
            // 移动子串右端点
            let c = c as usize;
            cnt[c] -= 1; // 右端点字母移入子串
            if cnt[c] == 0 {
                // 原来窗口内 c 的出现次数比 t 的少，现在一样多
                less -= 1;
            }
            while less == 0 {
                // 涵盖：所有字母的出现次数都是 >=
                if right - left < ans_right - ans_left {
                    // 找到更短的子串
                    ans_left = left; // 记录此时的左右端点
                    ans_right = right;
                }
                let x = s[left] as usize; // 左端点字母
                if cnt[x] == 0 {
                    // x 移出窗口之前，检查出现次数，
                    // 如果窗口内 x 的出现次数和 t 一样，
                    // 那么 x 移出窗口后，窗口内 x 的出现次数比 t 的少
                    less += 1;
                }
                cnt[x] += 1; // 左端点字母移出子串
                left += 1;
            }
        }

        if ans_right < m {
            S[ans_left..=ans_right].to_string()
        } else {
            String::new()
        }
    }
}

// 632. 最小区间
pub mod n632 {
    pub fn smallest_range(nums: Vec<Vec<i32>>) -> Vec<i32> {
        let mut pairs = vec![];
        for (i, arr) in nums.iter().enumerate() {
            for &x in arr {
                pairs.push((x, i));
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut ans_l = pairs[0].0;
        let mut ans_r = pairs[pairs.len() - 1].0;
        let mut empty = nums.len();
        let mut cnt = vec![0; empty];
        let mut left = 0;
        for &(r, i) in &pairs {
            if cnt[i] == 0 {
                // 包含 nums[i] 的数字
                empty -= 1;
            }
            cnt[i] += 1;
            while empty == 0 {
                // 每个列表都至少包含一个数
                let (l, i) = pairs[left];
                if r - l < ans_r - ans_l {
                    ans_l = l;
                    ans_r = r;
                }
                cnt[i] -= 1;
                if cnt[i] == 0 {
                    // 不包含 nums[i] 的数字
                    empty += 1;
                }
                left += 1;
            }
        }
        vec![ans_l, ans_r]
    }
}

// 713. 乘积小于 K 的子数组
pub mod n713 {
    pub fn num_subarray_product_less_than_k(nums: Vec<i32>, k: i32) -> i32 {
        if k <= 1 {
            return 0;
        }
        let mut ans = 0;
        let mut prod = 1;
        let mut left = 0;
        for (right, &x) in nums.iter().enumerate() {
            prod *= x;
            while prod >= k {
                prod /= nums[left];
                left += 1; // 缩小窗口
            }
            // 对于固定的 right，有 right-left+1 个合法的左端点
            ans += right - left + 1;
        }
        ans as _
    }
}

// 3258. 统计满足 K 约束的子字符串数量 I
pub mod n3258 {
    pub fn count_k_constraint_substrings(s: String, k: i32) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = [0, 0]; // 分别统计0和1的出现次数
        let s_bytes = s.as_bytes(); // 转换为字节数组便于索引

        for (i, &c) in s_bytes.iter().enumerate() {
            // 计算当前字符是0还是1（利用ASCII码特性：'0'是48，'1'是49，&1后分别为0和1）
            let idx = (c & 1) as usize;
            cnt[idx] += 1;

            // 当0和1的数量都超过k时，移动左指针缩小窗口
            while cnt[0] > k && cnt[1] > k {
                let left_idx = (s_bytes[left] & 1) as usize;
                cnt[left_idx] -= 1;
                left += 1;
            }

            // 窗口[left, i]内的所有子串均满足条件，数量为i - left + 1
            ans += (i - left + 1) as i32;
        }

        ans
    }
}

// 2302. 统计得分小于 K 的子数组数目
pub mod n2302 {
    pub fn count_subarrays(nums: Vec<i32>, k: i64) -> i64 {
        let mut ans = 0;
        let mut sum = 0;
        let mut left = 0;
        for (right, &x) in nums.iter().enumerate() {
            sum += x as i64;
            while sum * ((right - left + 1) as i64) >= k {
                sum -= nums[left] as i64;
                left += 1;
            }
            ans += (right - left + 1) as i64;
        }
        ans
    }
}

// 2762. 不间断子数组
pub mod n2762 {
    use std::collections::{BTreeMap, HashMap};

    // 方法一：使用 BTreeMap 维护有序性，便于快速获取最大最小值
    pub fn continuous_subarrays(nums: Vec<i32>) -> i64 {
        let mut ans: i64 = 0;
        let mut left = 0;
        let mut cnt = BTreeMap::new(); // BTreeMap 会自动排序键，便于获取最大最小值

        for (right, &x) in nums.iter().enumerate() {
            *cnt.entry(x).or_insert(0) += 1;

            // 当最大值与最小值之差超过 2 时，移动左指针
            while {
                let first = *cnt.keys().next().unwrap();
                let last = *cnt.keys().next_back().unwrap();
                last - first > 2
            } {
                let y = nums[left];
                *cnt.get_mut(&y).unwrap() -= 1;
                if cnt[&y] == 0 {
                    cnt.remove(&y);
                }
                left += 1;
            }

            // 累加当前右指针位置下的有效子数组数量
            ans += (right - left + 1) as i64;
        }

        ans
    }

    // 方法二：使用 HashMap 配合手动一个变量手动跟踪最大最小值
    pub fn continuous_subarrays1(nums: Vec<i32>) -> i64 {
        let mut ans: i64 = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();
        let mut current_max = i32::MIN;
        let mut current_min = i32::MAX;

        for (right, &x) in nums.iter().enumerate() {
            *cnt.entry(x).or_insert(0) += 1;
            current_max = current_max.max(x);
            current_min = current_min.min(x);

            // 当最大值与最小值之差超过 2 时，移动左指针并更新最大最小值
            while current_max - current_min > 2 {
                let y = nums[left];
                *cnt.get_mut(&y).unwrap() -= 1;
                if cnt[&y] == 0 {
                    cnt.remove(&y);
                    // 如果移除的是当前最大或最小值，需要重新计算
                    if y == current_max {
                        current_max = *cnt.keys().max().unwrap_or(&i32::MIN);
                    }
                    if y == current_min {
                        current_min = *cnt.keys().min().unwrap_or(&i32::MAX);
                    }
                }
                left += 1;
            }

            ans += (right - left + 1) as i64;
        }

        ans
    }
}

// 1358. 包含所有三种字符的子字符串数目
pub mod n1358 {
    pub fn number_of_substrings(s: String) -> i32 {
        let s = s.as_bytes();
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = [0; 3];
        for &c in s {
            cnt[(c - b'a') as usize] += 1;
            while cnt[0] > 0 && cnt[1] > 0 && cnt[2] > 0 {
                cnt[(s[left] - b'a') as usize] -= 1;
                left += 1;
            }
            ans += left;
        }
        ans as _
    }
}

// 2962. 统计最大元素出现至少 K 次的子数组
pub mod n2962 {
    pub fn count_subarrays(nums: Vec<i32>, k: i32) -> i64 {
        let mx = *nums.iter().max().unwrap();
        let mut ans = 0;
        let mut cnt_mx = 0;
        let mut left = 0;
        for &x in &nums {
            if x == mx {
                cnt_mx += 1;
            }
            while cnt_mx == k {
                if nums[left] == mx {
                    cnt_mx -= 1;
                }
                left += 1;
            }
            ans += left;
        }
        ans as _
    }
}

// 3325. 字符至少出现 K 次的子字符串 I
pub mod n3325 {
    use std::collections::HashMap;

    pub fn number_of_substrings(s: String, k: i32) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();
        // 将字符串转为字符切片，便于遍历和索引访问
        let chars: Vec<char> = s.chars().collect();

        for c in &chars {
            // 计数当前字符出现次数
            *cnt.entry(*c).or_insert(0) += 1;

            // 当当前字符计数 >= k 时，移动左指针并更新计数
            while cnt[c] >= k {
                let left_char = chars[left];
                *cnt.get_mut(&left_char).unwrap() -= 1;
                left += 1;
            }

            // 累加当前左指针位置（代表符合条件的子串数量）
            ans += left as i32;
        }

        ans
    }
}

// 2799. 统计完全子数组的数目
pub mod n2799 {
    use std::collections::{HashMap, HashSet};

    pub fn count_complete_subarrays(nums: Vec<i32>) -> i32 {
        let k = nums.iter().collect::<HashSet<_>>().len();
        let mut cnt = HashMap::new();
        let mut ans = 0;
        let mut left = 0;
        for &x in &nums {
            *cnt.entry(x).or_insert(0) += 1;
            while cnt.len() == k {
                let out = nums[left];
                let e = cnt.get_mut(&out).unwrap();
                *e -= 1;
                if *e == 0 {
                    cnt.remove(&out);
                }
                left += 1;
            }
            ans += left;
        }
        ans as _
    }
}

// 2537. 统计好子数组的数目
pub mod n2537 {
    use std::collections::HashMap;

    pub fn count_good(nums: Vec<i32>, k: i32) -> i64 {
        let mut ans = 0;
        let mut cnt = HashMap::new();
        let mut pairs = 0;
        let mut left = 0;
        for &x in &nums {
            let e = cnt.entry(x).or_insert(0);
            pairs += *e;
            *e += 1;
            while pairs >= k {
                let e = cnt.get_mut(&nums[left]).unwrap();
                *e -= 1;
                pairs -= *e;
                left += 1;
            }
            ans += left;
        }
        ans as _
    }
}

// 3298. 统计重新排列后包含另一个字符串的子字符串数目 II
pub mod n3298 {
    pub fn valid_substring_count(s: String, t: String) -> i64 {
        if s.len() < t.len() {
            return 0;
        }

        let mut diff = vec![0; 26]; // t 的字母出现次数与 s 的字母出现次数之差
        for c in t.bytes() {
            diff[(c - b'a') as usize] += 1;
        }

        // 统计窗口内有多少个字母的出现次数比 t 的少
        let mut less = diff.iter().filter(|&&d| d > 0).count() as i32;

        let mut ans = 0;
        let mut left = 0;
        let s = s.as_bytes();
        for c in s {
            let c = (c - b'a') as usize;
            diff[c] -= 1;
            if diff[c] == 0 {
                // c 移入窗口后，窗口内 c 的出现次数和 t 的一样
                less -= 1;
            }
            while less == 0 {
                // 窗口符合要求
                let out_char = (s[left] - b'a') as usize; // 准备移出窗口的字母
                if diff[out_char] == 0 {
                    // out_char 移出窗口之前，检查出现次数，
                    // 如果窗口内 out_char 的出现次数和 t 的一样，
                    // 那么 out_char 移出窗口后，窗口内 out_char 的出现次数比 t 的少
                    less += 1;
                }
                diff[out_char] += 1;
                left += 1;
            }
            ans += left;
        }
        ans as _
    }
}

// 930. 和相同的二元子数组
pub mod n930 {
    pub fn num_subarrays_with_sum(nums: Vec<i32>, goal: i32) -> i32 {
        let mut l1 = 0;
        let mut l2 = 0;
        let mut res = 0;
        let mut tmp_sum_1 = 0;
        let mut tmp_sum_2 = 0;

        for (r, &num) in nums.iter().enumerate() {
            tmp_sum_1 += num;
            tmp_sum_2 += num;

            // 计算 sum >= goal 的左边界
            while l1 <= r && tmp_sum_1 >= goal {
                tmp_sum_1 -= nums[l1];
                l1 += 1;
            }

            // 计算 sum >= goal+1 的左边界
            while l2 <= r && tmp_sum_2 >= goal + 1 {
                tmp_sum_2 -= nums[l2];
                l2 += 1;
            }

            // 符合条件的子数组数量为两者差值
            res += (l1 - l2) as i32;
        }

        res
    }
}

// 1248. 统计「优美子数组」
pub mod n1248 {
    pub fn number_of_subarrays(nums: Vec<i32>, k: i32) -> i32 {
        let mut cnt1 = 0; // 用于统计大于等于k个奇数的窗口
        let mut cnt2 = 0; // 用于统计大于等于k+1个奇数的窗口
        let mut ans = 0;
        let mut left1 = 0;
        let mut left2 = 0;

        for (right, &num) in nums.iter().enumerate() {
            // 更新奇数计数（判断是否为奇数）
            cnt1 += (num & 1) as i32;
            cnt2 += (num & 1) as i32;

            // 计算 cnt >= k 的左边界
            while left1 <= right && cnt1 >= k {
                cnt1 -= (nums[left1] & 1) as i32;
                left1 += 1;
            }

            // 计算 cnt >= k + 1 的左边界
            while left2 <= right && cnt2 >= k + 1 {
                cnt2 -= (nums[left2] & 1) as i32;
                left2 += 1;
            }

            // 两者差值即为以right为结尾的、恰好包含k个奇数的子数组数量
            ans += (left1 - left2) as i32;
        }

        ans
    }
}

// 3306. 元音辅音字符串计数 II
pub mod n3306 {
    use std::collections::HashMap;

    pub fn count_of_substrings(word: String, k: i32) -> i64 {
        // 两个元音计数器
        let mut cnt_vowel1: HashMap<char, i32> = HashMap::new();
        let mut cnt_vowel2: HashMap<char, i32> = HashMap::new();
        // 两个辅音计数器（
        let mut cnt_consonant1 = 0;
        let mut cnt_consonant2 = 0;
        let mut ans = 0;
        let mut left1 = 0;
        let mut left2 = 0;
        // 将字符串转为字符数组，方便按索引访问
        let chars: Vec<char> = word.chars().collect();

        for (right, &c) in chars.iter().enumerate() {
            // 1. 统计当前字符（元音/辅音）
            if ['a', 'e', 'i', 'o', 'u'].contains(&c) {
                *cnt_vowel1.entry(c).or_insert(0) += 1;
                *cnt_vowel2.entry(c).or_insert(0) += 1;
            } else {
                cnt_consonant1 += 1;
                cnt_consonant2 += 1;
            }

            // 2. 调整 left1：确保窗口 [left1, right] 满足「元音全5种 + 辅音 ≥k」
            while left1 <= right && cnt_vowel1.len() == 5 && cnt_consonant1 >= k {
                let out = chars[left1];
                // 移除窗口左边界字符的统计
                if ['a', 'e', 'i', 'o', 'u'].contains(&out) {
                    *cnt_vowel1.get_mut(&out).unwrap() -= 1;
                    // 若该元音计数为0，从哈希表中删除（保证 len 准确）
                    if cnt_vowel1[&out] == 0 {
                        cnt_vowel1.remove(&out);
                    }
                } else {
                    cnt_consonant1 -= 1;
                }
                left1 += 1;
            }

            // 3. 调整 left2：确保窗口 [left2, right] 满足「元音全5种 + 辅音 >k」
            while left2 <= right && cnt_vowel2.len() == 5 && cnt_consonant2 > k {
                let out = chars[left2];
                // 移除窗口左边界字符的统计
                if ['a', 'e', 'i', 'o', 'u'].contains(&out) {
                    *cnt_vowel2.get_mut(&out).unwrap() -= 1;
                    if cnt_vowel2[&out] == 0 {
                        cnt_vowel2.remove(&out);
                    }
                } else {
                    cnt_consonant2 -= 1;
                }
                left2 += 1;
            }

            // 4. 两者差值即为「元音全5种 + 辅音恰好k个」的子数组数量
            ans += (left1 - left2) as i64;
        }

        ans
    }
}

// 992. K 个不同整数的子数组
pub mod n992 {
    use std::collections::HashMap;

    pub fn subarrays_with_k_distinct(nums: Vec<i32>, k: i32) -> i32 {
        let k = k as usize;
        let mut cnt1 = HashMap::new(); // 用于统计大于等于k个不同整数的窗口
        let mut cnt2 = HashMap::new(); // 用于统计大于等于k+1个不同整数的窗口
        let mut ans = 0;
        let mut left1 = 0;
        let mut left2 = 0;

        for (right, &num) in nums.iter().enumerate() {
            *cnt1.entry(num).or_insert(0) += 1;
            *cnt2.entry(num).or_insert(0) += 1;

            // 计算 cnt >= k 的左边界
            while left1 <= right && cnt1.len() >= k {
                let out = nums[left1];
                *cnt1.get_mut(&out).unwrap() -= 1;
                if cnt1[&out] == 0 {
                    cnt1.remove(&out);
                }
                left1 += 1;
            }

            // 计算 cnt >= k + 1 的左边界
            while left2 <= right && cnt2.len() >= k + 1 {
                let out = nums[left2];
                *cnt2.get_mut(&out).unwrap() -= 1;
                if cnt2[&out] == 0 {
                    cnt2.remove(&out);
                }
                left2 += 1;
            }

            // 两者差值即为以right为结尾的、恰好包含k个不同整数的子数组数量
            ans += (left1 - left2) as i32;
        }

        ans
    }
}
