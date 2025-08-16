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
