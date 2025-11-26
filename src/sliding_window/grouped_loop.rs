// 485. 最大连续 1 的个数
pub mod n485 {
    pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut cnt = 0; // 存储当前连续1的个数

        // 遍历nums中的每个元素
        for &x in nums.iter() {
            if x == 1 {
                cnt += 1; // 遇到1，当前连续计数+1
                ans = ans.max(cnt); // 更新最大值
            } else {
                cnt = 0; // 遇到0，重置当前连续计数
            }
        }

        ans
    }
}

// 1446. 连续字符
pub mod n1446 {
    pub fn max_power(s: String) -> i32 {
        if s.is_empty() {
            return 0;
        }

        let mut ans = 1;
        let mut cnt = 1;
        let bytes = s.as_bytes();

        for i in 1..bytes.len() {
            if bytes[i] == bytes[i - 1] {
                cnt += 1;
                ans = ans.max(cnt);
            } else {
                cnt = 1;
            }
        }

        ans
    }
}

// 1869. 哪种连续子字符串更长
pub mod n1869 {
    pub fn check_zero_ones(s: &str) -> bool {
        let mut max1 = 0;
        let mut max0 = 0;
        let bytes = s.as_bytes();
        let mut i = 0;
        let n = bytes.len();

        while i < n {
            let start = i;
            let current = bytes[start];

            // 移动指针至不同字符处
            while i < n && bytes[i] == current {
                i += 1;
            }

            // 更新最长连续长度
            let length = i - start;
            if current == b'1' {
                max1 = max1.max(length);
            } else {
                max0 = max0.max(length);
            }
        }

        max1 > max0
    }
}

// 2414. 最长的字母序连续子字符串的长度
pub mod n2414 {
    pub fn longest_continuous_substring(s: String) -> i32 {
        let mut ans = 1;
        let mut cnt = 1;
        let s = s.as_bytes();
        for i in 1..s.len() {
            if s[i - 1] + 1 == s[i] {
                cnt += 1;
                ans = ans.max(cnt);
            } else {
                cnt = 1;
            }
        }
        ans
    }
}

// 3456. 找出长度为 K 的特殊子字符串
pub mod n3456 {
    pub fn has_special_substring(s: String, k: i32) -> bool {
        if k == 0 {
            return false;
        }
        let mut cnt = 0;
        let bytes = s.as_bytes();
        for i in 0..bytes.len() {
            cnt += 1;
            // 检查是否为最后一个字符，或当前字符与下一个字符不同
            if i == bytes.len() - 1 || bytes[i] != bytes[i + 1] {
                if cnt == k {
                    return true;
                }
                cnt = 0;
            }
        }
        false
    }
}

// 2348. 全 0 子数组的数目
pub mod n2348 {
    pub fn zero_filled_subarray(nums: Vec<i32>) -> i64 {
        let mut ans = 0;
        let mut last = -1;
        for (i, x) in nums.into_iter().enumerate() {
            let i = i as i64;
            if x != 0 {
                last = i; // 记录上一个非 0 元素的位置
            } else {
                ans += (i - last) as i64;
            }
        }
        ans
    }
}
