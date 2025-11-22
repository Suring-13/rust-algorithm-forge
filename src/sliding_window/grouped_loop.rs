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
