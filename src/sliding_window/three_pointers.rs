// 2367. 等差三元组的数目
pub mod n2367 {
    pub fn arithmetic_triplets(nums: Vec<i32>, diff: i32) -> i32 {
        let mut ans = 0;
        let mut i = 0;
        let mut j = 1;
        // 遍历 nums[k]（即 x）
        for &x in &nums {
            // 移动 j 指针，确保 nums[j] + diff 不小于 x
            while nums[j] + diff < x {
                j += 1;
            }
            // 若 nums[j] + diff 大于 x，跳过当前 x
            if nums[j] + diff > x {
                continue;
            }
            // 移动 i 指针，确保 nums[i] + 2*diff 不小于 x
            while nums[i] + 2 * diff < x {
                i += 1;
            }
            // 若满足条件，计数加 1
            if nums[i] + 2 * diff == x {
                ans += 1;
            }
        }
        ans
    }
}

// 795. 区间子数组个数
pub mod n795 {
    pub fn num_subarray_bounded_max(nums: Vec<i32>, left: i32, right: i32) -> i32 {
        let mut res = 0;
        let mut last2 = -1; // 最近一个 > right 的元素索引
        let mut last1 = -1; // 最近一个在 [left, right] 内的元素索引

        for (i, &x) in nums.iter().enumerate() {
            let i = i as i32;
            // 情况1：当前元素在 [left, right] 区间内（x >= left 且 x <= right）
            if x >= left && x <= right {
                last1 = i;
            }
            // 情况2：当前元素 > right，重置边界
            else if x > right {
                last2 = i;
                last1 = -1;
            }
            // 情况3：当前元素 < left，不更新边界（直接跳过）

            // 存在有效 last1 时，累加合法子数组数量
            if last1 != -1 {
                res += last1 - last2;
            }
        }

        res
    }
}

// 2444. 统计定界子数组的数目
pub mod n2444 {
    pub fn count_subarrays(nums: Vec<i32>, min_k: i32, max_k: i32) -> i64 {
        let mut res = 0 as i64;
        let (mut border, mut last_min, mut last_max) = (-1, -1, -1);
        for (i, &x) in nums.iter().enumerate() {
            if x < min_k || x > max_k {
                border = i as i64;
                last_min = -1 as i64;
                last_max = -1 as i64;
            }
            if x == min_k {
                last_min = i as i64;
            }
            if x == max_k {
                last_max = i as i64;
            }
            if last_min != -1 && last_max != -1 {
                res += last_min.min(last_max) - border;
            }
        }
        res
    }
}

// 3347. 执行操作后元素的最高频率 II
pub mod n3347 {
    pub fn max_frequency(mut nums: Vec<i32>, k: i32, num_operations: i32) -> i32 {
        nums.sort_unstable();
        let n = nums.len();
        let mut ans = 0;
        let mut cnt = 0;
        let mut left = 0;
        let mut right = 0;

        // 第一阶段：计算有多少个数能变成 x，其中 x=nums[i]。用同向三指针实现。
        for i in 0..n {
            let x = nums[i];
            cnt += 1;
            // 循环直到连续相同段的末尾，这样可以统计出 x 的出现次数
            if i < n - 1 && x == nums[i + 1] {
                continue;
            }
            // 收缩左边界：确保左边界元素 >= x - k
            while nums[left] < x - k {
                left += 1;
            }
            // 扩展右边界：确保右边界元素 > x + k（最终right-left为区间长度）
            while right < n && nums[right] <= x + k {
                right += 1;
            }

            ans = ans.max((right - left).min(cnt + num_operations as usize));
            cnt = 0; // 重置当前数字段的计数
        }

        // 若第一阶段结果已满足，直接返回
        if ans as i32 >= num_operations {
            return ans as i32;
        }

        // 第二阶段：计算有多少个数能变成 x，其中 x 不一定在 nums 中。用同向双指针实现
        let mut left = 0;
        for right in 0..n {
            let x = nums[right];
            // 收缩左边界：确保左边界元素 >= x - 2*k
            while nums[left] < x - 2 * k {
                left += 1;
            }
            // 更新最大区间长度
            ans = ans.max(right - left + 1);
        }

        // 最终结果不超过操作数上限
        ans.min(num_operations as usize) as i32
    }
}
