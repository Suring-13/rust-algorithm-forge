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
