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
