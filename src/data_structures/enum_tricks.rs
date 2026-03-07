// 1. 两数之和
pub mod n1 {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut idx = std::collections::HashMap::new(); // 创建一个空哈希表
        // 枚举 j
        for (j, &x) in nums.iter().enumerate() {
            // 在左边找 nums[i]，满足 nums[i]+x=target
            if let Some(&i) = idx.get(&(target - x)) {
                return vec![i as i32, j as i32]; // 返回两个数的下标
            }
            idx.insert(x, j); // 保存 nums[j] 和 j
        }
        unreachable!() // 题目保证有解，循环中一定会 return
    }
}

// 1512. 好数对的数目
pub mod n1512 {
    pub fn num_identical_pairs(nums: Vec<i32>) -> i32 {
        let mut ans = 0; // 最终答案：好数对总数
        let mut cnt = std::collections::HashMap::new(); // 记录每个数字已经出现几次

        for x in nums {
            // 遍历每个数 x（相当于 j 位置）
            // 拿到 x 之前出现的次数，没有就是 0
            let e = cnt.entry(x).or_insert(0);

            ans += *e; // 之前有多少个 x，就新增多少个好数对
            *e += 1; // 再把当前 x 计入计数
        }

        ans
    }
}
