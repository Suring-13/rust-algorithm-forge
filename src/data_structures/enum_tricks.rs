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

// 2441. 与对应负数同时存在的最大正整数
pub mod n2441 {
    pub fn find_max_k(nums: Vec<i32>) -> i32 {
        let mut ans = -1;
        let mut s = std::collections::HashSet::new();

        for &x in &nums {
            // 检查当前数字的相反数是否在集合中
            if s.contains(&(-x)) {
                // 更新最大的绝对值
                ans = ans.max(x.abs());
            }
            // 将当前数字加入集合
            s.insert(x);
        }

        ans
    }
}

// 121. 买卖股票的最佳时机
pub mod n121 {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut min_price = prices[0];
        for p in prices {
            ans = ans.max(p - min_price);
            min_price = min_price.min(p);
        }
        ans
    }
}
