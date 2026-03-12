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

// 2016. 增量元素之间的最大差值
pub mod n2016 {
    pub fn maximum_difference(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut pre_min = nums[0];
        for x in nums {
            ans = ans.max(x - pre_min);
            pre_min = pre_min.min(x);
        }
        if ans > 0 { ans } else { -1 }
    }
}

// 624. 数组列表中的最大距离
pub mod n624 {
    pub fn max_distance(arrays: Vec<Vec<i32>>) -> i32 {
        let mut ans = 0;
        let mut mn = i32::MAX / 2; // 防止减法溢出
        let mut mx = i32::MIN / 2;
        for a in arrays {
            let x = a[0];
            let y = a[a.len() - 1];
            // 优化：不需要求绝对值，因为
            // 如果 A[n−1]−mn<0，则 A[0]≤A[n−1]<mn≤mx，mx−A[0]≥mn−A[n−1]=|A[n−1]−mn| >0
            // 如果 mx−A[0]<0， 则 mn≤mx<A[0]≤A[n−1], A[n−1]−mn≥A[0]-mx=|mx−A[0]| > 0
            ans = ans.max(y - mn).max(mx - x);
            mn = mn.min(x);
            mx = mx.max(y);
        }
        ans
    }
}

// 2342. 数位和相等数对的最大和
pub mod n2342 {
    pub fn maximum_sum(nums: Vec<i32>) -> i32 {
        let mut ans = -1;
        let mut mx = [i32::MIN; 82]; // 至多 9 个 9 相加
        for num in nums {
            // 枚举 num = nums[j]
            let mut s = 0; // num 的数位和
            let mut x = num;
            while x > 0 {
                // 枚举 num 的每个数位
                s += (x % 10) as usize;
                x /= 10;
            }
            ans = ans.max(mx[s] + num); // 左边找一个数位和也为 s 的最大的 nums[i]
            mx[s] = mx[s].max(num); // 维护数位和等于 s 的最大元素
        }
        ans
    }
}
