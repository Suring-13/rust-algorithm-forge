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

// 1128. 等价多米诺骨牌对的数量
pub mod n1128 {
    pub fn num_equiv_domino_pairs(dominoes: Vec<Vec<i32>>) -> i32 {
        let mut ans = 0;
        let mut cnt = [[0; 10]; 10];
        for d in dominoes {
            let mut a = d[0] as usize;
            let mut b = d[1] as usize;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            ans += cnt[a][b];
            cnt[a][b] += 1;
        }
        ans
    }
}

// 1679. K 和数对的最大数目
pub mod n1679 {
    pub fn max_operations(nums: Vec<i32>, k: i32) -> i32 {
        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0;
        for x in nums {
            if let Some(c) = cnt.get_mut(&(k - x))
                && *c > 0
            {
                *c -= 1;
                ans += 1;
                continue;
            }
            *cnt.entry(x).or_insert(0) += 1;
        }
        ans
    }
}

// 219. 存在重复元素 II
pub mod n219 {
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
        let mut last = std::collections::HashMap::new();
        let k = k as usize;

        for (i, x) in nums.iter().enumerate() {
            if let Some(&j) = last.get(x)
                && i - j <= k
            {
                return true;
            }

            last.insert(x, i);
        }

        false
    }
}

// 2260. 必须拿起的最小连续卡牌数
pub mod n2260 {
    pub fn minimum_card_pickup(cards: Vec<i32>) -> i32 {
        let mut ans = cards.len() + 1;
        let mut pos = std::collections::HashMap::new();

        for (i, &v) in cards.iter().enumerate() {
            if let Some(&p) = pos.get(&v) {
                ans = ans.min(i - p + 1);
            }
            pos.insert(v, i);
        }

        if ans <= cards.len() { ans as i32 } else { -1 }
    }
}

// 2001. 可互换矩形的组数
pub mod n2001 {
    pub fn interchangeable_rectangles(rectangles: Vec<Vec<i32>>) -> i64 {
        // 求最大公约数
        fn gcd(a: i32, b: i32) -> i32 {
            if b == 0 { a } else { gcd(b, a % b) }
        }

        let mut map = std::collections::HashMap::new();
        let mut ans = 0i64;

        for rect in rectangles {
            let w = rect[0];
            let h = rect[1];
            let g = gcd(w, h);
            // 因为浮点数不能比较，所以使用约分后的分式作为 key
            let key = (w / g, h / g);

            ans += *map.get(&key).unwrap_or(&0);
            *map.entry(key).or_insert(0) += 1;
        }

        ans
    }
}

// 2815. 数组中的最大数对和
pub mod n2815 {
    pub fn max_sum(nums: Vec<i32>) -> i32 {
        let mut ans = -1; // 最终答案，初始为 -1
        let mut max_val = [i32::MIN; 10]; // max_val[d] = 最大位是 d 的最大数

        for &v in &nums {
            // 遍历每个数
            let mut max_d = 0; // 求当前数字 v 的最大位
            let mut x = v;
            while x > 0 {
                max_d = max_d.max(x % 10); // 取个位，更新最大位
                x /= 10;
            }

            let max_d = max_d as usize;
            // 当前数 + 之前同最大位的最大数 → 更新答案
            ans = ans.max(v + max_val[max_d]);
            // 更新同最大位的最大数
            max_val[max_d] = max_val[max_d].max(v);
        }

        ans
    }
}

// 3623. 统计梯形的数目 I
pub mod n3623 {
    pub fn count_trapezoids(points: Vec<Vec<i32>>) -> i32 {
        const MOD: i64 = 1_000_000_007;
        let mut cnt = std::collections::HashMap::new();

        // 统计每一行（水平线）有多少个点
        for p in &points {
            let y = p[1];
            *cnt.entry(y).or_insert(0) += 1;
        }

        let mut ans = 0i64;
        let mut s = 0i64;

        for &c in cnt.values() {
            let k = (c as i64) * (c as i64 - 1) / 2;
            ans += s * k;
            ans %= MOD;
            s += k;
        }

        ans as i32
    }
}

// 2364. 统计坏数对的数目
pub mod n2364 {
    pub fn count_bad_pairs(nums: Vec<i32>) -> i64 {
        let n = nums.len() as i64;
        let mut ans = n * (n - 1) / 2;
        let mut cnt = std::collections::HashMap::new();
        for (i, x) in nums.into_iter().enumerate() {
            let e = cnt.entry(x - i as i32).or_insert(0);
            ans -= *e as i64;
            *e += 1;
        }
        ans
    }
}

// 3805. 统计凯撒加密对数目
pub mod n3805 {
    pub fn count_pairs(words: Vec<String>) -> i32 {
        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0;

        for s in words {
            let bytes = s.as_bytes();

            let base = bytes[0];
            // 构建归一化后的 bytes 数组
            let mut normalized = Vec::with_capacity(bytes.len());
            for &b in bytes {
                // 加26是为了确保非负
                let offset = (b + 26 - base) % 26;
                // 归一化为从 b'a' 开始的字节； 也可以把加a去掉，因为所有的加a，相当与大家都没加
                normalized.push(b'a' + offset);
            }

            ans += cnt.get(&normalized).unwrap_or(&0);
            *cnt.entry(normalized).or_insert(0) += 1;
        }

        ans
    }
}

// 3371. 识别数组中的最大异常值
pub mod n3371 {
    pub fn get_largest_outlier(nums: Vec<i32>) -> i32 {
        // 1. 统计数字出现次数
        let mut cnt = std::collections::HashMap::new();
        for num in nums.iter() {
            *cnt.entry(num).or_insert(0) += 1;
        }

        // 2. 计算总和
        let total: i32 = nums.iter().sum();

        // 3. 初始化结果为极小值
        let mut ans = i32::MIN;

        // 4. 遍历每个数字计算 t
        for &y in nums.iter() {
            let t = total - 2 * y;

            // 检查 t 是否在统计结果中
            if let Some(&count) = cnt.get(&t) {
                // 满足条件：t≠y 或 t=y但出现次数>1
                if t != y || count > 1 {
                    ans = ans.max(t);
                }
            }
        }

        ans
    }
}

// 3761. 镜像对之间最小绝对距离
pub mod n3761 {
    pub fn min_mirror_pair_distance(nums: Vec<i32>) -> i32 {
        // 数字反转
        fn reverse_num(mut x: i32) -> i32 {
            let mut rev = 0i32;
            while x > 0 {
                rev = rev * 10 + (x % 10);
                x /= 10;
            }
            rev
        }

        let mut last_index = std::collections::HashMap::new();
        let mut min_dist = i32::MAX;

        for (j, &x) in nums.iter().enumerate() {
            let j = j as i32;

            // 如果当前数已经被作为「反转数」存过，更新最小距离
            if let Some(&prev) = last_index.get(&x) {
                min_dist = min_dist.min(j - prev);
            }

            let rev = reverse_num(x);
            last_index.insert(rev, j);
        }

        if min_dist == i32::MAX { -1 } else { min_dist }
    }
}

// 1014. 最佳观光组合
pub mod n1014 {
    pub fn max_score_sightseeing_pair(values: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut mx = 0; // j 左边的 values[i] + i 的最大值
        for (j, &v) in values.iter().enumerate() {
            ans = ans.max(mx + v - j as i32);
            mx = mx.max(v + j as i32);
        }
        ans
    }
}

// 1814. 统计一个数组中好对子的数目
pub mod n1814 {
    pub fn count_nice_pairs(nums: Vec<i32>) -> i32 {
        fn reverse(mut x: i32) -> i32 {
            let mut rev = 0;
            while x > 0 {
                rev = rev * 10 + x % 10;
                x /= 10;
            }
            rev
        }
        const MOD: i64 = 1_000_000_007;

        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0i64;

        for &num in &nums {
            let rev = reverse(num);
            let key = num - rev;
            ans += *cnt.get(&key).unwrap_or(&0) as i64;
            *cnt.entry(key).or_insert(0) += 1;
        }

        (ans % MOD) as _
    }
}

// 3584. 子序列首尾元素的最大乘积
pub mod n3584 {
    pub fn maximum_product(nums: Vec<i32>, m: i32) -> i64 {
        let m = m as usize;
        let nums: Vec<i64> = nums.into_iter().map(|x| x as i64).collect();
        let mut ans = i64::MIN;
        let mut mx = i64::MIN;
        let mut mn = i64::MAX;

        for i in (m - 1)..nums.len() {
            // 维护左边 [0, i-m+1] 的最小/最大值
            let y = nums[i - m + 1];
            mn = mn.min(y);
            mx = mx.max(y);

            // 枚举右端点
            let x = nums[i];
            ans = ans.max((x * mn).max(x * mx));
        }

        ans
    }
}
