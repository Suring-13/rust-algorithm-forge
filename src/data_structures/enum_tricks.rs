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

// 2905. 找出满足差值条件的下标 II
pub mod n2905 {
    pub fn find_indices(nums: Vec<i32>, index_difference: i32, value_difference: i32) -> Vec<i32> {
        let mut max_idx = 0;
        let mut min_idx = 0;
        for j in index_difference as usize..nums.len() {
            let i = j - index_difference as usize;
            if nums[i] > nums[max_idx] {
                max_idx = i;
            } else if nums[i] < nums[min_idx] {
                min_idx = i;
            }
            if nums[max_idx] - nums[j] >= value_difference {
                return vec![max_idx as i32, j as i32];
            }
            if nums[j] - nums[min_idx] >= value_difference {
                return vec![min_idx as i32, j as i32];
            }
        }
        vec![-1, -1]
    }
}

// 1010. 总持续时间可被 60 整除的歌曲
pub mod n1010 {
    pub fn num_pairs_divisible_by60(time: Vec<i32>) -> i32 {
        let mut ans = 0i64;
        let mut cnt = [0; 60];

        for &t in &time {
            let rem = (t % 60) as usize;
            let target = (60 - rem) % 60;
            ans += cnt[target] as i64;
            cnt[rem] += 1;
        }

        ans as i32
    }
}

// 3185. 构成整天的下标对数目 II
pub mod n3185 {
    pub fn count_complete_day_pairs(hours: Vec<i32>) -> i64 {
        const H: usize = 24;
        let mut ans = 0i64;
        let mut cnt = [0; H];
        for t in hours {
            let t = t as usize % H;
            ans += cnt[(H - t) % H] as i64;
            cnt[t] += 1;
        }
        ans
    }
}

// 2748. 美丽下标对的数目
pub mod n2748 {
    pub fn count_beautiful_pairs(nums: Vec<i32>) -> i32 {
        // 最大公约数 gcd
        fn gcd(x: i32, y: i32) -> i32 {
            if y == 0 { x } else { gcd(y, x % y) }
        }

        let mut ans = 0;
        let mut cnt = [0; 10];

        for &x in &nums {
            let last = x % 10;

            for (y, &cnt_item) in cnt.iter().enumerate() {
                if cnt_item > 0 && gcd(y as i32, last) == 1 {
                    ans += cnt_item;
                }
            }

            let mut first = x;
            while first >= 10 {
                first /= 10;
            }
            cnt[first as usize] += 1;
        }

        ans
    }
}

// 2506. 统计相似字符串对的数目
pub mod n2506 {
    pub fn similar_pairs(words: Vec<String>) -> i32 {
        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0;
        for s in words {
            let mut mask = 0; // 初始化一个空的集合
            for c in s.bytes() {
                mask |= 1 << (c - b'a'); // 把 c 加到集合中
            }
            let e = cnt.entry(mask).or_insert(0);
            ans += *e;
            *e += 1;
        }
        ans
    }
}

// 2874. 有序三元组中的最大值 II
pub mod n2874 {
    pub fn maximum_triplet_value(nums: Vec<i32>) -> i64 {
        let mut ans = 0;
        let mut max_diff = 0;
        let mut pre_max = 0;
        for x in nums {
            ans = ans.max(max_diff as i64 * x as i64);
            max_diff = max_diff.max(pre_max - x);
            pre_max = pre_max.max(x);
        }
        ans
    }
}

// 1497. 检查数组对是否可以被 k 整除
pub mod n1497 {
    pub fn can_arrange(arr: Vec<i32>, k: i32) -> bool {
        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0;

        for &x in &arr {
            // 让余数 m 一定落在 [0, k-1], 也可以用数学求模的API：x.rem_euclid(k)
            let m = (x % k + k) % k;

            // 必须 (k - m) % k
            // 原因：当 m == 0 时，k - m = k，而我们真正要匹配的是 0
            // 不加 %k 会去查 key=k，永远匹配不到存的 key=0，直接错误
            let target = (k - m) % k;

            if *cnt.get(&target).unwrap_or(&0) > 0 {
                ans += 1;
                *cnt.get_mut(&target).unwrap() -= 1;
            } else {
                cnt.entry(m).and_modify(|c| *c += 1).or_insert(1);
            }
        }

        ans == arr.len() / 2
    }
}

// 1031. 两个无重叠子数组的最大和
pub mod n1031 {
    pub fn max_sum_two_no_overlap(nums: Vec<i32>, first_len: i32, second_len: i32) -> i32 {
        let first_len = first_len as usize;
        let second_len = second_len as usize;
        // 计算前缀和数组
        let mut s = vec![0];
        let mut sum = 0;
        for &num in &nums {
            sum += num;
            s.push(sum);
        }

        let mut ans = 0;
        let mut max_sum_a = 0;
        let mut max_sum_b = 0;
        let total_len = first_len + second_len;

        for i in total_len..s.len() {
            // 更新 maxSumA：前一段是 firstLen，后一段是 secondLen
            max_sum_a = max_sum_a.max(s[i - second_len] - s[i - first_len - second_len]);
            // 更新 maxSumB：前一段是 secondLen，后一段是 firstLen
            max_sum_b = max_sum_b.max(s[i - first_len] - s[i - first_len - second_len]);

            // 计算两种组合的最大值
            let case1 = max_sum_a + (s[i] - s[i - second_len]);
            let case2 = max_sum_b + (s[i] - s[i - first_len]);
            ans = ans.max(case1).max(case2);
        }

        ans
    }
}

// 2555. 两个线段获得的最多奖品
pub mod n2555 {
    pub fn maximize_win(prize_positions: Vec<i32>, k: i32) -> i32 {
        let n = prize_positions.len();

        if k * 2 + 1 >= prize_positions[n - 1] - prize_positions[0] {
            return n as _;
        }

        let mut ans = 0;
        let mut mx = 0;
        let mut left = 0;
        let mut right = 0;

        for (mid, &p) in prize_positions.iter().enumerate() {
            // 把 prize_positions[mid] 视作第二条线段的左端点，计算第二条线段可以覆盖的最大奖品下标
            while right < n && prize_positions[right] - p <= k {
                right += 1;
            }
            // 循环结束后，right-1 是第二条线段可以覆盖的最大奖品下标
            ans = ans.max(mx + right - mid);

            // 把 prize_positions[mid] 视作第一条线段的右端点，计算第一条线段可以覆盖的最小奖品下标
            while p - prize_positions[left] > k {
                left += 1;
            }
            // 循环结束后，left 是第一条线段可以覆盖的最小奖品下标
            mx = mx.max(mid - left + 1);
        }

        ans as _
    }
}

// 1995. 统计特殊四元组
pub mod n1995 {
    pub fn count_quadruplets(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut ans = 0;
        let mut hashmap = std::collections::HashMap::new();

        for c in 0..n {
            // 枚举nums[d] - nums[c]
            for d in (c + 1)..n {
                let key = nums[d] - nums[c];
                ans += hashmap.get(&key).copied().unwrap_or(0);
            }
            // 维护nums[a] + nums[b]
            for a in 0..c {
                let key = nums[a] + nums[c];
                *hashmap.entry(key).or_insert(0) += 1;
            }
        }

        ans
    }
}

// 3404. 统计特殊子序列的数目
pub mod n3404 {
    pub fn number_of_subsequences(nums: Vec<i32>) -> i64 {
        let mut ans: i64 = 0;
        let mut cnt = std::collections::HashMap::new();
        let n = nums.len();

        // 求最大公约数
        fn gcd(a: i32, b: i32) -> i32 {
            if b == 0 { a } else { gcd(b, a % b) }
        }

        // 枚举 c 的位置
        for i in 4..n.saturating_sub(2) {
            // b = nums[i-2]
            let b = nums[i - 2];

            // 枚举 a: nums[0 .. i-3]
            for &a_num in &nums[0..i - 3] {
                let a = a_num;
                let g = gcd(a, b);
                // 因为浮点数不能比较，所以使用约分后的分式作为 key
                let key = (a / g, b / g);
                *cnt.entry(key).or_insert(0) += 1;
            }

            let c = nums[i];

            // 枚举 d: nums[i+2 ..]
            for &d_num in &nums[i + 2..] {
                let d = d_num;
                let g = gcd(d, c);
                // 因为浮点数不能比较，所以使用约分后的分式作为 key
                let key = (d / g, c / g);
                ans += cnt.get(&key).unwrap_or(&0);
            }
        }

        ans
    }
}

// 3267. 统计近似相等数对 II
pub mod n3267 {
    pub fn count_pairs(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable();

        let mut cnt = std::collections::HashMap::new();
        let mut ans = 0;

        for &x in &nums {
            let mut st = std::collections::HashSet::new();
            st.insert(x);

            let mut s: Vec<char> = x.to_string().chars().collect();
            let m = s.len();

            for i in 0..m {
                for j in (i + 1)..m {
                    s.swap(i, j);
                    let num1 = s.iter().collect::<String>().parse::<i32>().unwrap();
                    st.insert(num1);

                    for p in (i + 1)..m {
                        for q in (p + 1)..m {
                            s.swap(p, q);
                            let num2 = s.iter().collect::<String>().parse::<i32>().unwrap();
                            st.insert(num2);
                            s.swap(p, q);
                        }
                    }
                    s.swap(i, j);
                }
            }

            let sum: i32 = st.iter().map(|v| cnt.get(v).unwrap_or(&0)).sum();
            ans += sum;
            *cnt.entry(x).or_insert(0) += 1;
        }

        ans
    }
}

// 3480. 删除一个冲突对后最大子数组数目
pub mod n3480 {
    pub fn max_subarrays(n: i32, conflicting_pairs: Vec<Vec<i32>>) -> i64 {
        let mut groups = vec![vec![]; (n + 1) as usize];
        for p in conflicting_pairs {
            let a = p[0];
            let b = p[1];
            let (x, y) = if a < b { (a, b) } else { (b, a) };
            groups[x as usize].push(y);
        }

        let mut ans = 0i64;
        let mut extra = vec![0i64; (n + 2) as usize];
        let mut b = vec![n + 1, n + 1];

        for i in (1..=n).rev() {
            // b + groups[i]
            b.extend(&groups[i as usize]);

            // 排序取前2个最小
            b.sort_unstable();
            b.truncate(2);

            ans += (b[0] - i) as i64;
            extra[b[0] as usize] += (b[1] - b[0]) as i64;
        }

        let max_extra = *extra.iter().max().unwrap();
        ans + max_extra
    }
}

// 454. 四数相加 II
pub mod n454 {
    pub fn four_sum_count(
        nums1: Vec<i32>,
        nums2: Vec<i32>,
        nums3: Vec<i32>,
        nums4: Vec<i32>,
    ) -> i32 {
        let mut cnt = std::collections::HashMap::new();
        for x in nums1 {
            for &y in &nums2 {
                *cnt.entry(x + y).or_insert(0) += 1;
            }
        }

        let mut ans = 0;
        for x in nums3 {
            for &y in &nums4 {
                ans += cnt.get(&(-x - y)).unwrap_or(&0);
            }
        }
        ans
    }
}

// 220. 存在重复元素 III
pub mod n220 {
    pub fn contains_nearby_almost_duplicate(
        nums: Vec<i32>,
        index_diff: i32,
        value_diff: i32,
    ) -> bool {
        let mut window: Vec<i32> = Vec::new();
        let idx_diff = index_diff as usize;

        for r in 0..nums.len() {
            // 移除窗口左侧过期元素 [r-index_diff , r]
            if r > idx_diff {
                let val = nums[r - idx_diff - 1];
                // 找到删除位置
                let pos = window.binary_search(&val).unwrap();
                window.remove(pos);
            }

            let cur = nums[r];
            // 找 >= cur - value_diff 的第一个元素
            let target = cur - value_diff;
            let pos = window.binary_search(&target).unwrap_or_else(|p| p);

            if pos < window.len() && window[pos] <= cur + value_diff {
                return true;
            }

            // 有序插入当前元素
            let insert_pos = window.binary_search(&cur).unwrap_or_else(|p| p);
            window.insert(insert_pos, cur);
        }

        false
    }
}

// 3027. 人员站位的方案数 II
pub mod n3027 {
    pub fn number_of_pairs(mut points: Vec<Vec<i32>>) -> i32 {
        points.sort_unstable_by_key(|p| (p[0], -p[1])); // x 升序，y 降序
        let mut ans = 0;
        for (i, p) in points.iter().enumerate() {
            let y1 = p[1];
            let mut max_y = i32::MIN;
            for q in &points[i + 1..] {
                let y2 = q[1];
                if y2 <= y1 && y2 > max_y {
                    max_y = y2;
                    ans += 1;
                }
                if max_y == y1 {
                    break;
                }
            }
        }
        ans
    }
}

// 3713. 最长的平衡子串 I
pub mod n3713 {
    pub fn longest_balanced(s: String) -> i32 {
        let s = s.as_bytes();
        let mut ans = 0;

        for i in 0..s.len() {
            let mut cnt = std::collections::HashMap::new();
            let mut mx = 0;

            for (j, s_item) in s.iter().enumerate().skip(i) {
                *cnt.entry(*s_item).or_insert(0) += 1;

                mx = mx.max(cnt[s_item]);

                let total = (j - i + 1) as i32;
                if mx * cnt.len() as i32 == total {
                    ans = ans.max(total);
                }
            }
        }

        ans
    }
}

// 2909. 元素和最小的山形三元组 II
pub mod n2909 {
    pub fn minimum_sum(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut suf = vec![0; n]; // 后缀最小值
        suf[n - 1] = nums[n - 1];
        for i in (2..n - 1).rev() {
            suf[i] = suf[i + 1].min(nums[i]);
        }

        let mut ans = i32::MAX;
        let mut pre = nums[0]; // 前缀最小值
        for j in 1..n - 1 {
            // 山形
            if pre < nums[j] && nums[j] > suf[j + 1] {
                ans = ans.min(pre + nums[j] + suf[j + 1]);
            }
            pre = pre.min(nums[j]);
        }
        if ans == i32::MAX { -1 } else { ans }
    }
}

// 3583. 统计特殊三元组
pub mod n3583 {
    pub fn special_triplets(nums: Vec<i32>) -> i32 {
        const MOD: i64 = 1_000_000_007;
        // 后缀计数器初始化
        let mut suf = std::collections::HashMap::new();
        for v in nums.iter() {
            *suf.entry(*v).or_insert(0) += 1;
        }

        let mut ans: i64 = 0;
        let mut pre = std::collections::HashMap::new();

        for x in nums.iter() {
            // 先把当前 x 从后缀剔除
            if let Some(cnt) = suf.get_mut(x) {
                *cnt -= 1;
                if *cnt == 0 {
                    suf.remove(x);
                }
            }

            let target = x * 2;
            let left = pre.get(&target).copied().unwrap_or(0) as i64;
            let right = suf.get(&target).copied().unwrap_or(0) as i64;
            ans += left * right;

            // 当前 x 加入前缀
            *pre.entry(x).or_insert(0) += 1;
        }

        (ans % MOD) as _
    }
}

// 1930. 长度为 3 的不同回文子序列
pub mod n1930 {
    pub fn count_palindromic_subsequence(s: String) -> i32 {
        let s = s.as_bytes();
        let mut ans = 0;

        // 枚举 26 个字母
        for c in b'a'..=b'z' {
            // 找到左边界和右边界
            let left = s.iter().position(|&ch| ch == c);
            let right = s.iter().rposition(|&ch| ch == c);

            // 如果左边界存在且右边界比左边界大 (至少间隔 1 个字符)
            if let (Some(l), Some(r)) = (left, right)
                && r - l >= 2
            {
                // 保证中间有字符
                let mut middle_chars = [false; 26];
                // 遍历中间区域
                for &ch in s.iter().take(r).skip(l + 1) {
                    let idx = (ch - b'a') as usize;
                    if !middle_chars[idx] {
                        middle_chars[idx] = true;
                        ans += 1;
                    }
                }
            }
        }
        ans
    }
}

// 3128. 直角三角形
pub mod n3128 {
    pub fn number_of_right_triangles(grid: Vec<Vec<i32>>) -> i64 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        let row_cnt: Vec<i64> = grid
            .iter()
            .map(|r| r.iter().filter(|&&x| x == 1).count() as i64)
            .collect();
        let n = grid[0].len();
        let col_cnt: Vec<i64> = (0..n)
            .map(|j| grid.iter().filter(|&r| r[j] == 1).count() as i64)
            .collect();

        grid.iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
            .filter(|(_, _, v)| *v == 1)
            .map(|(i, j, _)| (row_cnt[i] - 1) * (col_cnt[j] - 1))
            .sum()
    }
}

// 447. 回旋镖的数量
pub mod n447 {
    pub fn number_of_boomerangs(points: Vec<Vec<i32>>) -> i32 {
        let mut ans = 0;
        let mut dist_count = std::collections::HashMap::new();

        for p1 in &points {
            dist_count.clear();
            for p2 in &points {
                let dx = p1[0] - p2[0];
                let dy = p1[1] - p2[1];
                let dist_sq = dx * dx + dy * dy;

                let cnt = dist_count.entry(dist_sq).or_insert(0);
                ans += 2 * *cnt;
                *cnt += 1;
            }
        }
        ans
    }
}

// 456. 132 模式
pub mod n456 {
    pub fn find132pattern(nums: Vec<i32>) -> bool {
        let n = nums.len();
        if n < 3 {
            return false;
        }

        let mut left_min = nums[0];

        // 右侧用 Vec + 保持有序
        let mut right_all = nums[2..].to_vec();
        right_all.sort_unstable();

        for j in 1..n - 1 {
            let num_j = nums[j];

            if left_min < num_j {
                // 二分查找：找到第一个 > left_min 的值
                let idx = right_all.partition_point(|&x| x <= left_min);
                if idx < right_all.len() && right_all[idx] < num_j {
                    return true;
                }
            }

            // 更新左侧最小值
            left_min = left_min.min(num_j);

            // 删除一个 nums[j+1]（保持有序）
            let val = nums[j + 1];
            if let Ok(pos) = right_all.binary_search(&val) {
                right_all.remove(pos);
            }
        }

        false
    }
}

// 1534. 统计好三元组
pub mod n1534 {
    pub fn count_good_triplets(arr: Vec<i32>, a: i32, b: i32, c: i32) -> i32 {
        let mut idx = (0..arr.len()).collect::<Vec<_>>();
        idx.sort_unstable_by_key(|&i| arr[i]);

        let mut ans = 0;
        for &j in &idx {
            let y = arr[j];
            let mut left = vec![];
            for &i in &idx {
                if i < j && (arr[i] - y).abs() <= a {
                    left.push(arr[i]);
                }
            }

            let mut right = vec![];
            for &k in &idx {
                if k > j && (arr[k] - y).abs() <= b {
                    right.push(arr[k]);
                }
            }

            let mut k1 = 0;
            let mut k2 = 0;
            for x in left {
                while k2 < right.len() && right[k2] <= x + c {
                    k2 += 1;
                }
                while k1 < right.len() && right[k1] < x - c {
                    k1 += 1;
                }
                ans += k2 - k1;
            }
        }
        ans as _
    }
}

// 2242. 节点序列的最大得分
pub mod n2242 {
    pub fn maximum_score(scores: Vec<i32>, edges: Vec<Vec<i32>>) -> i32 {
        let n = scores.len();
        let mut g: Vec<Vec<(i32, usize)>> = vec![vec![]; n];

        for e in &edges {
            let x = e[0] as usize;
            let y = e[1] as usize;
            g[x].push((scores[y], y));
            g[y].push((scores[x], x));
        }

        // 每个点只保留分数最大的 3 个邻居
        for adj in &mut g {
            adj.sort_by(|a, b| b.0.cmp(&a.0));
            adj.truncate(3);
        }

        let mut ans = -1;

        for e in &edges {
            let x = e[0] as usize;
            let y = e[1] as usize;

            for &(score_a, a) in &g[x] {
                for &(score_b, b) in &g[y] {
                    if y != a && a != b && b != x {
                        let total = score_a + scores[x] + scores[y] + score_b;
                        ans = ans.max(total);
                    }
                }
            }
        }

        ans
    }
}
