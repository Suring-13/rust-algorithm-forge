// 303. 区域和检索 - 数组不可变
pub mod n303 {
    pub struct NumArray {
        pub pre_sum: Vec<i32>,
    }

    impl NumArray {
        pub fn new(nums: Vec<i32>) -> Self {
            let mut pre_sum = vec![0; nums.len() + 1];
            // pre_sum[0] = 0
            // pre_sum[i] = nums[0] + nums[1] + ... + nums[i-1]
            for (i, &num) in nums.iter().enumerate() {
                pre_sum[i + 1] = pre_sum[i] + num;
            }
            Self { pre_sum }
        }

        // [left, right] 区间和
        pub fn sum_range(&self, left: i32, right: i32) -> i32 {
            let l = left as usize;
            let r = right as usize;
            self.pre_sum[r + 1] - self.pre_sum[l]
        }
    }
}

// 3427. 变长子数组求和
pub mod n3427 {
    pub fn subarray_sum(nums: &[i32]) -> i32 {
        let len = nums.len();
        let mut prefix = vec![0i32; len + 1];
        let mut ans = 0i32;

        for (i, &x) in nums.iter().enumerate() {
            prefix[i + 1] = prefix[i] + x;
            let left = i.saturating_sub(x as usize);
            ans += prefix[i + 1] - prefix[left];
        }

        ans
    }
}

// 2559. 统计范围内的元音字符串数
pub mod n2559 {
    pub fn vowel_strings(words: Vec<String>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let vowels = [b'a', b'e', b'i', b'o', b'u'];

        // 前缀和数组
        let mut pre_sum = vec![0; words.len() + 1];
        for (i, w) in words.iter().enumerate() {
            let first = w.as_bytes()[0];
            let last = w.as_bytes().last().unwrap();
            let valid = vowels.contains(&first) && vowels.contains(last);
            pre_sum[i + 1] = pre_sum[i] + if valid { 1 } else { 0 };
        }

        queries
            .into_iter()
            .map(|q| {
                let l = q[0] as usize;
                let r = q[1] as usize;
                pre_sum[r + 1] - pre_sum[l]
            })
            .collect()
    }
}

// 1310. 子数组异或查询
pub mod n1310 {
    pub fn xor_queries(arr: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        // 构造 0 开头的迭代器: 0 followed by arr
        let iter = std::iter::once(0).chain(arr);

        // 前缀异或
        let mut prexor = Vec::new();
        iter.fold(0, |acc, val| {
            let new_acc = acc ^ val;
            prexor.push(new_acc);
            new_acc
        });

        queries
            .into_iter()
            .map(|q| prexor[q[0] as usize] ^ prexor[q[1] as usize + 1])
            .collect()
    }
}

// 3152. 特殊数组 II
pub mod n3152 {
    pub fn is_array_special(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<bool> {
        let s: Vec<i32> = std::iter::once(0)
            .chain(
                nums.windows(2)
                    .map(|w| (w[0] % 2 == w[1] % 2) as i32)
                    .scan(0, |s, x| {
                        *s += x;
                        Some(*s)
                    }),
            )
            .collect();

        queries
            .into_iter()
            .map(|q| s[q[0] as usize] == s[q[1] as usize])
            .collect()
    }
}

// 1749. 任意子数组和的绝对值的最大值
pub mod n1749 {
    pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
        let mut s = 0;
        let mut mx = 0;
        let mut mn = 0;

        for x in nums {
            s += x;
            mx = mx.max(s);
            mn = mn.min(s);
        }

        mx - mn
    }
}

// 53. 最大子数组和
pub mod n53 {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut ans = i32::MIN;
        let mut min_pre_sum = 0;
        let mut pre_sum = 0;
        for x in nums {
            pre_sum += x; // 当前的前缀和
            ans = ans.max(pre_sum - min_pre_sum); // 减去前缀和的最小值
            min_pre_sum = min_pre_sum.min(pre_sum); // 维护前缀和的最小值
        }
        ans
    }
}

// 3652. 按策略买卖股票的最佳时机
pub mod n3652 {
    pub fn max_profit(prices: Vec<i32>, strategy: Vec<i32>, k: i32) -> i64 {
        let n = prices.len();

        // 前缀和 s: accumulate(p * s)
        let mut s = vec![0; n + 1];
        for i in 0..n {
            s[i + 1] = s[i] + prices[i] as i64 * strategy[i] as i64;
        }

        // 前缀和 s_sell: accumulate(prices)
        let mut s_sell = vec![0; n + 1];
        for i in 0..n {
            s_sell[i + 1] = s_sell[i] + prices[i] as i64;
        }

        // 修改一次的最大收益
        let mut ans = i64::MIN;
        let k = k as usize;
        for i in k..=n {
            let half = k / 2;
            let val = s[i - k] + (s[n] - s[i]) + (s_sell[i] - s_sell[i - half]);
            ans = ans.max(val);
        }

        // 不修改 与 修改一次 取最大
        ans.max(s[n])
    }
}

// 3361. 两个字符串的切换距离
pub mod n3361 {
    pub fn shift_distance(
        s: String,
        t: String,
        next_cost: Vec<i32>,
        previous_cost: Vec<i32>,
    ) -> i64 {
        const ORD_A: u8 = b'a';
        // 构造双倍长度数组，处理环形移位
        let mut nxt_arr = next_cost.clone();
        nxt_arr.extend(&next_cost);
        // 前缀和：带 initial 0
        let mut nxt_sum = vec![0i64];
        let mut sum = 0i64;
        for &val in &nxt_arr {
            sum += val as i64;
            nxt_sum.push(sum);
        }

        // 构造双倍长度数组
        let mut pre_arr = previous_cost.clone();
        pre_arr.extend(&previous_cost);
        // 前缀和：不带 initial 0
        let mut pre_sum = Vec::new();
        let mut sum = 0i64;
        for &val in &pre_arr {
            sum += val as i64;
            pre_sum.push(sum);
        }

        let mut ans = 0i64;
        // 逐字符遍历计算
        for (sc, tc) in s.bytes().zip(t.bytes()) {
            let x = (sc - ORD_A) as usize;
            let y = (tc - ORD_A) as usize;

            // 计算正向代价 next
            let nxt_idx = if y < x { y + 26 } else { y };
            let cost_nxt = nxt_sum[nxt_idx] - nxt_sum[x];

            // 计算反向代价 previous
            let pre_idx = if x < y { x + 26 } else { x };
            let cost_pre = pre_sum[pre_idx] - pre_sum[y];

            // 取最小值累加
            ans += cost_nxt.min(cost_pre);
        }

        ans
    }
}

// 560. 和为 K 的子数组
pub mod n560 {
    pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
        // 哈希表：key=前缀和，value=该前缀和出现次数
        let mut cnt = std::collections::HashMap::with_capacity(nums.len() + 1);
        // 初始化：前缀和为0出现1次，适配 s - k = 0 的情况
        cnt.insert(0, 1);
        let mut s = 0; // 当前前缀和
        let mut ans = 0; // 答案计数

        for x in nums {
            s += x; // 累加更新前缀和
            // 前面有多少个前缀和等于 s - k，就加多少个子数组
            if let Some(&c) = cnt.get(&(s - k)) {
                ans += c;
            }
            // 当前前缀和次数+1，不存在则初始化为0再+1
            *cnt.entry(s).or_insert(0) += 1;
        }
        ans
    }
}

// 930. 和相同的二元子数组
pub mod n930 {
    pub fn num_subarrays_with_sum(nums: Vec<i32>, goal: i32) -> i32 {
        let mut sum = 0;
        let mut cnt = std::collections::HashMap::new();
        let mut ret = 0;

        for &num in &nums {
            *cnt.entry(sum).or_insert(0) += 1;
            sum += num;
            ret += cnt.get(&(sum - goal)).copied().unwrap_or(0);
        }

        ret
    }
}

// 1524. 和为奇数的子数组数目
pub mod n1524 {
    pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
        const MOD: i64 = 1_000_000_007;
        let mut odd = 0;
        let mut even = 1;
        let mut subarrays: i64 = 0;
        let mut total: i64 = 0;

        for &x in &arr {
            total += x as i64;

            subarrays += if total % 2 == 0 {
                odd as i64
            } else {
                even as i64
            };

            if total % 2 == 0 {
                even += 1;
            } else {
                odd += 1;
            }
        }

        (subarrays % MOD) as i32
    }
}

// 974. 和可被 K 整除的子数组
pub mod n974 {
    pub fn subarrays_div_by_k(nums: Vec<i32>, k: i32) -> i32 {
        let mut map = std::collections::HashMap::from([(0, 1)]);
        let mut s = 0;
        let mut res = 0;
        for x in nums {
            s = (s + x).rem_euclid(k);
            res += map.get(&s).unwrap_or(&0);
            *map.entry(s).or_insert(0) += 1;
        }
        res
    }
}

// 523. 连续的子数组和
pub mod n523 {
    pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
        let mut sum = 0;
        let mut map = std::collections::HashMap::new();
        map.insert(0, -1);

        for (idx, &n) in nums.iter().enumerate() {
            sum += n;
            let rem = sum % k;
            match map.get(&rem) {
                Some(&pre) => {
                    if idx as i32 - pre >= 2 {
                        return true;
                    }
                }
                None => {
                    map.insert(rem, idx as i32);
                }
            }
        }
        false
    }
}

// 2588. 统计美丽子数组数目
pub mod n2588 {
    pub fn beautiful_subarrays(nums: Vec<i32>) -> i64 {
        let mut ans = 0;
        let mut s = 0;
        let mut cnt = std::collections::HashMap::with_capacity(nums.len() + 1);
        cnt.insert(0, 1);
        for x in nums {
            s ^= x;
            let e = cnt.entry(s).or_insert(0);
            ans += *e as i64;
            *e += 1;
        }
        ans
    }
}

// 525. 连续数组
pub mod n525 {
    pub fn find_max_length(nums: Vec<i32>) -> i32 {
        // key: 前缀和, value: 第一次出现的下标
        let mut pos = std::collections::HashMap::from([(0, -1)]);
        let mut ans = 0;
        let mut sum = 0;

        for (i, &x) in nums.iter().enumerate() {
            sum += if x == 1 { 1 } else { -1 };
            if let Some(&prev_idx) = pos.get(&sum) {
                ans = ans.max(i as i32 - prev_idx);
            } else {
                pos.insert(sum, i as i32);
            }
        }
        ans
    }
}

// 3755. 最大平衡异或子数组的长度
pub mod n3755 {
    pub fn max_balanced_subarray(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        // key: (xor, diff), value: 首次下标
        let mut pos = std::collections::HashMap::from([((0, 0), -1)]);

        let mut xor = 0;
        let mut diff = 0;

        for (i, &x) in nums.iter().enumerate() {
            xor ^= x;
            if x % 2 == 1 {
                diff += 1;
            } else {
                diff -= 1;
            }

            let key = (xor, diff);
            if let Some(&prev) = pos.get(&key) {
                ans = ans.max(i as i32 - prev);
            } else {
                pos.insert(key, i as i32);
            }
        }

        ans
    }
}

// 3026. 最大好子数组和
pub mod n3026 {
    pub fn maximum_subarray_sum(nums: Vec<i32>, k: i32) -> i64 {
        let mut min_s = std::collections::HashMap::new();
        // 默认初始无穷大
        const INF: i64 = i64::MAX / 2;
        let mut s: i64 = 0;
        let mut ans: i64 = -INF;

        for &x in &nums {
            // 取 min_s[x-k] 和 min_s[x+k] 的较小值
            let val1 = *min_s.get(&(x - k)).unwrap_or(&INF);
            let val2 = *min_s.get(&(x + k)).unwrap_or(&INF);
            let pre_min = val1.min(val2);
            if pre_min != INF {
                ans = ans.max(s + x as i64 - pre_min);
            }

            // 更新 min_s[x]：保留最小的前缀和
            let entry = min_s.entry(x).or_insert(INF);
            *entry = (*entry).min(s);

            s += x as i64;
        }

        if ans == -INF { 0 } else { ans }
    }
}

// 1477. 找两个和为目标值且不重叠的子数组
pub mod n1477 {
    pub fn min_sum_of_lengths(mut arr: Vec<i32>, target: i32) -> i32 {
        // key: 前缀和, value: 该前缀和对应的下标
        let mut pos = std::collections::HashMap::from([(0, -1)]);

        let n = arr.len() as i32;
        // 前缀和
        let mut s = 0;
        // 最终答案，初始设为不可能的最大值 n+1
        let mut ans = n + 1;
        // 记录遍历到当前位置之前，满足和为target的最短子数组长度
        let mut min_l = n;

        for i in 0..(n as usize) {
            let x = arr[i];
            s += x;

            // 找前缀和 = s - target 的位置 j
            if let Some(&j) = pos.get(&(s - target)) {
                // 当前满足条件的子数组长度 (j, i]
                let len = i as i32 - j;

                // j == -1 说明当前是第一段，前面没有可用第二段，用n占位（不参与更新ans）
                // arr[j as usize] 存的是 j 位置之前的最小合法长度
                let pre_min = if j == -1 { n } else { arr[j as usize] };
                ans = ans.min(len + pre_min);

                // 更新全局最短单段长度
                min_l = min_l.min(len);
            }

            // 把当前位置的值覆盖为：到当前为止前面的最小合法长度
            arr[i] = min_l;
            // 更新/记录当前前缀和所在下标
            pos.insert(s, i as i32);
        }

        // 没找到合法两个子数组返回-1
        if ans == n + 1 { -1 } else { ans }
    }
}

// 1546. 和为目标值且不重叠的非空子数组的最大数目
pub mod n1546 {
    pub fn max_non_overlapping(nums: Vec<i32>, target: i32) -> i32 {
        let mut prefix_sum = 0;
        let mut res = 0;
        let mut map = std::collections::HashMap::from([(0, 0)]);
        let mut last = 0;

        for i in 1..=nums.len() {
            prefix_sum += nums[i - 1];

            if let Some(&pos) = map.get(&(prefix_sum - target))
                && pos >= last
            {
                res += 1;
                last = i;
            }

            map.insert(prefix_sum, i);
        }

        res
    }
}

// 1124. 表现良好的最长时间段
pub mod n1124 {
    // 由于我们只需要考虑值在闭区间 [−n,0] 内的前缀和，用数组记录是更加高效的。同时，为了避免用负数访问数组，可以在计算过程中把前缀和取反。
    pub fn longest_wpi(hours: Vec<i32>) -> i32 {
        let n = hours.len();
        let mut pos = vec![0; n + 2];
        let mut ans = 0;
        let mut s = 0;

        for (i, &h) in hours.iter().enumerate() {
            let idx = (i + 1) as i32;
            // 取反，改为减法
            s -= if h > 8 { 1 } else { -1 };

            if s < 0 {
                ans = idx;
            } else {
                // 原本是 s-1，取反改为 s+1
                let key = (s + 1) as usize;
                if pos[key] != 0 {
                    ans = ans.max(idx - pos[key]);
                }
                // 首次出现则记录位置
                let s_key = s as usize;
                if pos[s_key] == 0 {
                    pos[s_key] = idx;
                }
            }
        }

        ans
    }
}

// 3728. 边界与内部和相等的稳定子数组
pub mod n3728 {
    pub fn count_stable_subarrays(capacity: &[i32]) -> i64 {
        let mut map = std::collections::HashMap::new();
        if capacity.len() < 2 {
            return 0;
        }

        let mut pre_sum = capacity[0] as i64;
        let mut res = 0i64;

        for pair in capacity.windows(2) {
            let prev = pair[0] as i64;
            let curr = pair[1] as i64;

            res += map.get(&(curr, pre_sum)).copied().unwrap_or(0);
            *map.entry((prev, prev + pre_sum)).or_insert(0) += 1;
            pre_sum += curr;
        }
        res
    }
}

// 3381. 长度可被 K 整除的子数组的最大元素和
pub mod n3381 {
    pub fn max_subarray_sum(nums: Vec<i32>, k: i32) -> i64 {
        let k = k as usize;
        let mut min_s = vec![i64::MAX / 2; k];
        // 为了适配从数组开头开始的合法子数组
        min_s[k - 1] = 0;
        let mut s = 0i64;
        let mut ans = i64::MIN;

        for (j, &x) in nums.iter().enumerate() {
            s += x as i64;
            let i = j % k;
            ans = ans.max(s - min_s[i]);
            min_s[i] = s.min(min_s[i]);
        }
        ans
    }
}

// 2488. 统计中位数为 K 的子数组
pub mod n2488 {
    pub fn count_subarrays(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let mut s = n as i32;
        let mut ans = 0i32;
        let mut cnt = vec![0i32; n * 2];
        cnt[n] = 1;
        let mut found_k = false;

        for &x in &nums {
            if x == k {
                found_k = true;
            } else if x < k {
                s -= 1;
            } else {
                s += 1;
            }

            if !found_k {
                cnt[s as usize] += 1;
            } else {
                ans += cnt[s as usize] + cnt[(s - 1) as usize];
            }
        }
        ans
    }
}

// 1590. 使数组和能被 P 整除
pub mod n1590 {
    pub fn min_subarray(nums: Vec<i32>, p: i32) -> i32 {
        let total_sum: i64 = nums.iter().map(|&v| v as i64).sum();
        let x = total_sum.rem_euclid(p as i64) as i32;
        if x == 0 {
            return 0;
        }

        let n = nums.len() as i32;
        let mut ans = n;
        let mut s: i64 = 0;
        // 由于下面 i 是从 0 开始的，前缀和下标就要从 -1 开始了
        let mut last = std::collections::HashMap::new();
        last.insert(0, -1);

        for (i, &v) in nums.iter().enumerate() {
            let idx = i as i32;
            s += v as i64;
            last.insert(s.rem_euclid(p as i64) as i32, idx);
            // 如果不存在，-n 可以保证 i-j >= n
            let target = (s - x as i64).rem_euclid(p as i64) as i32;
            let j = *last.get(&target).unwrap_or(&(-n));
            ans = ans.min(idx - j);
        }

        if ans < n { ans } else { -1 }
    }
}

// 2845. 统计趣味子数组的数目
pub mod n2845 {
    pub fn count_interesting_subarrays(nums: Vec<i32>, modulo: i32, k: i32) -> i64 {
        let mut cnt = std::collections::HashMap::from([(0, 1)]); // s[0]=0
        let mut ans = 0i64;
        let mut s = 0i32;

        for x in nums {
            if x % modulo == k {
                s += 1;
            }
            let rem = (s - k).rem_euclid(modulo);
            ans += cnt.get(&rem).unwrap_or(&0);
            *cnt.entry(s.rem_euclid(modulo)).or_insert(0) += 1;
        }
        ans
    }
}

// 3739. 统计主要元素子数组数目 II
pub mod n3739 {
    pub fn count_majority_subarrays(nums: Vec<i32>, target: i32) -> i64 {
        let n = nums.len();
        let mut s = n; // 为避免下标越界，把 s 初始化成 n
        let mut cnt = vec![0; 2 * n + 1];
        cnt[s] = 1;

        let mut ans = 0i64;
        let mut f = 0i64;

        for &x in &nums {
            if x == target {
                f += cnt[s];
                s += 1;
            } else {
                s -= 1;
                f -= cnt[s];
            }
            ans += f;
            cnt[s] += 1;
        }

        ans
    }
}

// 3900. 一次交换后的最长平衡子串
pub mod n3900 {
    pub fn longest_balanced(s: String) -> i32 {
        // 统计 0 和 1 的总数量
        let total0 = s.bytes().filter(|&b| b == b'0').count() as i32;
        let total1 = s.len() as i32 - total0;

        // 哈希表：key = 前缀和，value = 存储最早/次早的索引
        let mut pos = std::collections::HashMap::from([(0, vec![-1])]); // // 初始条件：前缀和 0 出现在索引 -1

        let mut ans = 0;
        let mut pre_sum = 0; // 前缀和：1→+1，0→-1

        // 遍历字符串
        for (i, ch) in s.bytes().enumerate() {
            let i = i as i32;
            // 更新前缀和
            pre_sum += if ch == b'1' { 1 } else { -1 };

            // 仅保存前两个出现的索引
            let indices = pos.entry(pre_sum).or_insert_with(Vec::new);
            if indices.len() < 2 {
                indices.push(i);
            }

            // 情况1：不交换，直接取最早出现的位置计算长度
            let first_pos = indices[0];
            ans = ans.max(i - first_pos);

            // 情况2：交换子串内1 和 子串外0 → 查找 pre_sum - 2
            if let Some(p) = pos.get(&(pre_sum - 2)) {
                let p0 = p[0];
                let condition = (i - p0 - 2) / 2 < total0;
                if condition {
                    ans = ans.max(i - p0);
                } else if p.len() > 1 {
                    ans = ans.max(i - p[1]);
                }
            }

            // 情况3：交换子串内0 和 子串外1 → 查找 pre_sum + 2
            if let Some(p) = pos.get(&(pre_sum + 2)) {
                let p0 = p[0];
                let condition = (i - p0 - 2) / 2 < total1;
                if condition {
                    ans = ans.max(i - p0);
                } else if p.len() > 1 {
                    ans = ans.max(i - p[1]);
                }
            }
        }

        ans
    }
}

// 1074. 元素和为目标值的子矩阵数量
pub mod n1074 {
    pub fn num_submatrix_sum_target(matrix: Vec<Vec<i32>>, target: i32) -> i32 {
        fn subarray_sum(nums: &[i32], k: i32) -> i32 {
            let mut ans = 0;
            let mut s = 0; // 当前前缀和
            let mut cnt = std::collections::HashMap::with_capacity(nums.len());
            for x in nums {
                // 先记录当前前缀和出现次数
                *cnt.entry(s).or_insert(0) += 1;
                s += x;
                // 查找 s-k 前缀和存在个数，即为合法子数组数
                if let Some(&c) = cnt.get(&(s - k)) {
                    ans += c;
                }
            }
            ans
        }

        let col_len = matrix[0].len();
        let mut ans = 0;

        // 枚举子矩阵上边界
        for top in 0..matrix.len() {
            // 列累加和数组，初始全0
            let mut col_sum = vec![0; col_len];
            // 向下扩展下边界，逐行叠加列值
            for row in &matrix[top..] {
                for (col_idx, &val) in row.iter().enumerate() {
                    col_sum[col_idx] += val;
                }
                // 一维数组统计合法子数组，等价子矩阵
                ans += subarray_sum(&col_sum, target);
            }
        }
        ans
    }
}

// 1442. 形成两个异或相等数组的三元组数目
pub mod n1442 {
    pub fn count_triplets(arr: Vec<i32>) -> i32 {
        let mut cnt = std::collections::HashMap::new();
        let mut total = std::collections::HashMap::new();
        let mut ans = 0;
        let mut s = 0;

        for (k, &val) in arr.iter().enumerate() {
            let t = s ^ val;
            if let Some(&c) = cnt.get(&t) {
                let sum_k = total[&t];
                ans += c * k as i32 - sum_k;
            }

            *cnt.entry(s).or_insert(0) += 1;
            *total.entry(s).or_insert(0) += k as i32;
            s = t;
        }

        ans
    }
}

// 3714. 最长的平衡子串 II
pub mod n3714 {
    pub fn longest_balanced(s: &str) -> i32 {
        let s_chars: Vec<char> = s.chars().collect();
        let n = s_chars.len();
        let mut ans = 0;

        // 1. 只包含一种字母的最长连续子串
        let mut i = 0;
        while i < n {
            let start = i;
            i += 1;
            while i < n && s_chars[i] == s_chars[i - 1] {
                i += 1;
            }
            ans = ans.max((i - start) as i32);
        }

        // 2. 处理两种字母组合: x, y
        fn two_char(s: &[char], x: char, y: char, ans: &mut i32) {
            let n = s.len();
            let mut i = 0;
            while i < n {
                let mut pos = std::collections::HashMap::new();
                pos.insert(0, i as i32 - 1); // 前缀和初始0，位置 i-1
                let mut d = 0; // x数量 - y数量

                while i < n && (s[i] == x || s[i] == y) {
                    if s[i] == x {
                        d += 1;
                    } else {
                        d -= 1;
                    }

                    if let Some(&pre_idx) = pos.get(&d) {
                        *ans = (*ans).max(i as i32 - pre_idx);
                    } else {
                        pos.insert(d, i as i32);
                    }
                    i += 1;
                }
                i += 1;
            }
        }

        // 枚举所有两两组合
        two_char(&s_chars, 'a', 'b', &mut ans);
        two_char(&s_chars, 'a', 'c', &mut ans);
        two_char(&s_chars, 'b', 'c', &mut ans);

        // 3. 处理三种字母 a/b/c 共存的情况
        let mut pos = std::collections::HashMap::new();
        pos.insert((0, 0), -1); // 初始状态 (0,0) 位于下标 -1
        let (mut cnt_a, mut cnt_b, mut cnt_c) = (0, 0, 0);

        for (idx, &ch) in s_chars.iter().enumerate() {
            match ch {
                'a' => cnt_a += 1,
                'b' => cnt_b += 1,
                'c' => cnt_c += 1,
                _ => (), // 题目仅 a/b/c，忽略其他
            }
            // 状态: (a-b, b-c)
            let state = (cnt_a - cnt_b, cnt_b - cnt_c);
            if let Some(&pre_idx) = pos.get(&state) {
                ans = ans.max(idx as i32 - pre_idx);
            } else {
                pos.insert(state, idx as i32);
            }
        }

        ans
    }
}

// 2025. 分割数组的最多方案数
pub mod n2025 {
    pub fn ways_to_partition(nums: &[i32], k: i32) -> i32 {
        let n = nums.len();
        let mut sum = vec![0; n];
        sum[0] = nums[0];

        let mut cnt_r = std::collections::HashMap::new();
        for i in 1..n {
            sum[i] = sum[i - 1] + nums[i];
            *cnt_r.entry(sum[i - 1]).or_insert(0) += 1;
        }

        let tot = sum[n - 1];
        let mut ans = if tot % 2 == 0 {
            *cnt_r.get(&(tot / 2)).unwrap_or(&0)
        } else {
            0
        };

        let mut cnt_l = std::collections::HashMap::new();
        for (i, &s) in sum.iter().enumerate() {
            let d = k - nums[i];
            if (tot + d) % 2 == 0 {
                let left = *cnt_l.get(&((tot + d) / 2)).unwrap_or(&0);
                let right = *cnt_r.get(&((tot - d) / 2)).unwrap_or(&0);
                ans = ans.max(left + right);
            }

            // 迁移计数：左表增加，右表减少
            *cnt_l.entry(s).or_insert(0) += 1;
            if let Some(v) = cnt_r.get_mut(&s) {
                *v -= 1;
            }
        }

        ans
    }
}

// 3729. 统计有序数组中可被 K 整除的子数组数量
pub mod n3729 {
    pub fn num_good_subarrays(nums: Vec<i32>, k: i32) -> i64 {
        let k = k as i64;
        let mut cnt = std::collections::HashMap::from([(0, 1)]);
        let mut pre_sum = 0i64; // 前缀和
        let mut last_start = 0usize; // 上一个连续相同段的起始下标
        let mut ans = 0i64;

        for (i, &x) in nums.iter().enumerate() {
            if i > 0 && x != nums[i - 1] {
                // 上一个连续相同段结束，可以把上一段对应的前缀和添加到 cnt
                let v = nums[i - 1] as i64;
                let mut s = pre_sum;
                for _ in last_start..i {
                    *cnt.entry(s.rem_euclid(k)).or_insert(0) += 1;
                    s -= v;
                }
                last_start = i;
            }

            pre_sum += x as i64;
            let rem = pre_sum.rem_euclid(k);
            ans += cnt.get(&rem).unwrap_or(&0);
        }

        ans
    }
}

// 2949. 统计美丽子字符串 II
pub mod n2949 {
    pub fn beautiful_substrings(s: String, k: i32) -> i32 {
        // 对 n 做特殊因数分解求值
        fn calc_factor(mut n: i32) -> i32 {
            let mut res = 1;
            let mut i = 2;
            while i * i <= n {
                let i2 = i * i;
                while n % i2 == 0 {
                    res *= i;
                    n /= i2;
                }
                if n % i == 0 {
                    res *= i;
                    n /= i;
                }
                i += 1;
            }
            if n > 1 {
                res *= n;
            }
            res
        }

        let k = calc_factor(k * 4);
        let mut cnt = std::collections::HashMap::from([((k - 1, 0), 1)]);

        let mut ans = 0;
        let mut pre_sum = 0;
        let vowels = ['a', 'e', 'i', 'o', 'u'];

        for (i, c) in s.chars().enumerate() {
            if vowels.contains(&c) {
                pre_sum += 1;
            } else {
                pre_sum -= 1;
            }
            let p = ((i as i32).rem_euclid(k), pre_sum);
            // 累加之前出现的次数
            ans += *cnt.get(&p).unwrap_or(&0);
            // 当前键计数 +1
            *cnt.entry(p).or_insert(0) += 1;
        }

        ans
    }
}

// 3364. 最小正和子数组
pub mod n3364 {
    pub fn minimum_sum_subarray(nums: Vec<i32>, l: i32, r: i32) -> i32 {
        let l = l as usize;
        let r = r as usize;
        let n = nums.len();

        // 前缀和数组
        let mut prefix = vec![0; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] + nums[i];
        }

        let mut ans = i32::MAX;
        let mut map = std::collections::BTreeMap::new(); // 关键：用 Map 计数，支持重复值

        // 遍历右端点 j
        for j in l..prefix.len() {
            // 加入 s[j-l]，维护窗口左边界
            *map.entry(prefix[j - l]).or_insert(0) += 1;

            // 二分查找：找到 <= prefix[j] 的最大值
            if let Some((&val, _)) = map.range(..prefix[j]).next_back() {
                ans = ans.min(prefix[j] - val);
            }

            // 超过 r 长度时，移除最左侧元素 s[j-r]
            if j >= r {
                let key = prefix[j - r];
                *map.get_mut(&key).unwrap() -= 1;
                if map[&key] == 0 {
                    map.remove(&key);
                }
            }
        }

        // 无结果返回 -1
        if ans == i32::MAX { -1 } else { ans }
    }
}

// 363. 矩形区域不超过 K 的最大数值和
pub mod n363 {
    pub fn max_sum_submatrix(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        if m == 0 {
            return i32::MIN;
        }
        let n = matrix[0].len();
        let mut ans = i32::MIN;

        // 枚举上边界
        for i in 0..m {
            let mut total = vec![0; n];
            // 枚举下边界
            for matrix_item in matrix.iter().take(m).skip(i) {
                // 逐列累加，压缩成一维数组
                for c in 0..n {
                    total[c] += matrix_item[c];
                }

                let mut set = std::collections::BTreeSet::new();
                set.insert(0);
                let mut s = 0;

                for &v in &total {
                    s += v;
                    // 找第一个 >= (s - k) 的前缀和
                    let target = s - k;
                    if let Some(&pre) = set.range(target..).next() {
                        ans = ans.max(s - pre);
                    }
                    set.insert(s);
                }
            }
        }

        ans
    }
}

// 437. 路径总和 III
pub mod n437 {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    //  Definition for a binary tree node.
    #[derive(Debug, PartialEq, Eq)]
    pub struct TreeNode {
        pub val: i32,
        pub left: Option<Rc<RefCell<TreeNode>>>,
        pub right: Option<Rc<RefCell<TreeNode>>>,
    }

    impl TreeNode {
        #[inline]
        pub fn new(val: i32) -> Self {
            TreeNode {
                val,
                left: None,
                right: None,
            }
        }
    }

    pub fn path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> i32 {
        // s 表示从根到 node 的父节点的节点值之和（node 的节点值尚未计入）
        fn dfs(
            node: &Option<Rc<RefCell<TreeNode>>>,
            s: i64,
            target_sum: i64,
            ans: &mut i32,
            cnt: &mut HashMap<i64, i32>,
        ) {
            if let Some(node) = node {
                let node = node.borrow();

                let s = s + node.val as i64;
                // 把 node 当作路径的终点，统计有多少个起点
                *ans += *cnt.get(&(s - target_sum)).unwrap_or(&0);

                *cnt.entry(s).or_insert(0) += 1; // cnt[s] += 1
                dfs(&node.left, s, target_sum, ans, cnt);
                dfs(&node.right, s, target_sum, ans, cnt);
                *cnt.entry(s).or_insert(0) -= 1; // 恢复现场（撤销 cnt[s] += 1）
            }
        }

        // key：从根到 node 的节点值之和
        // value：节点值之和的出现次数
        // 注意在递归过程中，哈希表只保存根到 node 的路径的前缀的节点值之和
        let mut cnt = HashMap::new();
        cnt.insert(0, 1);
        let mut ans = 0;
        dfs(&root, 0, target_sum as i64, &mut ans, &mut cnt);
        ans
    }
}

// 1685. 有序数组中差绝对值之和
pub mod n1685 {
    pub fn get_sum_absolute_differences(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        // 前缀和 pre_sum[0]=0, pre_sum[1]=nums[0], pre_sum[k]=sum(nums[0..k-1])
        let mut pre_sum = vec![0; n + 1];
        for i in 0..n {
            pre_sum[i + 1] = pre_sum[i] + nums[i];
        }

        let mut res = Vec::with_capacity(n);
        for (i, &num) in nums.iter().enumerate() {
            let left = i as i32 * num - pre_sum[i];
            let right = pre_sum[n] - pre_sum[i] - (n - i) as i32 * num;
            res.push(left + right);
        }
        res
    }
}
