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
