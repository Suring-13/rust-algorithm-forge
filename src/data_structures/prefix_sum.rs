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
