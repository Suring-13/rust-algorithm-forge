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
