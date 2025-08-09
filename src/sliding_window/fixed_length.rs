// 1456. 定长子串中元音的最大数目
pub mod n1456 {
    pub fn max_vowels(s: String, k: i32) -> i32 {
        fn is_vowel(ch: char) -> i32 {
            match ch {
                'a' | 'e' | 'i' | 'o' | 'u' => 1,
                _ => 0,
            }
        }

        let k = k as usize;
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();

        let mut vowel_count: i32 = chars.iter().take(k).map(|&ch| is_vowel(ch)).sum();
        let mut ans = vowel_count;

        for i in k..n {
            vowel_count += is_vowel(chars[i]) - is_vowel(chars[i - k]);
            ans = ans.max(vowel_count);
        }

        ans
    }
}

// 643. 子数组最大平均数 I
pub mod n643 {
    pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
        let k = k as usize;
        let n = nums.len();

        let mut sum: i32 = nums.iter().take(k).sum();
        let mut max_sum = sum;

        for i in k..n {
            sum += nums[i] - nums[i - k];
            max_sum = max_sum.max(sum);
        }

        max_sum as f64 / k as f64
    }
}

// 1343. 大小为 K 且平均值大于等于阈值的子数组数目
pub mod n1343 {
    pub fn num_of_subarrays(arr: Vec<i32>, k: i32, threshold: i32) -> i32 {
        let k = k as usize;
        let n = arr.len();

        let mut ans = 0;
        let mut sum: i32 = arr.iter().take(k).sum();
        if sum >= k as i32 * threshold && n >= k {
            ans += 1;
        }

        for i in k..n {
            sum += arr[i] - arr[i - k];
            if sum >= k as i32 * threshold {
                ans += 1;
            }
        }

        ans
    }
}

// 2090. 半径为 k 的子数组平均值
pub mod n2090 {
    pub fn get_averages(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let window_size = 2 * k + 1; // 窗口大小
        let n = nums.len();
        let mut avgs = vec![-1; n];

        // 如果数组长度小于窗口大小，直接返回全-1的结果
        if n < window_size {
            return avgs;
        }

        // 计算初始窗口的和
        let mut sum: i64 = nums.iter().take(window_size).map(|&x| x as i64).sum();
        avgs[k] = (sum / window_size as i64) as i32;

        // 滑动窗口计算后续平均值
        for i in window_size..n {
            sum += (nums[i] - nums[i - window_size]) as i64;
            avgs[i - k] = (sum / window_size as i64) as i32;
        }

        avgs
    }
}

// 2379. 得到 K 个黑块的最少涂色次数
pub mod n2379 {
    pub fn minimum_recolors(blocks: String, k: i32) -> i32 {
        fn is_w(ch: char) -> i32 {
            match ch {
                'W' => 1,
                _ => 0,
            }
        }

        let chars: Vec<char> = blocks.chars().collect();
        let n = chars.len();
        let k = k as usize;

        if n < k {
            return -1;
        }

        let mut vowel_count: i32 = chars.iter().take(k).map(|&ch| is_w(ch)).sum();
        let mut ans = vowel_count;

        for i in k..n {
            vowel_count += is_w(chars[i]) - is_w(chars[i - k]);
            ans = ans.min(vowel_count);
        }

        ans
    }
}

// 2841. 几乎唯一子数组的最大和
pub mod n2841 {
    use std::collections::HashMap;

    pub fn max_sum(nums: Vec<i32>, m: i32, k: i32) -> i64 {
        let n = nums.len();
        let m = m as usize;
        let k = k as usize;
        let mut ans = 0;
        if n < k {
            return ans;
        }
        let mut sum = 0;
        let mut count = HashMap::new();

        nums.iter().take(k).for_each(|&x| {
            sum += x as i64;
            *count.entry(x).or_insert(0) += 1;
        });

        if count.len() >= m {
            ans = sum;
        }

        for i in k..n {
            sum += nums[i] as i64 - nums[i - k] as i64;
            *count.entry(nums[i]).or_insert(0) += 1;
            let c = count.entry(nums[i - k]).or_insert(0);
            *c -= 1;
            if *c == 0 {
                count.remove(&nums[i - k]);
            }

            if count.len() >= m {
                ans = ans.max(sum);
            }
        }

        ans
    }
}

// 2461. 长度为 K 子数组中的最大和
pub mod n2461 {
    use std::collections::HashMap;

    pub fn maximum_subarray_sum(nums: Vec<i32>, k: i32) -> i64 {
        let n = nums.len();
        let k = k as usize;
        let mut ans = 0;
        if n < k {
            return ans;
        }
        let mut sum = 0;
        let mut count = HashMap::new();

        nums.iter().take(k).for_each(|&x| {
            sum += x as i64;
            *count.entry(x).or_insert(0) += 1;
        });

        if count.len() == k {
            ans = sum;
        }

        for i in k..n {
            sum += nums[i] as i64 - nums[i - k] as i64;
            *count.entry(nums[i]).or_insert(0) += 1;
            let c = count.entry(nums[i - k]).or_insert(0);
            *c -= 1;
            if *c == 0 {
                count.remove(&nums[i - k]);
            }

            if count.len() == k {
                ans = ans.max(sum);
            }
        }

        ans
    }
}

// 1423. 可获得的最大点数
pub mod n1423 {
    pub fn max_score(card_points: Vec<i32>, k: i32) -> i32 {
        let n = card_points.len();
        if n < k as usize {
            return 0;
        }
        let window_size = n - k as usize;
        let mut sum = card_points.iter().take(window_size).sum::<i32>();
        let mut min_sum = sum;
        for i in window_size..n {
            sum += card_points[i] - card_points[i - window_size];
            min_sum = min_sum.min(sum);
        }
        card_points.iter().sum::<i32>() - min_sum
    }
}

// 1052. 爱生气的书店老板
pub mod n1052 {
    pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, minutes: i32) -> i32 {
        let window_size = minutes as usize;
        let mut s = [0, 0]; // s[0] 老板不生气时的顾客数量，s[1] 老板生气时的顾客数量
        let mut max_s1 = 0;

        customers
            .iter()
            .zip(grumpy.iter())
            .take(window_size)
            .for_each(|(&c, &g)| {
                s[g as usize] += c;
            });
        if customers.len() < window_size {
            return s[0] + s[1];
        } else {
            max_s1 = s[1];
        }

        for (i, (&c, &g)) in customers
            .iter()
            .zip(grumpy.iter())
            .enumerate()
            .skip(window_size)
        {
            s[g as usize] += c;
            if grumpy[i - window_size] == 1 {
                s[1] -= customers[i - window_size];
            }
            max_s1 = max_s1.max(s[1]);
        }

        s[0] + max_s1
    }
}
