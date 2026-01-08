// 34. 在排序数组中查找元素的第一个和最后一个位置
pub mod n34 {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        fn custom_partition_point<T, P>(slice: &[T], mut pred: P) -> usize
        where
            P: FnMut(&T) -> bool,
        {
            let mut size = slice.len();
            let (mut left, mut right) = (0, size); // 左闭右开区间 [left, right)
            while left < right {
                // 循环不变量：
                // pred(&slice[left-1]) 为 true
                // pred(&slice[right]) 为 false
                let mid = left + size / 2;
                if pred(&slice[mid]) {
                    left = mid + 1; // 范围缩小到 [mid+1, right)
                } else {
                    right = mid; // 范围缩小到 [left, mid)
                }
                size = right - left;
            }

            // 循环结束后 left = right
            // 此时 pred(&slice[left-1]) 为 true 而 pred(&slice[right])、pred(&slice[right]) 为 false
            // 所以 lefth/right 就是第一个 pred 为 false 的元素下标

            right // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
        }

        let start = custom_partition_point(&nums, |&x| x < target);
        if start == nums.len() || nums[start] != target {
            return vec![-1, -1];
        }
        let end = custom_partition_point(&nums, |&x| x <= target) - 1;
        vec![start as i32, end as i32]
    }
}

// 35. 搜索插入位置
pub mod n35 {
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len(); // 为避免出现负数，使用左闭右开区间 [left, right) 是最方便的
        while left < right {
            // 循环不变量：
            // nums[left-1] < target
            // nums[right] >= target
            let mid = left + (right - left) / 2; // 为了防止溢出，不使用 （left + right） / 2
            if nums[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        // 循环结束后 left = right
        // 此时 nums[left-1] < target 而 nums[left] = nums[right] >= target
        // 所以 lefth/right 就是第一个 >= target 的元素下标

        right as _ // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
    }
}

// 704. 二分查找
pub mod n704 {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        fn custom_partition_point<T, P>(slice: &[T], mut pred: P) -> usize
        where
            P: FnMut(&T) -> bool,
        {
            let mut size = slice.len();
            let (mut left, mut right) = (0, size); // 左闭右开区间 [left, right)
            while left < right {
                // 循环不变量：
                // pred(&slice[left-1]) 为 true
                // pred(&slice[right]) 为 false
                let mid = left + size / 2;
                if pred(&slice[mid]) {
                    left = mid + 1; // 范围缩小到 [mid+1, right)
                } else {
                    right = mid; // 范围缩小到 [left, mid)
                }
                size = right - left;
            }

            // 循环结束后 left = right
            // 此时 pred(&slice[left-1]) 为 true 而 pred(&slice[right])、pred(&slice[right]) 为 false
            // 所以 lefth/right 就是第一个 pred 为 false 的元素下标

            right // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
        }

        let index = custom_partition_point(&nums, |&x| x < target);
        if index < nums.len() && nums[index] == target {
            index as _
        } else {
            -1
        }
    }
}

// 744. 寻找比目标字母大的最小字母
pub mod n744 {
    pub fn next_greatest_letter(letters: Vec<char>, target: char) -> char {
        if letters.len() > 0 && letters.last().unwrap() <= &target {
            return letters[0];
        }
        let (mut left, mut right) = (0, letters.len()); // 左闭右开区间 [left, right)
        while left < right {
            let mid = left + (right - left) / 2; // 为了防止溢出，不使用 （left + right） / 2
            if letters[mid] > target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        letters[right]
    }
}

// 2529. 正整数和负整数的最大计数
pub mod n2529 {
    pub fn maximum_count(nums: Vec<i32>) -> i32 {
        fn custom_partition_point<T, P>(slice: &[T], mut pred: P) -> usize
        where
            P: FnMut(&T) -> bool,
        {
            let mut size = slice.len();
            let (mut left, mut right) = (0, size); // 左闭右开区间 [left, right)
            while left < right {
                // 循环不变量：
                // pred(&slice[left-1]) 为 true
                // pred(&slice[right]) 为 false
                let mid = left + size / 2;
                if pred(&slice[mid]) {
                    left = mid + 1; // 范围缩小到 [mid+1, right)
                } else {
                    right = mid; // 范围缩小到 [left, mid)
                }
                size = right - left;
            }

            // 循环结束后 left = right
            // 此时 pred(&slice[left-1]) 为 true 而 pred(&slice[right])、pred(&slice[right]) 为 false
            // 所以 lefth/right 就是第一个 pred 为 false 的元素下标

            right // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
        }

        let neg = custom_partition_point(&nums, |&x| x < 0);
        let pos = nums.len() - custom_partition_point(&nums, |&x| x < 1);
        neg.max(pos) as _
    }
}

// 2389. 和有限的最长子序列
pub mod n2389 {
    pub fn answer_queries(nums: Vec<i32>, queries: Vec<i32>) -> Vec<i32> {
        fn custom_partition_point<T, P>(slice: &[T], mut pred: P) -> usize
        where
            P: FnMut(&T) -> bool,
        {
            let mut size = slice.len();
            let (mut left, mut right) = (0, size); // 左闭右开区间 [left, right)
            while left < right {
                // 循环不变量：
                // pred(&slice[left-1]) 为 true
                // pred(&slice[right]) 为 false
                let mid = left + size / 2;
                if pred(&slice[mid]) {
                    left = mid + 1; // 范围缩小到 [mid+1, right)
                } else {
                    right = mid; // 范围缩小到 [left, mid)
                }
                size = right - left;
            }

            // 循环结束后 left = right
            // 此时 pred(&slice[left-1]) 为 true 而 pred(&slice[right])、pred(&slice[right]) 为 false
            // 所以 lefth/right 就是第一个 pred 为 false 的元素下标

            right // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
        }

        // 1. 对 nums 进行排序
        let mut sorted_nums = nums;
        sorted_nums.sort_unstable(); // 不稳定排序，效率更高（若需稳定排序用 sort()）

        // 2. 原地计算前缀和
        for i in 1..sorted_nums.len() {
            sorted_nums[i] += sorted_nums[i - 1];
        }

        // 3. 对每个 query 执行二分查找，生成结果
        queries
            .into_iter()
            .map(|q| {
                let pos = custom_partition_point(&sorted_nums, |&x| x <= q);
                pos as i32
            })
            .collect()
    }
}

// 1170. 比较字符串最小字母出现频次
pub mod n1170 {
    pub fn num_smaller_by_frequency(queries: Vec<String>, words: Vec<String>) -> Vec<i32> {
        // 计算字符串中最小字符的出现频率
        fn f(s: &str) -> usize {
            let chars: Vec<char> = s.chars().collect();
            let min_char = *chars.iter().min().unwrap();
            chars.iter().filter(|&&c| c == min_char).count()
        }
        // 1. 计算所有 words 的 f值并排序
        let mut words_freqs: Vec<usize> = words.iter().map(|w| f(w)).collect();
        words_freqs.sort_unstable();

        // 2. 遍历每个 query，二分查找统计结果
        queries
            .iter()
            .map(|q| {
                let q_freq = f(q);
                // 找到第一个 > q_freq 的位置，后面的元素数量即为答案
                let cnt = words_freqs.partition_point(|&x| x <= q_freq);
                (words_freqs.len() - cnt) as i32
            })
            .collect()
    }
}

// 2300. 咒语和药水的成功对数
pub mod n2300 {
    pub fn successful_pairs(mut spells: Vec<i32>, mut potions: Vec<i32>, success: i64) -> Vec<i32> {
        potions.sort_unstable();
        let potion_len = potions.len();
        let last = potions[potion_len - 1] as i64;

        for spell in spells.iter_mut() {
            let target = (success + *spell as i64 - 1) / *spell as i64; // success / spell 向上取整
            // 防止 i64 转成 i32 截断（这样不需要把 potions 中的数转成 i64 比较）
            if last >= target {
                let idx = potions.partition_point(|&x| x < target as i32);
                *spell = (potion_len - idx) as i32;
            } else {
                *spell = 0;
            }
        }

        spells
    }
}

// 2080. 区间内查询数字的频率
pub mod n2080 {
    #![allow(dead_code)]
    use std::collections::HashMap;

    struct RangeFreqQuery {
        pos: HashMap<i32, Vec<usize>>,
    }

    impl RangeFreqQuery {
        fn new(arr: Vec<i32>) -> Self {
            let mut pos: HashMap<i32, Vec<usize>> = HashMap::new();
            for (i, &x) in arr.iter().enumerate() {
                pos.entry(x).or_default().push(i);
            }
            Self { pos }
        }

        fn query(&self, left: i32, right: i32, value: i32) -> i32 {
            if let Some(a) = self.pos.get(&value) {
                let p = a.partition_point(|&i| i < left as usize);
                let q = a.partition_point(|&i| i <= right as usize);
                (q - p) as _
            } else {
                0
            }
        }
    }
}

// 3488. 距离最小相等元素查询
pub mod n3488 {
    use std::collections::HashMap;

    pub fn solve_queries(nums: Vec<i32>, queries: Vec<i32>) -> Vec<i32> {
        // 1. 构建数字到索引列表的哈希映射
        let mut indices: HashMap<i32, Vec<i32>> = HashMap::new();
        for (idx, &num) in nums.iter().enumerate() {
            indices.entry(num).or_insert_with(Vec::new).push(idx as i32);
        }

        let n = nums.len() as i32;
        // 2. 为每个索引列表添加哨兵值
        for p in indices.values_mut() {
            if p.is_empty() {
                continue;
            }
            let front_sentinel = p[p.len() - 1] - n; // 前向哨兵：为了方便计算“索引列表第一个元素往左绕到最后一个元素的距离”
            let back_sentinel = p[0] + n; // 后向哨兵：为了方便计算“索引列表最后一个元素往右绕到第一个元素的距离”
            p.insert(0, front_sentinel);
            p.push(back_sentinel);
        }

        // 3. 处理查询
        let mut result = Vec::with_capacity(queries.len());
        for &i in &queries {
            let num = nums[i as usize];
            let p = indices.get(&num).unwrap(); // 必然存在有效索引列表

            let res = if p.len() == 3 {
                // 仅1个真实元素（2个哨兵+1个索引），返回-1
                -1
            } else {
                let j = p.partition_point(|&x| x < i);

                // 计算前后最小距离
                let dist_prev = i - p[j - 1];
                let dist_next = p[j + 1] - i;
                dist_prev.min(dist_next)
            };

            result.push(res);
        }

        result
    }
}

// 2563. 统计公平数对的数目
pub mod n2563 {
    pub fn count_fair_pairs(mut nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        nums.sort_unstable();
        let mut ans = 0;
        for j in 0..nums.len() {
            let l = nums[..j].partition_point(|&x| x < lower - nums[j]);
            let r = nums[..j].partition_point(|&x| x <= upper - nums[j]);
            ans += r - l;
        }
        ans as _
    }
}

// 2070. 每一个查询的最大美丽值
pub mod n2070 {
    pub fn maximum_beauty(mut items: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        items.sort_unstable_by_key(|item| item[0]);
        items.dedup_by(|current, prev| current[1] <= prev[1]); // 去掉无用数据
        queries
            .into_iter()
            .map(|q| {
                let j = items.partition_point(|item| item[0] <= q);
                if j > 0 { items[j - 1][1] } else { 0 }
            })
            .collect()
    }
}

// 1146. 快照数组
pub mod n1146 {
    #![allow(dead_code)]
    use std::collections::HashMap;

    struct SnapshotArray {
        cur_snap_id: i32,
        history: HashMap<i32, Vec<(i32, i32)>>, // 每个 index 的历史修改记录
    }

    impl SnapshotArray {
        fn new(_: i32) -> Self {
            Self {
                cur_snap_id: 0,
                history: HashMap::new(),
            }
        }

        fn set(&mut self, index: i32, val: i32) {
            self.history
                .entry(index)
                .or_default()
                .push((self.cur_snap_id, val));
        }

        fn snap(&mut self) -> i32 {
            self.cur_snap_id += 1;
            self.cur_snap_id - 1
        }

        fn get(&self, index: i32, snap_id: i32) -> i32 {
            if let Some(h) = self.history.get(&index) {
                let j = h.partition_point(|&(id, _)| id <= snap_id);
                if j > 0 {
                    return h[j - 1].1;
                }
            }
            0
        }
    }
}

// 1818. 绝对差值和
pub mod n1818 {
    const MOD: i64 = 1_000_000_007;

    pub fn min_absolute_sum_diff(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        let mut rec = nums1.clone();
        rec.sort_unstable();

        let mut sum = 0i64;
        let mut maxn = 0i64;
        let n = nums1.len();

        for i in 0..n {
            let diff = (nums1[i] - nums2[i]).abs() as i64;
            sum = (sum + diff) % MOD;

            // 找第一个 >= nums2[i] 的位置
            let j = rec.partition_point(|&x| x < nums2[i]);

            // 检查右侧元素
            if j < n {
                let new_diff = (rec[j] - nums2[i]).abs() as i64;
                maxn = maxn.max(diff - new_diff);
            }

            // 检查左侧元素
            if j > 0 {
                let new_diff = (nums2[i] - rec[j - 1]).abs() as i64;
                maxn = maxn.max(diff - new_diff);
            }
        }

        ((sum - maxn + MOD) % MOD) as i32
    }
}

// 1287. 有序数组中出现次数超过25%的元素
pub mod n1287 {
    pub fn find_special_integer(arr: Vec<i32>) -> i32 {
        let m = arr.len() / 4;
        for i in [m, m * 2 + 1] {
            let x = arr[i];
            let j = arr.partition_point(|&y| y < x);
            // 因为题目保证有解，j+m 不会下标越界
            if arr[j + m] == x {
                return x;
            }
        }
        // 如果答案不是 arr[m] 也不是 arr[2m+1]，那么答案一定是 arr[3m+2]
        arr[m * 3 + 2]
    }
}

// 1283. 使结果不超过阈值的最小除数
pub mod n1283 {
    pub fn smallest_divisor(nums: Vec<i32>, threshold: i32) -> i32 {
        let check = |&m: &i32| -> bool {
            let mut sum = 0;
            for &x in &nums {
                sum += (x + m - 1) / m; // x / m 向上取整
                if sum > threshold {
                    // 不满足除数条件，返回 true
                    return true;
                }
            }
            // 满足除数条件，返回 false
            false
        };

        let max_num = *nums.iter().max().unwrap();
        // partition_point 找第一个不满足 check 的位置，即第一个满足除数条件的 m 的位置
        let index = (1..max_num + 1)
            .collect::<Vec<i32>>()
            .partition_point(check);

        // (1..max_num + 1) 在索引 index 处的值
        index as i32 + 1
    }
}

// 2187. 完成旅途的最少时间
pub mod n2187 {
    pub fn minimum_time(time: Vec<i32>, total_trips: i32) -> i64 {
        let total_trips = total_trips as i64;
        let min_t = *time.iter().min().unwrap() as i64;
        let mut left: i64 = min_t;
        let mut right: i64 = min_t * total_trips;

        while left < right {
            let mid = left + (right - left) / 2;
            let mut sum = 0;
            for &t in &time {
                sum += mid / t as i64;
            }
            if sum < total_trips {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}
