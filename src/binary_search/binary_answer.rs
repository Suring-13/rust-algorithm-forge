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

// 1011. 在 D 天内送达包裹的能力
pub mod n1011 {
    pub fn ship_within_days(weights: Vec<i32>, days: i32) -> i32 {
        // 确定二分查找的左右边界：左边界为最大包裹重量，右边界为总重量
        let mut left = *weights.iter().max().unwrap();
        let mut right = weights.iter().sum::<i32>();

        while left < right {
            let mid = left + (right - left) / 2; // 等价于 (left+right)/2，避免溢出
            let mut need_days = 1;
            let mut current_load = 0;

            // 计算当前运载能力 mid 下，需要的天数
            for &w in &weights {
                if current_load + w > mid {
                    need_days += 1;
                    current_load = 0;
                }
                current_load += w;
            }

            // 调整二分区间
            if need_days > days {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 875. 爱吃香蕉的珂珂
pub mod n875 {
    pub fn min_eating_speed(piles: Vec<i32>, h: i32) -> i32 {
        let check = |k: i32| -> bool {
            let mut sum = 0;

            for &p in &piles {
                sum += (p - 1) / k + 1; // p / k 向上取整

                if sum > h {
                    return true;
                }
            }

            false
        };

        let mut left = 1;
        let mut right = *piles.iter().max().unwrap();
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 3296. 移山所需的最少秒数
pub mod n3296 {
    pub fn min_number_of_seconds(mountain_height: i32, worker_times: Vec<i32>) -> i64 {
        // 校验函数：判断 t 秒内能否挖完指定高度的山
        let check = |t: i64| -> bool {
            let mut total = 0i64;
            for &time in worker_times.iter() {
                let time = time as i64;
                // 解方程: time * (1+2+...+k) ≤ t → k*(k+1) ≤ 2t/time
                let max_k = ((1.0 + 8.0 * t as f64 / time as f64).sqrt() - 1.0) / 2.0;
                let max_k = max_k.floor() as i64;
                total += max_k;
                if total >= mountain_height as i64 {
                    return false;
                }
            }
            total < mountain_height as i64
        };

        let n = worker_times.len() as i64;
        let max_t = *worker_times.iter().max().unwrap() as i64;
        let h = ((mountain_height - 1) as i64 / n) + 1;
        // 二分查找的上下界
        let mut left = 1i64;
        let mut right = max_t * h * (h + 1) / 2;

        // 二分查找最小满足条件的时间
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 3639. 变为活跃状态的最小时间
pub mod n3639 {
    pub fn min_time(s: String, order: Vec<i32>, k: i32) -> i32 {
        let n = s.len();
        // 提前判断：全改成星号子串数不足 k，直接返回 -1
        let total = (n as i64) * (n as i64 + 1) / 2;
        if total < k as i64 {
            return -1;
        }

        let mut star = vec![0; n];

        // 闭包：检查使用 order 前 m+1 个位置能否凑够 k 个子串
        let mut check = |m: usize| -> bool {
            let mark = m + 1;
            // 标记选中的位置
            for &idx in order.iter().take(mark) {
                star[idx as usize] = mark;
            }

            let mut cnt = 0;
            let mut last = -1; // 上一个 '*' 的位置（索引）
            for (i, &val) in star.iter().enumerate() {
                if val == mark {
                    last = i as i32;
                }
                cnt += last + 1; // 累加以当前位置结尾的合法子串数
                if cnt >= k {
                    return false;
                }
            }
            true
        };

        // 二分查找
        let mut left = 0;
        let mut right = (n - 1) as i32;
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid as usize) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 475. 供暖器
pub mod n475 {
    pub fn find_radius(mut houses: Vec<i32>, mut heaters: Vec<i32>) -> i32 {
        houses.sort_unstable();
        heaters.sort_unstable();

        let check = |x: i32| -> bool {
            let n = houses.len();
            let m = heaters.len();
            let mut j = 0;
            for i in 0..n {
                while j < m && houses[i] > heaters[j] + x {
                    j += 1;
                }
                if j >= m || houses[i] < heaters[j] - x {
                    return true;
                }
            }
            false
        };

        let mut left = 0;
        let mut right = 1e9 as i32;
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 2594. 修车的最少时间
pub mod n2594 {
    pub fn repair_cars(ranks: Vec<i32>, cars: i32) -> i64 {
        let mut cnt = std::collections::HashMap::new();
        ranks
            .into_iter()
            .for_each(|r| *cnt.entry(r).or_insert(0) += 1);

        let min_r = *cnt.keys().min().unwrap() as i64;
        let cars_target = cars as i64;
        let (mut left, mut right) = (1, min_r * cars_target * cars_target);

        let check = |mid: i64| -> bool {
            let mut total = 0i64;
            for (&r, &count) in &cnt {
                let r_i64 = r as i64;
                let num = ((mid / r_i64) as f64).sqrt() as i64;
                total += num * count as i64;
                if total >= cars_target {
                    return false;
                }
            }
            total < cars_target
        };

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 1482. 制作 m 束花所需的最少天数
pub mod n1482 {
    pub fn min_days(bloom_day: Vec<i32>, m: i32, k: i32) -> i32 {
        let n = bloom_day.len() as i32;
        // 提前判断无法满足的情况
        if m > n / k {
            return -1;
        }

        // 定义判断函数：给定天数 days，是否能制作出 m 束花
        let check = |days: i32| -> bool {
            let mut bouquets = 0;
            let mut flowers = 0;
            for &bloom in &bloom_day {
                if bloom <= days {
                    flowers += 1;
                    if flowers == k {
                        bouquets += 1;
                        if bouquets == m {
                            break;
                        }
                        flowers = 0;
                    }
                } else {
                    flowers = 0;
                }
            }
            bouquets < m
        };

        // 二分查找的边界
        let mut left = *bloom_day.iter().min().unwrap();
        let mut right = *bloom_day.iter().max().unwrap();

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 3048. 标记所有下标的最早秒数 I
pub mod n3048 {
    pub fn earliest_second_to_mark_indices(nums: Vec<i32>, change_indices: Vec<i32>) -> i32 {
        let n = nums.len();
        let m = change_indices.len();
        // 标记次数多于时间，直接返回-1
        if n > m {
            return -1;
        }

        // 计算完成所有标记和减一操作的最小时间
        let total_required = n as i32 + nums.iter().sum::<i32>();
        if total_required > m as i32 {
            return -1;
        }

        let mut done = vec![0; n];

        let mut check = |mx: usize| -> bool {
            // need_marks: 需要完成的标记数量
            let mut need_marks = n as i32;
            // need_sub: 需要执行减一的操作次数
            let mut need_sub = 0;
            // 倒序遍历前mx个时刻
            for i in (0..mx).rev() {
                let idx = (change_indices[i] - 1) as usize;
                if done[idx] != mx as i32 {
                    done[idx] = mx as i32;
                    need_marks -= 1;
                    need_sub += nums[idx];
                } else if need_sub > 0 {
                    // 执行减一操作
                    need_sub -= 1;
                }
            }
            need_marks > 0 || need_sub > 0
        };

        let mut left = total_required as usize;
        let mut right = m + 1;
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if right > m { -1 } else { right as _ }
    }
}

// 1870. 准时到达的列车最小时速
pub mod n1870 {
    // https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/solutions/791209/bi-mian-fu-dian-yun-suan-de-xie-fa-by-en-9fc6/
    pub fn min_speed_on_time(dist: Vec<i32>, hour: f64) -> i32 {
        let n = dist.len();
        // 由于双精度浮点数无法准确表示 2.01 这样的小数，我们在计算 2.01×100 时，算出的结果不是 201，而是 200.99999999999997 这样的数。
        // 所以代码不能直接转成整数，而是要 round 一下。
        let h100 = (hour * 100.0).round() as i64; // 下面不会用到任何浮点数
        let delta = h100 - (n as i64 - 1) * 100;
        if delta <= 0 {
            // 无法到达终点
            return -1;
        }

        let max_dist = *dist.iter().max().unwrap();
        if h100 <= n as i64 * 100 {
            // 特判
            // 见题解中的公式
            return max_dist.max(((dist[n - 1] * 100 - 1) as i64 / delta) as i32 + 1);
        }

        let check = |v: i32| -> bool {
            let mut t = 0i64;
            for &d in &dist[..n - 1] {
                t += ((d - 1) / v + 1) as i64; // d/v向上取整
            }
            (t * v as i64 + dist[n - 1] as i64) * 100 > h100 * v as i64
        };

        let sum_dist = dist.iter().map(|&x| x as i64).sum::<i64>();
        let mut left = ((sum_dist * 100 - 1) / h100) as i32 + 1; // sum_dist * 100 / h100 向上取整
        let h = (h100 / (n * 100) as i64) as i32;
        let mut right = (max_dist - 1) / h + 1;
        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }
}

// 3453. 分割正方形 I
pub mod n3453 {
    pub fn separate_squares(squares: Vec<Vec<i32>>) -> f64 {
        let mut max_y: f64 = 0.0;
        let mut total_area: f64 = 0.0;

        for sq in &squares {
            let l = sq[2] as f64;
            total_area += l * l;
            max_y = max_y.max((sq[1] + sq[2]) as f64);
        }
        let half_area = total_area / 2.0;

        // 定义检查函数：判断y高度以下的面积是否达到总面积的一半
        let check = |y: f64| -> bool {
            let mut area = 0.0;
            for square in squares.iter() {
                let yi = square[1] as f64;
                let l = square[2] as f64;
                if yi < y {
                    area += l * (l.min(y - yi));
                }
            }
            area < half_area
        };

        let mut left = 0.0;
        let mut right = max_y;
        let eps = 1e-5;

        // 二分查找
        while (right - left).abs() > eps {
            let mid = (left + right) / 2.0;
            if check(mid) {
                left = mid;
            } else {
                right = mid;
            }
        }

        (left + right) / 2.0
    }
}

// 275. H 指数 II
pub mod n275 {
    pub fn h_index(citations: Vec<i32>) -> i32 {
        let n = citations.len();
        let mut left = 1;
        let mut right = n + 1;
        while left < right {
            // 区间不为空
            // 循环不变量：
            // left-1 的回答一定为「是」
            // right 的回答一定为「否」
            let mid = (left + right) / 2;
            // 引用次数最多的 mid 篇论文，引用次数均 >= mid
            if citations[n - mid] >= mid as i32 {
                left = mid + 1; // 询问范围缩小到 [mid+1, right)
            } else {
                right = mid; // 询问范围缩小到 [left, mid)
            }
        }
        // 根据循环不变量，left-1 现在是最大的回答为「是」的数
        left as i32 - 1
    }
}

// 2226. 每个小孩最多能分到多少糖果
pub mod n2226 {
    pub fn maximum_candies(candies: Vec<i32>, k: i64) -> i32 {
        // 定义检查函数：判断每个孩子分 mid 颗糖时，是否能满足 k 个孩子
        let check = |mid: i64| -> bool {
            if mid == 0 {
                return true;
            }
            let mut total = 0i64;
            for &c in &candies {
                total += c as i64 / mid;
                if total >= k {
                    return true;
                }
            }
            total >= k
        };

        let sum_candies: i64 = candies.iter().map(|&x| x as i64).sum();
        if sum_candies < k {
            return 0;
        }

        let max_c = *candies.iter().max().unwrap_or(&0) as i64;
        let upper = max_c.min(sum_candies / k) + 1;
        let mut left = 0;
        let mut right = upper;

        // 二分查找：寻找最大满足条件的 mid
        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left as i32 - 1
    }
}

// 2982. 找出出现至少三次的最长特殊子字符串 II
pub mod n2982 {
    pub fn maximum_length(s: String) -> i32 {
        let bytes = s.as_bytes();
        let n = bytes.len();
        let mut cnt: std::collections::HashMap<u8, Vec<i32>> = std::collections::HashMap::new();

        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && bytes[j] == bytes[i] {
                j += 1;
            }
            // 连续长度 = j - i
            cnt.entry(bytes[i]).or_default().push((j - i) as i32);
            i = j;
        }

        let mut res = -1;
        for vec in cnt.values() {
            let mut left = 1;
            let mut right = (n as i32) - 1;
            while left < right {
                let mid = left + (right - left) / 2;
                let mut count = 0;
                for &x in vec {
                    if x >= mid {
                        count += x - mid + 1;
                    }
                }
                if count >= 3 {
                    // 满足条件，尝试更大值，左边界右移
                    res = res.max(mid);
                    left = mid + 1;
                } else {
                    // 不满足，右边界左移
                    right = mid;
                }
            }
        }

        res
    }
}

// 2576. 求出最多标记下标
pub mod n2576 {
    pub fn max_num_of_marked_indices(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable();

        // 检查k个数对是否合法：前k个和后k个一一匹配
        let check = |k: usize| -> bool {
            nums[0..k]
                .iter()
                .zip(nums[nums.len() - k..].iter())
                .all(|(&a, &b)| a * 2 <= b)
        };

        let mut left = 0;
        let mut right = nums.len() / 2 + 1;
        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                left = mid + 1; // 合法，尝试更大的k
            } else {
                right = mid; // 不合法，尝试更小的k
            }
        }

        if left == 0 { 0 } else { (left as i32 - 1) * 2 }
    }
}

// 1898. 可移除字符的最大数目
pub mod n1898 {
    pub fn maximum_removals(s: String, p: String, removable: Vec<i32>) -> i32 {
        let s_bytes = s.as_bytes();
        let p_bytes = p.as_bytes();
        let ns = s_bytes.len();
        let np = p_bytes.len();
        let n = removable.len();
        // 辅助函数：检查移除k个元素后，p是否是s的子序列
        let check = |k: usize| -> bool {
            let mut state = vec![true; ns];
            // 标记k个要移除的位置为false
            for i in 0..k {
                let idx = removable[i] as usize;
                state[idx] = false;
            }
            let mut j = 0;
            for i in 0..ns {
                if state[i] && j < np && s_bytes[i] == p_bytes[j] {
                    j += 1;
                    if j == np {
                        return true;
                    }
                }
            }
            j == np
        };

        let mut l = 0;
        let mut r = n + 1;
        while l < r {
            let mid = l + (r - l) / 2;
            if check(mid) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        (l - 1) as i32
    }
}

// 1802. 有界数组中指定下标处的最大值
pub mod n1802 {
    pub fn max_value(n: i32, index: i32, max_sum: i32) -> i32 {
        // 定义求和函数，计算长度为cnt的连续递减序列的和（从x开始）
        fn sum(x: i32, cnt: i32) -> i64 {
            if x >= cnt {
                // 等差数列求和：(首项 + 末项) * 项数 / 2
                (x as i64 + (x - cnt + 1) as i64) * cnt as i64 / 2
            } else {
                // 先算1到x的和，再加上剩余的1的个数
                (x as i64 * (x + 1) as i64) / 2 + (cnt - x) as i64
            }
        }

        let mut left = 1;
        let mut right = max_sum + 1;

        // 二分查找，寻找满足条件的最大值
        while left < right {
            let mid = (left + right) / 2;
            // 计算左侧部分和 + 右侧部分和
            let total = sum(mid - 1, index) + sum(mid, n - index);

            if total <= max_sum as i64 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left - 1
    }
}

// 1642. 可以到达的最远建筑
pub mod n1642 {
    pub fn furthest_building(heights: Vec<i32>, bricks: i32, ladders: i32) -> i32 {
        // 预处理高度差数组：heights[i] - heights[i-1]
        let diff: Vec<i32> = heights.windows(2).map(|w| w[1] - w[0]).collect();

        // 检查第 mid 个建筑是否可达（mid 对应原数组索引，0-based）
        let check = |mid: usize| -> bool {
            // 筛选前 mid 个高度差中的正数，转成 Vec 方便排序
            let mut cur_d: Vec<i32> = diff[0..mid].iter().filter(|&&d| d > 0).cloned().collect();
            // 降序排序：大的差值用梯子，小的用砖块
            cur_d.sort_by(|a, b| b.cmp(a));

            let ladders = ladders as usize;
            // 计算梯子用过后，剩余需要砖块的总和
            if ladders < cur_d.len() {
                let need_bricks: i32 = cur_d[ladders..].iter().sum();
                need_bricks <= bricks
            } else {
                true
            }
        };

        let (mut left, mut right) = (0, heights.len());
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left as i32 - 1
    }
}

// 2861. 最大合金数
pub mod n2861 {
    pub fn max_number_of_alloys(
        budget: i32,
        composition: Vec<Vec<i32>>,
        stock: Vec<i32>,
        cost: Vec<i32>,
    ) -> i32 {
        let mut ans = 0;
        for comp in composition {
            // 检查生产num个合金是否在预算内
            let check = |num: i32| -> bool {
                let mut money = 0;
                for ((&s, &base), &c) in stock.iter().zip(&comp).zip(&cost) {
                    let need = base as i64 * num as i64;
                    if (s as i64) < need {
                        money += (need - s as i64) * c as i64;
                        // 超预算立即返回
                        if money > budget as i64 {
                            return false;
                        }
                    }
                }
                money <= budget as i64
            };

            let mut left = 0;
            let mut right = stock.iter().min().copied().unwrap_or(0) + budget + 1;
            while left < right {
                let mid = left + (right - left) / 2;
                if check(mid) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            ans = ans.max(left - 1);
        }

        ans
    }
}
