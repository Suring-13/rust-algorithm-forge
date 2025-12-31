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
