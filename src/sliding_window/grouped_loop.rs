// 485. 最大连续 1 的个数
pub mod n485 {
    pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        let mut cnt = 0; // 存储当前连续1的个数

        // 遍历nums中的每个元素
        for &x in nums.iter() {
            if x == 1 {
                cnt += 1; // 遇到1，当前连续计数+1
                ans = ans.max(cnt); // 更新最大值
            } else {
                cnt = 0; // 遇到0，重置当前连续计数
            }
        }

        ans
    }
}

// 1446. 连续字符
pub mod n1446 {
    pub fn max_power(s: String) -> i32 {
        if s.is_empty() {
            return 0;
        }

        let mut ans = 1;
        let mut cnt = 1;
        let bytes = s.as_bytes();

        for i in 1..bytes.len() {
            if bytes[i] == bytes[i - 1] {
                cnt += 1;
                ans = ans.max(cnt);
            } else {
                cnt = 1;
            }
        }

        ans
    }
}

// 1869. 哪种连续子字符串更长
pub mod n1869 {
    pub fn check_zero_ones(s: &str) -> bool {
        let mut max1 = 0;
        let mut max0 = 0;
        let bytes = s.as_bytes();
        let mut i = 0;
        let n = bytes.len();

        while i < n {
            let start = i;
            let current = bytes[start];

            // 移动指针至不同字符处
            while i < n && bytes[i] == current {
                i += 1;
            }

            // 更新最长连续长度
            let length = i - start;
            if current == b'1' {
                max1 = max1.max(length);
            } else {
                max0 = max0.max(length);
            }
        }

        max1 > max0
    }
}

// 2414. 最长的字母序连续子字符串的长度
pub mod n2414 {
    pub fn longest_continuous_substring(s: String) -> i32 {
        let mut ans = 1;
        let mut cnt = 1;
        let s = s.as_bytes();
        for i in 1..s.len() {
            if s[i - 1] + 1 == s[i] {
                cnt += 1;
                ans = ans.max(cnt);
            } else {
                cnt = 1;
            }
        }
        ans
    }
}

// 3456. 找出长度为 K 的特殊子字符串
pub mod n3456 {
    pub fn has_special_substring(s: String, k: i32) -> bool {
        if k == 0 {
            return false;
        }
        let mut cnt = 0;
        let bytes = s.as_bytes();
        for i in 0..bytes.len() {
            cnt += 1;
            // 检查是否为最后一个字符，或当前字符与下一个字符不同
            if i == bytes.len() - 1 || bytes[i] != bytes[i + 1] {
                if cnt == k {
                    return true;
                }
                cnt = 0;
            }
        }
        false
    }
}

// 2348. 全 0 子数组的数目
pub mod n2348 {
    pub fn zero_filled_subarray(nums: Vec<i32>) -> i64 {
        let mut ans = 0;
        let mut last = -1;
        for (i, x) in nums.into_iter().enumerate() {
            let i = i as i64;
            if x != 0 {
                last = i; // 记录上一个非 0 元素的位置
            } else {
                ans += (i - last) as i64;
            }
        }
        ans
    }
}

// 1513. 仅含 1 的子串数
pub mod n1513 {
    pub fn num_sub(s: String) -> i32 {
        const MOD: i64 = 1_000_000_007;
        let mut ans = 0;
        let mut last0 = -1;
        for (i, ch) in s.bytes().enumerate() {
            if ch == b'0' {
                last0 = i as i32; // 记录上个 0 的位置
            } else {
                ans += (i as i32 - last0) as i64; // 右端点为 i 的全 1 子串个数
            }
        }
        (ans % MOD) as _
    }
}

// 1957. 删除字符使字符串变好
pub mod n1957 {
    pub fn make_fancy_string(s: String) -> String {
        let s = s.into_bytes();
        let mut ans = vec![];
        let mut cnt = 0;
        for (i, &ch) in s.iter().enumerate() {
            cnt += 1;
            if cnt < 3 {
                ans.push(ch);
            }
            if i + 1 < s.len() && ch != s[i + 1] {
                cnt = 0; // 当前字母和下个字母不同，重置计数器
            }
        }
        String::from_utf8(ans).unwrap()
    }
}

// 674. 最长连续递增序列
pub mod n674 {
    pub fn find_length_of_lcis(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut ans = 1;
        let mut cnt = 1;
        for i in 1..n {
            if nums[i] > nums[i - 1] {
                cnt += 1;
                ans = ans.max(cnt);
            } else {
                cnt = 1;
            }
        }
        ans
    }
}

// 3708. 最长斐波那契子数组
pub mod n3708 {
    pub fn longest_subarray(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        // 题目隐含nums长度≥3，直接初始化结果为最小可能长度2
        let mut ans = 2;
        let mut start = 0;

        // 从第三个元素（索引2）开始遍历
        for i in 2..n {
            // 不满足斐波那契条件时，更新最大长度并重置起始位置
            if nums[i] != nums[i - 1] + nums[i - 2] {
                ans = ans.max(i - start);
                start = i - 1;
            }
        }

        // 处理循环结束后剩余的子数组
        ans.max(n - start) as i32
    }
}

// 696. 计数二进制子串
pub mod n696 {
    pub fn count_binary_substrings(s: String) -> i32 {
        let mut ptr = 0;
        let n = s.len();
        let mut last = 0;
        let mut ans = 0;
        let s_bytes = s.as_bytes();

        while ptr < n {
            let c = s_bytes[ptr];
            let mut count = 0;
            // 统计当前连续字符的长度
            while ptr < n && s_bytes[ptr] == c {
                ptr += 1;
                count += 1;
            }
            ans += count.min(last);
            last = count;
        }

        ans
    }
}

// 978. 最长湍流子数组
pub mod n978 {
    pub fn max_turbulence_size(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        let mut ans = 1;
        let mut i = 0;
        let n = nums.len();

        while i < n {
            let i0 = i;
            i += 1;

            // 跳过连续相等的元素
            if i < n && nums[i - 1] == nums[i] {
                continue;
            }

            // 扩展湍流子数组
            if i < n {
                i += 1;
                while i < n
                    && nums[i] != nums[i - 1]
                    && (nums[i] < nums[i - 1]) != (nums[i - 1] < nums[i - 2])
                {
                    i += 1;
                }
                // 更新最大长度
                ans = ans.max(i - i0);
                i -= 1;
            }
        }

        ans as _
    }
}

// 2110. 股票平滑下跌阶段的数目
pub mod n2110 {
    pub fn get_descent_periods(prices: Vec<i32>) -> i64 {
        let mut ans = 0i64;
        let n = prices.len();
        let mut i = 0;

        while i < n {
            let i0 = i;
            // 移动指针，直到不满足“当前价格 = 前一个价格 - 1”的递减条件
            i += 1;
            while i < n && prices[i] == prices[i - 1] - 1 {
                i += 1;
            }
            // 计算当前递减序列的长度，套用“n(n+1)/2”公式累加结果
            let len = (i - i0) as i64;
            ans += len * (len + 1) / 2;
        }

        ans
    }
}

// 228. 汇总区间
pub mod n228 {
    pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;
        let n = nums.len();

        while i < n {
            let start = i;
            // 移动i指针，直到不满足连续递增条件
            while i < n && (i == start || nums[i] == nums[i - 1] + 1) {
                i += 1;
            }
            // 根据区间长度判断是单个数字还是范围
            if i - start == 1 {
                result.push(nums[start].to_string());
            } else {
                result.push(format!("{}->{}", nums[start], nums[i - 1]));
            }
        }

        result
    }
}

// 2760. 最长奇偶子数组
pub mod n2760 {
    pub fn longest_alternating_subarray(nums: Vec<i32>, threshold: i32) -> i32 {
        let n = nums.len();
        let mut ans = 0;
        let mut i = 0;
        while i < n {
            if nums[i] > threshold || nums[i] % 2 != 0 {
                i += 1;
                continue;
            }
            let start = i; // 记录这一组的开始位置
            i += 1; // 开始位置已经满足要求，从下一个位置开始判断
            while i < n && nums[i] <= threshold && nums[i] % 2 != nums[i - 1] % 2 {
                i += 1;
            }
            // 从 start 到 i-1 是满足题目要求的（并且无法再延长的）子数组
            ans = ans.max(i - start);
        }
        ans as i32
    }
}

// 1887. 使数组元素相等的减少操作次数
pub mod n1887 {
    pub fn reduction_operations(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable();
        let n = nums.len();
        let mut res = 0; // 总操作次数
        let mut cnt = 0; // 当前元素需要的操作次数

        for i in 1..n {
            // 如果当前元素和前一个不同，说明需要多一次操作
            if nums[i] != nums[i - 1] {
                cnt += 1;
            }
            // 累加当前元素的操作次数到总次数
            res += cnt;
        }

        res
    }
}

// 845. 数组中的最长山脉
pub mod n845 {
    pub fn longest_mountain(arr: &Vec<i32>) -> i32 {
        let mut ret = 0;
        let n = arr.len();
        let mut i = 0;

        while i < n {
            // 确定山脉起始点，若前一个元素小于当前，起始点为前一个索引
            let start = if i > 0 && arr[i - 1] < arr[i] {
                i - 1
            } else {
                i
            };

            // 遍历严格上升段，找到峰顶
            while i + 1 < n && arr[i] < arr[i + 1] {
                i += 1;
            }
            let top = i; // 记录峰顶索引

            // 遍历严格下降段
            while i + 1 < n && arr[i] > arr[i + 1] {
                i += 1;
            }

            // 仅当存在完整上升段和下降段时，更新最大长度
            if top > start && i > top {
                ret = ret.max(i - start + 1);
            }

            i += 1;
        }

        ret as _
    }
}

// 2038. 如果相邻两个颜色均相同则删除当前颜色
pub mod n2038 {
    pub fn winner_of_game(colors: String) -> bool {
        let mut cnt = [0; 2];
        let bytes = colors.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            let i0 = i;
            let c = bytes[i0];
            // 移动指针直到字节不同
            while i < n && bytes[i] == c {
                i += 1;
            }
            // 计算连续长度，超过2则累加可操作次数
            let len = i - i0;
            if len > 2 {
                // 'A' - b'A' = 0，'B' - b'A' = 1，直接映射到cnt的索引
                cnt[(c - b'A') as usize] += len - 2;
            }
        }

        cnt[0] > cnt[1]
    }
}

// 2900. 最长相邻不相等子序列 I
pub mod n2900 {
    pub fn get_longest_subsequence(words: Vec<String>, groups: Vec<i32>) -> Vec<String> {
        let n = groups.len();
        let mut ans = vec![];
        for (i, word) in words.into_iter().enumerate() {
            if i == n - 1 || groups[i] != groups[i + 1] {
                // i 是连续相同段的末尾
                ans.push(word);
            }
        }
        ans
    }
}

// 1759. 统计同质子字符串的数目
pub mod n1759 {
    pub fn count_homogenous(s: String) -> i32 {
        const MOD: i64 = 1_000_000_007;

        let s_bytes = s.as_bytes();
        let n = s_bytes.len();
        let mut ans: i64 = 0;
        let mut l = 0;

        while l < n {
            let mut r = l + 1;
            // 移动右指针，找到连续相同字符的边界
            while r < n && s_bytes[r] == s_bytes[r - 1] {
                r += 1;
            }
            // 计算当前连续段的同质子字符串数量（等差数列求和）
            let len = (r - l) as i64;
            ans = (ans + len * (len + 1) / 2) % MOD;
            // 左指针跳到当前连续段的下一个位置
            l = r;
        }

        ans as i32
    }
}

// 3011. 判断一个数组是否可以变为有序
pub mod n3011 {
    pub fn can_sort_array(nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut i = 0;
        let mut pre_max = 0;

        while i < n {
            let mut current_max = 0;
            // 计算当前元素的二进制1的个数，作为分组依据
            let ones = nums[i].count_ones();

            // 遍历当前同1数量的分组
            while i < n && nums[i].count_ones() == ones {
                // 若当前元素小于前组最大值，直接返回false
                if nums[i] < pre_max {
                    return false;
                }
                // 更新当前组的最大值
                current_max = current_max.max(nums[i]);
                i += 1;
            }

            // 更新前组最大值
            pre_max = current_max;
        }

        true
    }
}

// 1578. 使绳子变成彩色的最短时间
pub mod n1578 {
    pub fn min_cost(colors: String, needed_time: Vec<i32>) -> i32 {
        let mut i = 0;
        let len = colors.len();
        let mut result = 0;
        let color_bytes = colors.as_bytes();

        while i < len {
            let current_byte = color_bytes[i];
            let mut max_time = 0;
            let mut sum_time = 0;

            // 遍历当前连续相同颜色的气球组
            while i < len && color_bytes[i] == current_byte {
                max_time = max_time.max(needed_time[i]);
                sum_time += needed_time[i];
                i += 1;
            }

            // 累加当前组的最小成本（总时间 - 最大时间）
            result += sum_time - max_time;
        }

        result
    }
}

// 1839. 所有元音按顺序排布的最长子字符串
pub mod n1839 {
    pub fn longest_beautiful_substring(word: String) -> i32 {
        let mut ans = 0;
        let bytes = word.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            let mut cur_len = 1;
            let mut vowel_type = 1;
            i += 1;

            // 对比相邻字节，确保非递减，同时统计元音种类
            while i < n && bytes[i - 1] <= bytes[i] {
                if bytes[i - 1] != bytes[i] {
                    vowel_type += 1;
                }
                cur_len += 1;
                i += 1;
            }

            // 仅当包含全部 5 种元音（a,e,i,o,u，ASCII 顺序递增）时更新最大值
            if vowel_type == 5 {
                ans = ans.max(cur_len);
            }
        }

        ans as i32
    }
}

// 2765. 最长交替子数组
pub mod n2765 {
    pub fn alternating_subarray(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut ans = -1;
        let mut i = 0;
        while i < n - 1 {
            if nums[i + 1] - nums[i] != 1 {
                i += 1;
                continue;
            }
            let i0 = i; // 记录这一组的开始位置
            i += 2; // i 和 i+1 已经满足要求，从 i+2 开始判断
            while i < n && nums[i] == nums[i - 2] {
                i += 1;
            }
            ans = ans.max((i - i0) as i32);
            i -= 1;
        }
        ans
    }
}

// 3255. 长度为 K 的子数组的能量值 II
pub mod n3255 {
    pub fn results_array(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let mut ans = vec![-1; nums.len() - k + 1];
        let mut cnt = 0;
        for i in 0..nums.len() {
            if i == 0 || nums[i] == nums[i - 1] + 1 {
                cnt += 1;
            } else {
                cnt = 1;
            }
            if cnt >= k {
                ans[i - k + 1] = nums[i];
            }
        }
        ans
    }
}

// 3350. 检测相邻递增子数组 II
pub mod n3350 {
    pub fn max_increasing_subarrays(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0; // 数组长度小于2时，无符合条件的子数组
        }

        let mut cnt = 1; // 当前递增子数组长度
        let mut precnt = 0; // 上一个递增子数组长度
        let mut ans = 0; // 最终结果

        // 遍历数组，从第二个元素开始对比
        for i in 1..n {
            if nums[i] > nums[i - 1] {
                cnt += 1;
            } else {
                precnt = cnt;
                cnt = 1;
            }
            ans = ans.max(precnt.min(cnt)).max(cnt / 2);
        }

        ans
    }
}

// 3105. 最长的严格递增或递减子数组
pub mod n3105 {
    pub fn longest_monotonic_subarray(nums: Vec<i32>) -> i32 {
        let mut ans = 1;
        let (mut i, n) = (0, nums.len());
        while i < n - 1 {
            if nums[i + 1] == nums[i] {
                i += 1; // 直接跳过
                continue;
            }
            let i0 = i; // 记录这一组的开始位置
            let inc = nums[i + 1] > nums[i]; // 定下基调：是严格递增还是严格递减
            i += 2; // i 和 i+1 已经满足要求，从 i+2 开始判断
            while i < n && nums[i] != nums[i - 1] && (nums[i] > nums[i - 1]) == inc {
                i += 1;
            }

            // 从 i0 到 i-1 是满足题目要求的（并且无法再延长的）子数组
            ans = ans.max(i - i0);

            i -= 1; // 第一个单调序列末尾和第二个单调序列开头，有一个元素是重叠的，循环末尾要把 i 减一
        }

        ans as i32
    }
}

// 838. 推多米诺
pub mod n838 {
    pub fn push_dominoes(dominoes: String) -> String {
        let mut s = dominoes.into_bytes();
        let n = s.len();
        let mut i = 0;
        let mut left = b'L';
        while i < n {
            let mut j = i;
            while j < n && s[j] == b'.' {
                // 找到一段连续的没有被推动的骨牌
                j += 1;
            }
            let right = if j < n { s[j] } else { b'R' };
            match (left, right) {
                // 方向相同，那么这些竖立骨牌也会倒向同一方向
                (b'L', b'L') | (b'R', b'R') => {
                    while i < j {
                        s[i] = right;
                        i += 1;
                    }
                }
                // 方向相对，那么就从两侧向中间倒
                (b'R', b'L') => {
                    let mut k = j - 1;
                    while i < k {
                        s[i] = b'R';
                        s[k] = b'L';
                        i += 1;
                        k -= 1;
                    }
                }
                _ => {}
            }
            left = right;
            i = j + 1;
        }
        String::from_utf8(s).unwrap()
    }
}

// 467. 环绕字符串中唯一的子字符串
pub mod n467 {
    pub fn find_substring_in_wrapround_string(s: String) -> i32 {
        let s_bytes = s.as_bytes();
        let n = s_bytes.len();
        if n == 0 {
            return 0;
        }

        let mut map = std::collections::HashMap::new();
        let mut i = 0;

        while i < n {
            let start = i;
            i += 1;
            // 判断是否是连续的环绕子串
            while i < n
                && (s_bytes[i] - s_bytes[i - 1] == 1
                    || s_bytes[i - 1] == b'z' && s_bytes[i] == b'a')
            {
                i += 1;
            }

            // 每个字符最多统计26个（字母只有26个）
            let end = i.min(start + 26);
            for j in start..end {
                let c = s_bytes[j];
                // 记录以当前字符开头的最长有效子串长度
                let entry = map.entry(c).or_insert(0);
                *entry = (*entry).max(i - j);
            }
        }

        // 累加所有字符对应的最长长度
        map.values().sum::<usize>() as i32
    }
}

// 3499. 操作后最大活跃区段数 I
pub mod n3499 {
    pub fn max_active_sections_after_trade(s: String) -> i32 {
        let s_bytes = s.as_bytes();
        let n = s_bytes.len();
        if n == 0 {
            return 0;
        }

        let mut total1 = 0;
        let mut max_0 = 0;
        let mut current_cnt = 0;
        let mut prev_0 = i32::MIN;

        for i in 0..n {
            current_cnt += 1;
            // 判断是否到达当前段的末尾
            if i == n - 1 || s_bytes[i] != s_bytes[i + 1] {
                if s_bytes[i] == b'1' {
                    total1 += current_cnt;
                } else {
                    // 更新 010 子串中 0 的最大个数，这里 010 子串是指一段连续的 0，紧跟着一段连续的 1，再紧跟着一段连续的 0
                    max_0 = max_0.max(prev_0 + current_cnt);
                    prev_0 = current_cnt;
                }
                current_cnt = 0;
            }
        }

        total1 + max_0
    }
}

// 413. 等差数列划分
pub mod n413 {
    pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 3 {
            return 0;
        }

        // 初始化公差（nums[0]-nums[1]）、临时计数t、结果ans
        let mut d = nums[0] - nums[1];
        let mut t = 0;
        let mut ans = 0;

        // 从索引2开始遍历（因为至少需要3个元素形成等差数列）
        for i in 2..n {
            if nums[i - 1] - nums[i] == d {
                // 连续满足公差，新增t+1个等差数列
                t += 1;
            } else {
                // 公差变化，重置公差和计数
                d = nums[i - 1] - nums[i];
                t = 0;
            }
            // 累加当前新增的等差数列数量
            ans += t;
        }

        ans
    }
}

// 68. 文本左右对齐
pub mod n68 {
    pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
        let n = words.len();
        let max_width = max_width as usize;
        let mut ans = vec![];
        let mut i = 0;
        while i < n {
            let start = i; // 这一行第一个单词的下标
            let mut sum_len = words[i].len(); // 第一个单词的长度
            i += 1;
            while i < n && sum_len + 1 + words[i].len() <= max_width {
                sum_len += words[i].len() + 1; // 单词之间至少要有一个空格
                i += 1;
            }

            let extra_spaces = max_width - sum_len; // 这一行剩余未分配的空格个数
            let gaps = i - start - 1; // 这一行单词之间的空隙个数（单词个数减一）

            // 特殊情况：如果只有一个单词，或者是最后一行，那么左对齐，末尾补空格
            if gaps == 0 || i == n {
                let row = words[start..i].join(" ") + &" ".repeat(extra_spaces); // 末尾补空格
                ans.push(row);
                continue;
            }

            // 一般情况：把 extra_spaces 个空格均匀分配到 gaps 个空隙中（靠左的空格更多）
            let avg = extra_spaces / gaps;
            let rem = extra_spaces % gaps;
            let spaces = &" ".repeat(avg + 1); // +1 表示加上单词之间已有的一个空格
            let row = words[start..start + rem + 1].join(&(spaces.clone() + " ")) + // 前 rem 个空隙多一个空格
                spaces + &words[start + rem + 1..i].join(spaces);
            ans.push(row);
        }
        ans
    }
}

// 135. 分发糖果
pub mod n135 {
    pub fn candy(ratings: Vec<i32>) -> i32 {
        let n = ratings.len();
        // 初始化：每个孩子先分1颗糖果，总数量为n
        let mut ans = n as i32;
        // 遍历指针，从数组起始位置开始
        let mut i = 0;

        // 遍历整个评分数组，处理每一段递增-递减区间
        while i < n {
            // 确定递增段的起点：如果当前处于递增序列（前一个评分 < 当前评分）
            // 则将起点回退到前一个位置，保证递增段的完整性
            let start = if i > 0 && ratings[i - 1] < ratings[i] {
                i - 1
            } else {
                i
            };

            // ===== 第一步：扫描严格递增段，找到峰顶位置 =====
            // 只要下一个评分更大，就继续往后走
            while i + 1 < n && ratings[i] < ratings[i + 1] {
                i += 1;
            }
            // 此时i的位置是当前区间的"峰顶"（比左右两边都大，或处于数组末尾）
            let top = i;

            // ===== 第二步：扫描严格递减段，找到区间终点 =====
            // 只要下一个评分更小，就继续往后走
            while i + 1 < n && ratings[i] > ratings[i + 1] {
                i += 1;
            }

            // 计算递增段和递减段的长度（均不包含峰顶本身）
            // inc：从start到top的元素个数（严格递增）
            let inc = (top - start) as i32;
            // dec：从top到i的元素个数（严格递减）
            let dec = (i - top) as i32;

            // ===== 第三步：计算当前区间需要额外添加的糖果数 =====
            // 1. 递增段额外糖果：0+1+2+...+(inc-1) = inc*(inc-1)/2
            // 2. 递减段额外糖果：0+1+2+...+(dec-1) = dec*(dec-1)/2
            // 3. 峰顶额外糖果：取inc和dec的最大值，因为峰顶要比左右两段的末尾都多1颗
            ans += (inc * (inc - 1) + dec * (dec - 1)) / 2 + inc.max(dec);

            // 指针后移，处理下一个区间
            i += 1;
        }

        // 返回最少需要的糖果总数
        ans
    }
}

// 2948. 交换得到字典序最小的数组
pub mod n2948 {
    // 本题用到一个经典结论。把序列中的每个元素看成图中的一个点，如果两个元素可以相互交换，则在两个元素之间连一条边。结论：处于同一连通块内的所有元素可以通过若干次操作排成任意顺序。
    // 一个连通块的元素无法移到另一个连通块的位置上
    pub fn lexicographically_smallest_array(nums: Vec<i32>, limit: i32) -> Vec<i32> {
        let n = nums.len();
        // 构建 (数值, 原始索引) 的元组向量，并按数值升序排序
        let mut sorted_pairs: Vec<(i32, usize)> =
            nums.into_iter().enumerate().map(|(i, x)| (x, i)).collect();
        sorted_pairs.sort_unstable_by_key(|&(val, _)| val);

        let mut ans = vec![0; n];
        let mut i = 0;

        while i < n {
            let start = i;
            i += 1;
            // 扩展当前分组：满足相邻元素差值 ≤ limit
            while i < n && sorted_pairs[i].0 - sorted_pairs[i - 1].0 <= limit {
                i += 1;
            }
            // 提取当前分组的原始索引并排序
            let mut indices: Vec<usize> =
                sorted_pairs[start..i].iter().map(|&(_, idx)| idx).collect();
            indices.sort_unstable();

            // 将分组内的有序数值，按排序后的原始索引填入结果数组
            for (pos, &idx) in indices.iter().enumerate() {
                ans[idx] = sorted_pairs[start + pos].0;
            }
        }

        ans
    }
}

// 2593. 标记所有元素后数组的分数
pub mod n2593 {
    pub fn find_score(nums: Vec<i32>) -> i64 {
        let mut ans = 0i64;
        let n = nums.len();
        let mut i = 0;

        while i < n {
            let i0 = i;
            // 找到下坡的坡底（连续递减的最后一个位置）
            while i + 1 < n && nums[i] > nums[i + 1] {
                i += 1;
            }
            // 从坡底 i 到坡顶 i0，每隔一个累加
            for j in (i0..=i).rev().step_by(2) {
                ans += nums[j] as i64;
            }
            // i 选了, i+1 不能选
            i += 2;
        }

        ans
    }
}
