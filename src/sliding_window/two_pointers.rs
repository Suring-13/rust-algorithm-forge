// 344. 反转字符串
pub mod n344 {
    pub fn reverse_string(s: &mut Vec<char>) {
        let n = s.len();
        let mut left = 0;
        let mut right = n - 1;
        while left < right {
            s.swap(left, right);
            left += 1;
            right -= 1;
        }
    }
}

// 3643. 垂直翻转子矩阵
pub mod n3643 {
    use std::mem::swap;

    pub fn reverse_submatrix(mut grid: Vec<Vec<i32>>, x: i32, y: i32, k: i32) -> Vec<Vec<i32>> {
        let (x, y, k) = (x as usize, y as usize, k as usize);
        let (mut l, mut r) = (x, x + k - 1);
        while l < r {
            // 使用split_at_mut将网格分割为两部分，避免同时获取多个可变引用
            let (left, right) = grid.split_at_mut(r);
            // left包含0..r的行，right包含r..的行，right[0]就是原来的grid[r]
            let row_l = &mut left[l];
            let row_r = &mut right[0];

            // 交换 l 行与 r 行的 [y, y+k) 列元素
            for j in y..y + k {
                swap(&mut row_l[j], &mut row_r[j]);
            }
            l += 1;
            r -= 1;
        }
        grid
    }
}

// 125. 验证回文串
pub mod n125 {
    pub fn is_palindrome(s: String) -> bool {
        let s = s.as_bytes();
        let mut i = 0;
        let mut j = s.len() - 1;
        while i < j {
            if !s[i].is_ascii_alphanumeric() {
                i += 1;
            } else if !s[j].is_ascii_alphanumeric() {
                j -= 1;
            } else if s[i].to_ascii_lowercase() == s[j].to_ascii_lowercase() {
                i += 1;
                j -= 1;
            } else {
                return false;
            }
        }
        true
    }
}

// 1750. 删除字符串两端相同字符后的最短长度
pub mod n1750 {
    pub fn minimum_length(s: String) -> i32 {
        let s_bytes = s.as_bytes(); // 转为字节数组，避免字符索引开销
        let mut left = 0;
        let mut right = s_bytes.len() - 1;

        while left < right && s_bytes[left] == s_bytes[right] {
            let c = s_bytes[left];
            // 收缩左指针：跳过所有与 c 相同的字符
            while left <= right && s_bytes[left] == c {
                left += 1;
            }
            // 收缩右指针：跳过所有与 c 相同的字符
            while right >= left && s_bytes[right] == c {
                right -= 1;
            }
        }

        // 计算剩余长度
        if left > right {
            0
        } else {
            (right - left + 1) as i32
        }
    }
}

// 2105. 给植物浇水 II
pub mod n2105 {
    pub fn minimum_refill(plants: Vec<i32>, capacity_a: i32, capacity_b: i32) -> i32 {
        let mut ans = 0;

        let mut a = capacity_a;
        let mut b = capacity_b;

        let mut i = 0;
        let mut j = plants.len() - 1;

        while i < j {
            // Alice 给植物 i 浇水
            if a < plants[i] {
                // 没有足够的水，重新灌满水罐
                ans += 1;
                a = capacity_a;
            }
            a -= plants[i];
            i += 1;

            // Bob 给植物 j 浇水
            if b < plants[j] {
                // 没有足够的水，重新灌满水罐
                ans += 1;
                b = capacity_b;
            }
            b -= plants[j];
            j -= 1;
        }

        // 如果 Alice 和 Bob 到达同一株植物，那么当前水罐中水更多的人会给这株植物浇水
        if i == j && a.max(b) < plants[i] {
            // 没有足够的水，重新灌满水罐
            ans += 1;
        }

        ans
    }
}

// 977. 有序数组的平方
pub mod n977 {
    pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut ans = vec![0; n];
        let mut i = 0;
        let mut j = n - 1;
        for p in (0..n).rev() {
            let x = nums[i] * nums[i];
            let y = nums[j] * nums[j];
            if x > y {
                ans[p] = x;
                i += 1;
            } else {
                ans[p] = y;
                j -= 1;
            }
        }
        ans
    }
}

// 658. 找到 K 个最接近的元素
pub mod n658 {
    // 双指针法
    pub fn find_closest_elements(arr: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
        let n = arr.len();
        let mut left = 0usize;
        let mut right = n - 1usize;
        // 计算初始左右指针元素与 x 的绝对差
        let mut a = (arr[left] - x).abs();
        let mut b = (arr[right] - x).abs();

        // 循环收缩指针范围，直到包含 k 个元素
        while (right - left + 1) as i32 != k {
            if a <= b {
                // 移除右侧元素：right 左移，更新 b
                right -= 1;
                b = (arr[right] - x).abs();
            } else {
                // 移除左侧元素：left 右移，更新 a
                left += 1;
                a = (arr[left] - x).abs();
            }
        }

        // 切片截取 [left..=right] 范围的元素
        arr[left..=right].to_vec()
    }

    // 二分法
    pub fn find_closest_elements2(arr: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
        let k_usize = k as usize;
        let mut left = 0usize;
        // 右边界初始化为“数组长度 - k”，确保窗口 [left, left+k) 始终合法（不越界）
        let mut right = arr.len() - k_usize;
        let mut mid: usize;

        while left < right {
            mid = left + (right - left) / 2;
            // 核心判断：比较当前窗口左端点（mid）和下一个窗口左端点（mid+1）的优劣
            // 逻辑：若 x 到当前窗口左端点的距离 > x 到下一个窗口右端点（mid+k）的距离，
            // 说明下一个窗口（mid+1 为左端点）更优，需右移左边界；反之则左移右边界
            if x - arr[mid] > arr[mid + k_usize] - x {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // 截取最优窗口 [left, left+k)
        arr[left..left + k_usize].to_vec()
    }
}

// 1471. 数组中的 k 个最强值
pub mod n1471 {
    pub fn get_strongest(mut arr: Vec<i32>, k: i32) -> Vec<i32> {
        // 排序数组
        arr.sort_unstable();
        let arr_len = arr.len();
        // 中位数
        let median = arr[(arr_len - 1) / 2];

        // 双指针
        let mut left = 0;
        let mut right = arr_len - 1;
        // 提前分配结果空间
        let mut res = Vec::with_capacity(k as usize);

        // 选k个最强元素
        for _ in 0..k {
            let left_diff = (arr[left] - median).abs();
            let right_diff = (arr[right] - median).abs();

            if left_diff > right_diff {
                res.push(arr[left]);
                left += 1;
            } else {
                res.push(arr[right]);
                right -= 1;
            }
        }

        res
    }
}

// 167. 两数之和 II - 输入有序数组
pub mod n167 {
    pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
        let mut left = 0;
        let mut right = numbers.len() - 1;

        loop {
            let s = numbers[left] + numbers[right];

            if s == target {
                return vec![left as i32 + 1, right as i32 + 1]; // 题目要求下标从 1 开始
            } else if s > target {
                right -= 1;
            } else {
                left += 1;
            }
        }
    }
}

// 633. 平方数之和
pub mod n633 {
    pub fn judge_square_sum(c: i32) -> bool {
        let mut a = 0;
        let mut b = (c as f64).sqrt() as i32;
        while a <= b {
            if a * a == c - b * b {
                return true;
            }
            if a * a < c - b * b {
                a += 1;
            } else {
                b -= 1;
            }
        }
        false
    }
}

// 2824. 统计和小于目标的下标对数目
pub mod n2824 {
    pub fn count_pairs(mut nums: Vec<i32>, target: i32) -> i32 {
        nums.sort_unstable();
        let mut ans = 0;
        let mut left = 0;
        let mut right = nums.len() - 1;
        while left < right {
            if nums[left] + nums[right] < target {
                ans += right - left;
                left += 1;
            } else {
                right -= 1;
            }
        }
        ans as _
    }
}

// 2563. 统计公平数对的数目
pub mod n2563 {
    pub fn count_fair_pairs(mut nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        nums.sort_unstable();

        let mut ans = 0;

        let mut l = nums.len();
        let mut r = nums.len();

        // 随着 nums[j] 的变大，upper−nums[j] 和 lower−nums[j] 都在变小，有单调性
        for (j, &x) in nums.iter().enumerate() {
            // 找 > upper−nums[j] 的第一个数
            while r > 0 && nums[r - 1] > upper - x {
                r -= 1;
            }

            // 找 ≥ lower−nums[j] 的第一个数
            while l > 0 && nums[l - 1] >= lower - x {
                l -= 1;
            }

            ans += r.min(j) - l.min(j);
        }

        ans as _
    }
}

// 15. 三数之和
pub mod n15 {
    pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let n = nums.len();
        let mut ans = vec![];
        for i in 0..n - 2 {
            let x = nums[i];
            if i > 0 && x == nums[i - 1] {
                // 跳过重复数字
                continue;
            }

            // 如果 nums[i] 与后面最小的两个数相加 nums[i]+nums[i+1]+nums[i+2]>0，那么后面不可能存在三数之和等于 0，break 外层循环
            if x + nums[i + 1] + nums[i + 2] > 0 {
                break;
            }

            // 如果 nums[i] 与后面最大的两个数相加 nums[i]+nums[n−2]+nums[n−1]<0，那么内层循环不可能存在三数之和等于 0，
            // 但继续枚举，nums[i] 可以变大，所以后面还有机会找到三数之和等于 0，continue 外层循环。
            if x + nums[n - 2] + nums[n - 1] < 0 {
                continue;
            }
            let mut j = i + 1;
            let mut k = n - 1;
            while j < k {
                let s = x + nums[j] + nums[k];
                if s > 0 {
                    k -= 1;
                } else if s < 0 {
                    j += 1;
                } else {
                    // 三数之和为 0
                    ans.push(vec![x, nums[j], nums[k]]);
                    j += 1;
                    while j < k && nums[j] == nums[j - 1] {
                        // 跳过重复数字
                        j += 1;
                    }
                    k -= 1;
                    while k > j && nums[k] == nums[k + 1] {
                        // 跳过重复数字
                        k -= 1;
                    }
                }
            }
        }
        ans
    }
}

// 16. 最接近的三数之和
pub mod n16 {
    pub fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
        nums.sort_unstable();
        let n = nums.len();
        let mut min_diff = i32::MAX;
        let mut ans = 0;

        for i in 0..n - 2 {
            let x = nums[i];
            if i > 0 && x == nums[i - 1] {
                continue;
            }

            // 当前最小和已大于target，后续和更大，直接break
            let s = x + nums[i + 1] + nums[i + 2];
            if s > target {
                if s - target < min_diff {
                    ans = s;
                }
                break;
            }

            // 当前最大和仍小于target，跳过双指针，直接continue
            let s = x + nums[n - 2] + nums[n - 1];
            if s < target {
                if target - s < min_diff {
                    min_diff = target - s;
                    ans = s;
                }
                continue;
            }

            // 双指针：用符号判断替代cmp枚举
            let mut j = i + 1;
            let mut k = n - 1;
            while j < k {
                let s = x + nums[j] + nums[k];
                // 直接通过差值符号判断和与目标的关系
                if s == target {
                    return s;
                } else if s > target {
                    if s - target < min_diff {
                        min_diff = s - target;
                        ans = s;
                    }
                    k -= 1;
                    while k > j && nums[k] == nums[k + 1] {
                        // 跳过重复数字
                        k -= 1;
                    }
                } else {
                    if target - s < min_diff {
                        min_diff = target - s;
                        ans = s;
                    }
                    j += 1;
                    while j < k && nums[j] == nums[j - 1] {
                        // 跳过重复数字
                        j += 1;
                    }
                }
            }
        }

        ans
    }
}

// 18. 四数之和
pub mod n18 {
    pub fn four_sum(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let target = target as i64;
        let n = nums.len();
        let mut ans = vec![];
        for a in 0..n.saturating_sub(3) {
            // 枚举第一个数
            let x = nums[a];
            if a > 0 && x == nums[a - 1] {
                // 跳过重复数字
                continue;
            }
            // 当前最小和已大于target，后续和更大，直接break
            if (x + nums[a + 1]) as i64 + (nums[a + 2] + nums[a + 3]) as i64 > target {
                break;
            }
            // 当前最大和仍小于target，跳过双指针，直接continue
            if ((x + nums[n - 3]) as i64 + (nums[n - 2] + nums[n - 1]) as i64) < target {
                continue;
            }

            for b in a + 1..n - 2 {
                // 枚举第二个数
                let y = nums[b];
                if b > a + 1 && y == nums[b - 1] {
                    // 跳过重复数字
                    continue;
                }
                // 当前最小和已大于target，后续和更大，直接break
                if (x + y) as i64 + (nums[b + 1] + nums[b + 2]) as i64 > target {
                    break;
                }
                // 当前最大和仍小于target，跳过双指针，直接continue
                if ((x + y) as i64 + (nums[n - 2] + nums[n - 1]) as i64) < target {
                    continue;
                }
                let mut c = b + 1;
                let mut d = n - 1;
                while c < d {
                    // 双指针枚举第三个数和第四个数
                    let s = (x + y) as i64 + (nums[c] + nums[d]) as i64; // 四数之和
                    if s > target {
                        d -= 1;
                    } else if s < target {
                        c += 1;
                    } else {
                        // s == target
                        ans.push(vec![x, y, nums[c], nums[d]]);
                        c += 1;
                        while c < d && nums[c] == nums[c - 1] {
                            c += 1; // 跳过重复数字
                        }
                        d -= 1;
                        while d > c && nums[d] == nums[d + 1] {
                            d -= 1; // 跳过重复数字
                        }
                    }
                }
            }
        }
        ans
    }
}

// 611. 有效三角形的个数
pub mod n611 {
    pub fn triangle_number(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable();
        let mut ans = 0;

        // 假设 1 ≤ a ≤ b ≤ c
        for k in 2..nums.len() {
            let c = nums[k];

            let mut i = 0; // a=nums[i]
            let mut j = k - 1; // b=nums[j]

            while i < j {
                if nums[i] + nums[j] > c {
                    ans += j - i;
                    j -= 1;
                } else {
                    i += 1;
                }
            }
        }

        ans as _
    }
}

// 1577. 数的平方等于两数乘积的方法数
pub mod n1577 {
    pub fn num_triplets(mut nums1: Vec<i32>, mut nums2: Vec<i32>) -> i32 {
        nums1.sort_unstable();
        nums2.sort_unstable();

        let mut ans = 0;

        // 第一部分：在nums2中查找与nums1[i]²相等的数对
        for &num in &nums1 {
            let target = (num as i64) * (num as i64);
            ans += count_pairs(&nums2, target);
        }

        // 第二部分：在nums1中查找与nums2[i]²相等的数对
        for &num in &nums2 {
            let target = (num as i64) * (num as i64);
            ans += count_pairs(&nums1, target);
        }

        ans
    }

    // 辅助函数：计算数组中乘积等于目标值的数对数量
    fn count_pairs(nums: &[i32], target: i64) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        let mut count = 0;

        while left < right {
            let product = (nums[left] as i64) * (nums[right] as i64);

            if product == target {
                if nums[left] == nums[right] {
                    // 所有元素都相等的情况，计算组合数
                    let n = right - left + 1;
                    count += (n * (n - 1) / 2) as i32;
                    break;
                } else {
                    // 统计左侧相同元素的数量
                    let left_val = nums[left];
                    let mut left_count = 0;
                    let mut l = left;
                    while l <= right && nums[l] == left_val {
                        left_count += 1;
                        l += 1;
                    }

                    // 统计右侧相同元素的数量
                    let right_val = nums[right];
                    let mut right_count = 0;
                    let mut r = right;
                    while r >= left && nums[r] == right_val {
                        right_count += 1;
                        r = r.wrapping_sub(1); // 避免underflow
                    }

                    count += (left_count * right_count) as i32;
                    left = l;
                    right = r;
                }
            } else if product < target {
                left += 1;
            } else {
                if right == 0 {
                    break;
                }
                right -= 1;
            }
        }

        count
    }
}

// 923. 三数之和的多种可能
pub mod n923 {
    pub fn three_sum_multi(mut nums: Vec<i32>, target: i32) -> i32 {
        // 1.排序：为双指针遍历提供有序基础，确保左右指针移动逻辑生效
        nums.sort_unstable();
        let n = nums.len();
        let mod_val = 1_000_000_007;
        let mut res = 0i64;

        // 2.遍历第一个数：作为三元组的固定左端点，后续用双指针找剩余两个数
        for i in 0..n - 2 {
            // 优化1：当前数与后续最小两个数之和已超target，后续组合均更大，直接终止循环
            if nums[i] + nums[i + 1] + nums[i + 2] > target {
                break;
            }
            // 优化2：当前数与数组最大两个数之和仍小于target，当前数过小，跳过进入下一轮
            if nums[i] + nums[n - 2] + nums[n - 1] < target {
                continue;
            }

            // 双指针初始化：left从i+1开始（避免重复使用元素），right从数组末尾开始
            let mut left = i + 1;
            let mut right = n - 1;

            while left < right {
                let sum = nums[i] + nums[left] + nums[right];
                // 总和小于target：左指针右移，增大数值以逼近target
                if sum < target {
                    left += 1;
                }
                // 总和大于target：右指针左移，减小数值以逼近target
                else if sum > target {
                    right -= 1;
                }
                // 总和等于target：计算当前组合下的三元组数量
                else {
                    // 分两种情况处理重复元素，避免重复计数
                    // 情况1：nums[left] == nums[right]，则[left, right]区间内所有数均相同
                    if nums[left] == nums[right] {
                        // 组合数C(k,2)，k为区间内元素个数，即right - left + 1
                        let count = (right - left + 1) as i64;
                        res = (res + count * (count - 1) / 2) % mod_val;
                        break; // 区间内所有组合已计算，直接退出当前双指针循环
                    }
                    // 情况2：nums[left] != nums[right]，分别统计两侧相同元素的个数
                    let mut left_count = 1;
                    // 统计left指针右侧连续相同的元素数量
                    while left < right && nums[left] == nums[left + 1] {
                        left += 1;
                        left_count += 1;
                    }
                    let mut right_count = 1;
                    // 统计right指针左侧连续相同的元素数量
                    while left < right && nums[right] == nums[right - 1] {
                        right -= 1;
                        right_count += 1;
                    }
                    // 两侧相同元素个数相乘，即为当前i、left、right组合下的三元组总数
                    res = (res + left_count as i64 * right_count as i64) % mod_val;
                    // 指针移动到下一组不同元素，避免重复计算
                    left += 1;
                    right -= 1;
                }
            }
        }

        res as i32
    }
}

// 948. 令牌放置
pub mod n948 {
    pub fn bag_of_tokens_score(mut tokens: Vec<i32>, mut power: i32) -> i32 {
        // 对tokens数组排序，为双指针策略奠定基础
        tokens.sort_unstable();
        let (mut left, mut right) = (0, tokens.len());
        let (mut current_score, mut max_score) = (0, 0);

        while left < right {
            if power >= tokens[left] {
                // 能量足够，消耗能量获取当前最小token，提升分数
                power -= tokens[left];
                current_score += 1;
                left += 1;
            } else if current_score > 0 {
                // 能量不足但有分数，消耗分数换取最大token，补充能量
                max_score = max_score.max(current_score);
                power += tokens[right - 1];
                current_score -= 1;
                right -= 1;
            } else {
                // 能量和分数均不足，无法继续操作，终止循环
                break;
            }
        }
        // 最终需比较最后一轮未更新的current_score与历史max_score
        max_score.max(current_score)
    }
}

// 11. 盛最多水的容器
pub mod n11 {
    pub fn max_area(height: Vec<i32>) -> i32 {
        let mut ans = 0;

        let mut left = 0;
        let mut right = height.len() - 1;

        while left < right {
            let area = (right - left) as i32 * height[left].min(height[right]);
            ans = ans.max(area);
            if height[left] < height[right] {
                // height[left] 与右边的任意垂线都无法组成一个比 ans 更大的面积
                left += 1;
            } else {
                // height[right] 与左边的任意垂线都无法组成一个比 ans 更大的面积
                right -= 1;
            }
        }

        ans
    }
}

// 42. 接雨水
pub mod n42 {
    pub fn trap(height: Vec<i32>) -> i32 {
        let mut ans = 0;

        let mut pre_max = 0; // 前缀最大值，随着左指针 left 的移动而更新
        let mut suf_max = 0; // 后缀最大值，随着右指针 right 的移动而更新

        let mut left = 0;
        let mut right = height.len() - 1;

        while left < right {
            pre_max = pre_max.max(height[left]);
            suf_max = suf_max.max(height[right]);

            if pre_max < suf_max {
                ans += pre_max - height[left];
                left += 1;
            } else {
                ans += suf_max - height[right];
                right -= 1;
            };
        }

        ans
    }
}

// 1616. 分割两个字符串得到回文串
pub mod n1616 {
    pub fn check_palindrome_formation(a: String, b: String) -> bool {
        // 辅助函数：检查 a_prefix + b_suffix 是否能形成回文串
        fn check(a: &[u8], b: &[u8]) -> bool {
            let mut i = 0;
            let mut j = a.len() - 1;
            // 相向双指针，尽可能匹配前后字符
            while i < j && a[i] == b[j] {
                i += 1;
                j -= 1;
            }
            // 检查中间剩余部分是否为回文
            is_palindrome(&a[i..=j]) || is_palindrome(&b[i..=j])
        }

        // 检查一个字符串片段是否为回文
        fn is_palindrome(s: &[u8]) -> bool {
            s.iter().eq(s.iter().rev())
        }

        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();

        // 两种组合方式都检查
        check(a_bytes, b_bytes) || check(b_bytes, a_bytes)
    }
}

// 1498. 满足条件的子序列数目
pub mod n1498 {
    const MOD: i64 = 1_000_000_007;

    pub fn num_subseq(mut nums: Vec<i32>, target: i32) -> i32 {
        nums.sort_unstable();
        let n = nums.len();

        // 预计算所需的幂值，只计算到所需的最大索引
        let max_pow = n;
        let mut pow2 = vec![1i64; max_pow];
        for i in 1..max_pow {
            pow2[i] = (pow2[i - 1] * 2) % MOD;
        }

        let mut ans: i64 = 0;
        let mut left = 0;
        let mut right = n - 1;

        while left <= right {
            if nums[left] + nums[right] <= target {
                // 计算右区间长度对应的幂值
                let count = right - left;
                ans = (ans + pow2[count]) % MOD;
                left += 1;
            } else {
                // 处理边界情况，避免下溢
                if right == 0 {
                    break;
                }
                right -= 1;
            }
        }

        ans as i32
    }
}

// 1782. 统计点对的数目
pub mod n1782 {
    use std::collections::HashMap;

    pub fn count_pairs(n: i32, edges: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        let n = n as usize;
        let mut deg = vec![0; n + 1]; // 节点1~n，索引0闲置
        let mut edge_cnt = HashMap::new();

        // 统计节点度数与边出现次数
        for e in edges {
            let x = e[0] as usize;
            let y = e[1] as usize;
            deg[x] += 1;
            deg[y] += 1;
            // 存储排序后的边，确保(x,y)与(y,x)视为同一条
            let key = if x < y { (x, y) } else { (y, x) };
            *edge_cnt.entry(key).or_insert(0) += 1;
        }

        let mut sorted_deg = deg.clone();
        sorted_deg.sort_unstable(); // 排序用于双指针
        let mut ans = vec![0; queries.len()];

        for (idx, &q) in queries.iter().enumerate() {
            let mut left = 1;
            let mut right = n;
            let mut cnt = 0;

            // 双指针统计初始符合条件的数对
            while left < right {
                if sorted_deg[left] + sorted_deg[right] <= q {
                    left += 1;
                } else {
                    cnt += right - left;
                    right -= 1;
                }
            }

            // 修正重复计算的边
            for (&(x, y), &c) in &edge_cnt {
                let sum = deg[x] + deg[y];
                if sum > q && sum <= q + c {
                    cnt -= 1;
                }
            }

            ans[idx] = cnt as i32;
        }

        ans
    }
}

// 3649. 完美对的数目
pub mod n3649 {
    pub fn perfect_pairs(mut nums: Vec<i32>) -> i64 {
        nums.sort_by(|a, b| a.abs().cmp(&b.abs()));

        let mut ans = 0;
        let mut left = 0;

        // 双指针遍历：j为右指针（当前元素b），left为左指针（找符合条件的最小a的索引）
        for (j, &b) in nums.iter().enumerate() {
            // 移动left，直到不满足 abs(a)*2 < abs(b)
            while nums[left].abs() * 2 < b.abs() {
                left += 1;
            }
            // 累加符合条件的(a,b)对数量：i范围[left, j-1]，共j-left个
            ans += (j - left) as i64;
        }

        ans
    }
}

// 1574. 删除最短的子数组使剩余数组有序
pub mod n1574 {
    pub fn find_length_of_shortest_subarray(arr: Vec<i32>) -> i32 {
        let n = arr.len();
        let mut right = n - 1;
        // 从右向左找到最长非递减后缀的起始位置
        while right > 0 && arr[right - 1] <= arr[right] {
            right -= 1;
        }
        // 若整个数组已非递减，直接返回0
        if right == 0 {
            return 0;
        }
        // 初始答案：删除前缀（从0到right-1），长度为right
        let mut ans = right;
        let mut left = 0;
        // 从左向右枚举最长非递减前缀的每个元素
        while left == 0 || arr[left - 1] <= arr[left] {
            // 找到后缀中第一个 >= 当前前缀元素的位置
            while right < n && arr[right] < arr[left] {
                right += 1;
            }
            // 计算删除区间[left+1, right-1]的长度，更新最小答案
            ans = ans.min(right - left - 1);
            left += 1;
        }
        ans as _
    }
}

// 2972. 统计移除递增子数组的数目 II
pub mod n2972 {
    pub fn incremovable_subarray_count(nums: Vec<i32>) -> i64 {
        let n = nums.len();
        let mut i = 0;
        while i < n - 1 && nums[i] < nums[i + 1] {
            i += 1;
        }
        if i == n - 1 {
            // 每个非空子数组都可以移除
            return n as i64 * (n + 1) as i64 / 2;
        }

        let mut i = i as i64;
        let mut ans = i + 2; // 不保留后缀的情况，一共 i+2 个
        // 枚举保留的后缀为 nums[j:]
        let mut j = n - 1;
        while j == n - 1 || nums[j] < nums[j + 1] {
            while i >= 0 && nums[i as usize] >= nums[j] {
                i -= 1;
            }
            // 可以保留前缀 nums[:i+1], nums[:i], ..., nums[:0] 一共 i+2 个
            ans += i + 2;
            j -= 1;
        }
        ans
    }
}
