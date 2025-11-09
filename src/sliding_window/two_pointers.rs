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

// 2122. 还原原数组
pub mod n2122 {
    pub fn recover_array(mut nums: Vec<i32>) -> Vec<i32> {
        nums.sort_unstable();
        let n = nums.len();
        let half_n = n / 2;

        // 枚举所有可能的第二个元素
        for i in 1..n {
            // 优化：跳过与前一个元素相同的情况（避免重复计算相同k）
            if nums[i] == nums[i - 1] {
                continue;
            }

            let d = nums[i] - nums[0];
            // 检查d是否为偶数（k必须是整数，d=2k）
            if d % 2 != 0 {
                continue;
            }
            let k = d / 2;

            let mut visited = vec![false; n]; // 标记higher组的元素下标
            visited[i] = true;
            let mut ans = vec![(nums[0] + nums[i]) / 2]; // 第一个原数组元素

            let mut lo = 0;
            let mut hi = i + 1;
            let mut valid = true;

            // 双指针验证：lo找lower组，hi找higher组
            while hi < n {
                // 移动lo到下一个未被标记的lower元素（跳过higher组元素）
                lo += 1;
                while lo < n && visited[lo] {
                    lo += 1;
                }
                // 移动hi到第一个满足 nums[hi] - nums[lo] >= 2k 的位置
                while hi < n && nums[hi] - nums[lo] < 2 * k {
                    hi += 1;
                }
                // 验证是否找到合法的higher元素
                if hi >= n || nums[hi] - nums[lo] != 2 * k {
                    valid = false;
                    break;
                }
                // 标记higher元素，记录原数组元素
                visited[hi] = true;
                ans.push((nums[lo] + nums[hi]) / 2);
                hi += 1; // 准备找下一个higher
            }

            // 若收集到足够的原数组元素（长度为n/2），直接返回
            if valid && ans.len() == half_n {
                return ans;
            }
        }

        unreachable!("题目保证存在合法原数组，不会执行到此处");
    }
}

// 2234. 花园的最大总美丽值
pub mod n2234 {
    pub fn maximum_beauty(
        mut flowers: Vec<i32>,
        new_flowers: i64,
        target: i32,
        full: i32,
        partial: i32,
    ) -> i64 {
        let n = flowers.len() as i64;
        let full = full as i64;
        let partial = partial as i64;

        // 如果全部种满，还剩下多少朵花？先允许可以为负数
        let mut left_flowers = new_flowers - target as i64 * n;
        for flower in &mut flowers {
            *flower = (*flower).min(target); // 把超过 target 的 flowers[i] 改成 target。这一来可以简化双指针的计算，二来可以加快排序的效率，尤其是当很多 flowers[i] 都超过 target 的情况。
            left_flowers += *flower as i64; // 把已有的加回来
        }

        // 没有种花，所有花园都已种满
        if left_flowers == new_flowers {
            return n * full;
        }

        // 可以全部种满
        if left_flowers >= 0 {
            // 两种策略取最大值：留一个花园种 target-1 朵花，其余种满；或者，全部种满
            return ((target - 1) as i64 * partial + (n - 1) * full).max(n * full);
        }

        flowers.sort_unstable(); // 时间复杂度的瓶颈在这，尽量写在后面

        let mut ans = 0;
        let mut pre_sum = 0;
        let mut j = 0;
        // 枚举 i，表示后缀 [i, n-1] 种满（i=0 的情况上面已讨论）
        for i in 1..=n as usize {
            // 撤销，flowers[i-1] 不种满
            left_flowers += (target - flowers[i - 1]) as i64;
            if left_flowers < 0 {
                // 花不能为负数，需要继续撤销
                continue;
            }

            // 满足以下条件说明 [0, j-1] 都可以种 flowers[j] 朵花
            while j < i && flowers[j] as i64 * j as i64 <= pre_sum + left_flowers {
                pre_sum += flowers[j] as i64;
                j += 1;
            }

            // 计算总美丽值
            // 在前缀 [0, j-1] 中均匀种花，这样最小值最大
            let avg = (left_flowers + pre_sum) / j as i64; // 由于上面特判了，这里 avg 一定小于 target
            let total_beauty = avg * partial + (n - i as i64) * full;
            ans = ans.max(total_beauty);
        }

        ans
    }
}

// 581. 最短无序连续子数组
pub mod n581 {
    pub fn find_unsorted_subarray(nums: &[i32]) -> i32 {
        let n = nums.len();
        if n <= 1 {
            return 0;
        }

        let (mut maxn, mut right) = (i32::MIN, n);
        let (mut minn, mut left) = (i32::MAX, n);

        for i in 0..n {
            // 从左向右找右边界：记录当前最大值，若遇到比最大值小的元素，更新右边界
            if maxn > nums[i] {
                right = i;
            } else {
                maxn = nums[i];
            }

            // 从右向左找左边界：记录当前最小值，若遇到比最小值大的元素，更新左边界
            let j = n - 1 - i;
            if minn < nums[j] {
                left = j;
            } else {
                minn = nums[j];
            }
        }

        // 若right未更新（数组已排序），返回0；否则返回子数组长度
        if right == n {
            0
        } else {
            (right - left + 1) as _
        }
    }
}

// 1793. 好子数组的最大分数
pub mod n1793 {
    pub fn maximum_score(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let k = k as usize;
        let mut ans = nums[k];
        let mut min_h = nums[k];
        let mut i = k;
        let mut j = k;
        for _ in 0..n - 1 {
            if j == n - 1 || i > 0 && nums[i - 1] > nums[j + 1] {
                i -= 1;
                min_h = min_h.min(nums[i]);
            } else {
                j += 1;
                min_h = min_h.min(nums[j]);
            }
            ans = ans.max(min_h * (j - i + 1) as i32);
        }
        ans
    }
}

// 27. 移除元素
pub mod n27 {
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len(); // 初始指向数组末尾的下一个位置，左闭右开区间 [left, right)

        while left < right {
            if nums[left] != val {
                // 左指针元素合法，直接右移
                left += 1;
            } else if nums[right - 1] == val {
                // 右指针前一个元素也是目标值，直接左移右边界
                right -= 1;
            } else {
                // 用右侧合法元素覆盖左侧目标值，双指针同时移动
                nums[left] = nums[right - 1];
                right -= 1;
                left += 1;
            }
        }

        left as _
    }
}

// 26. 删除有序数组中的重复项
pub mod n26 {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        let mut k = 1;

        for i in 1..nums.len() {
            if nums[i] != nums[i - 1] {
                nums[k] = nums[i];
                k += 1;
            }
        }

        k as _
    }
}

// 80. 删除有序数组中的重复项 II
pub mod n80 {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        let n = nums.len();
        // 长度小于等于2时无需处理，直接返回原长度
        if n <= 2 {
            return n as i32;
        }
        // 快慢指针均从索引2开始
        let mut slow = 2;
        let mut fast = 2;

        while fast < n {
            // 对比慢指针前两位与快指针元素，不相等则赋值并移动慢指针
            if nums[slow - 2] != nums[fast] {
                nums[slow] = nums[fast];
                slow += 1;
            }
            fast += 1;
        }
        // 返回慢指针位置（即去重后数组的有效长度）
        slow as i32
    }
}

// 2273. 移除字母异位词后的结果数组
pub mod n2273 {
    pub fn remove_anagrams(mut words: Vec<String>) -> Vec<String> {
        let mut base = vec![];
        let mut k = 0;
        for i in 0..words.len() {
            let mut s = words[i].as_bytes().to_vec();
            s.sort_unstable();
            if s != base {
                base = s;
                words[k] = words[i].clone();
                k += 1;
            }
        }
        words.truncate(k);
        words
    }
}

// 283. 移动零
pub mod n283 {
    pub fn move_zeroes(nums: &mut Vec<i32>) {
        let n = nums.len();
        let mut left = 0; // 指向非零元素应放置的位置

        for right in 0..n {
            if nums[right] != 0 {
                // 交换非零元素到left位置，left指针后移
                nums.swap(left, right);
                left += 1;
            }
        }
    }
}

// 905. 按奇偶排序数组
pub mod n905 {
    pub fn sort_array_by_parity(mut nums: Vec<i32>) -> Vec<i32> {
        let mut i = 0;
        let mut j = nums.len();
        while i < j {
            if nums[i] % 2 == 0 {
                // 寻找最左边的奇数
                i += 1;
            } else if nums[j - 1] % 2 == 1 {
                // 寻找最右边的偶数
                j -= 1;
            } else {
                nums.swap(i, j - 1);
                i += 1;
                j -= 1;
            }
        }
        nums
    }
}

// 922. 按奇偶排序数组 II
pub mod n922 {
    pub fn sort_array_by_parity_ii(mut nums: Vec<i32>) -> Vec<i32> {
        let mut i = 0;
        let mut j = 1;
        while i < nums.len() {
            if nums[i] & 1 == 0 {
                // 寻找偶数下标中最左边的奇数
                i += 2;
            } else if nums[j] & 1 == 1 {
                // 寻找奇数下标中最左边的偶数
                j += 2;
            } else {
                nums.swap(i, j);
                i += 2;
                j += 2;
            }
        }
        nums
    }
}

// 2460. 对数组执行操作
pub mod n2460 {
    pub fn apply_operations(nums: &mut Vec<i32>) -> &mut Vec<i32> {
        let n = nums.len();
        let mut j = 0;

        for i in 0..n {
            // 处理相邻元素：相等则当前翻倍、下一个置零
            if i + 1 < n && nums[i] == nums[i + 1] {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
            // 双指针交换非零元素到前部
            if nums[i] != 0 {
                nums.swap(i, j);
                j += 1;
            }
        }

        nums
    }
}

// 1089. 复写零
pub mod n1089 {
    pub fn duplicate_zeros(arr: &mut Vec<i32>) {
        let n = arr.len();
        let mut i = 0;
        let mut j = 0;

        // 第一步：统计最终填充后的位置，确定双指针终点
        while j < n {
            if arr[i] == 0 {
                j += 1; // 遇到0，j多走一步（模拟后续复制一个0）
            }
            i += 1;
            j += 1;
        }

        // 调整指针到最后一个有效元素的位置
        i -= 1;
        j -= 1;

        // 第二步：从后向前填充，处理0的重复
        while i < n {
            // i >=0 处理 usize 无符号下溢，i <n 确保边界
            // 先填充当前i对应的元素到j位置
            if j < n {
                arr[j] = arr[i];
            }
            // 若当前元素是0，需再向前填充一个0
            if arr[i] == 0 {
                j -= 1; // j先左移，避免覆盖
                if j < n {
                    arr[j] = 0;
                }
            }

            if i == 0 {
                break;
            }

            // 双指针同时左移，处理前一个元素
            i -= 1;
            j -= 1;
        }
    }
}

// 75. 颜色分类
pub mod n75 {
    pub fn sort_colors(nums: &mut Vec<i32>) {
        let n = nums.len();
        if n < 2 {
            return;
        }

        // 区间定义：
        // [0..zero) 全部为 0
        // [zero..i) 全部为 1
        // [two..n) 全部为 2
        let mut zero = 0;
        let mut two = n;
        let mut i = 0;

        while i < two {
            if nums[i] == 0 {
                // 交换当前元素与zero位置元素
                nums.swap(i, zero);
                zero += 1;
                i += 1;
            } else if nums[i] == 1 {
                // 直接归为1的区间
                i += 1;
            } else {
                // 交换前先将two指针左移（因为two指向的是下一个2应该放置的位置）
                two -= 1;
                nums.swap(i, two);
            }
        }
    }
}

// 2109. 向字符串添加空格
pub mod n2109 {
    pub fn add_spaces(s: String, spaces: Vec<i32>) -> String {
        let mut result = Vec::new();
        let mut j = 0; // 用于追踪spaces的当前索引
        let s_chars: Vec<char> = s.chars().collect(); // 将字符串转为char数组，便于索引访问

        for (i, &c) in s_chars.iter().enumerate() {
            // 检查当前索引是否需要插入空格
            if j < spaces.len() && spaces[j] == i as i32 {
                result.push(' ');
                j += 1;
            }
            result.push(c);
        }

        result.into_iter().collect() // 将char向量转为String返回
    }
}

// 2540. 最小公共值
pub mod n2540 {
    pub fn get_common(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        // 初始化两个指针，分别指向两个数组的起始位置
        let (mut i, mut j) = (0, 0);

        // 双指针遍历两个数组，直到任一指针超出数组长度（避免越界）
        while i < nums1.len() && j < nums2.len() {
            // 找到公共元素：因数组有序，首次找到的即为最小公共整数，直接返回
            if nums1[i] == nums2[j] {
                return nums1[i];
            }
            // 若nums1当前元素更小，移动nums1的指针（尝试找更大的元素匹配）
            else if nums1[i] < nums2[j] {
                i += 1;
            }
            // 若nums2当前元素更小，移动nums2的指针（尝试找更大的元素匹配）
            else {
                j += 1;
            }
        }

        // 遍历结束仍未找到公共元素，返回 -1
        -1
    }
}

// 88. 合并两个有序数组
pub mod n88 {
    pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
        let mut p1 = m as usize;
        let mut p2 = n as usize;
        let mut p = p1 + p2;
        // nums2 还有要合并的元素
        while p2 > 0 {
            // 如果 p1 < 0，那么走 else 分支，把 nums2 合并到 nums1 中
            if p1 > 0 && nums1[p1 - 1] > nums2[p2 - 1] {
                nums1[p - 1] = nums1[p1 - 1]; // 填入 nums1[p1-1]
                p1 -= 1;
            } else {
                nums1[p - 1] = nums2[p2 - 1]; // 填入 nums2[p2-1]
                p2 -= 1;
            }
            p -= 1;
        }
    }
}

// 2570. 合并两个二维数组 - 求和法
pub mod n2570 {
    pub fn merge_arrays(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        let (n, m) = (a.len(), b.len());

        loop {
            // 若a遍历完，追加b剩余元素并返回
            if i == n {
                result.extend_from_slice(&b[j..]);
                return result;
            }
            // 若b遍历完，追加a剩余元素并返回
            if j == m {
                result.extend_from_slice(&a[i..]);
                return result;
            }

            if a[i][0] < b[j][0] {
                // a的ID更小，添加a的元素并移动指针
                result.push(a[i].clone());
                i += 1;
            } else if a[i][0] > b[j][0] {
                // b的ID更小，添加b的元素并移动指针
                result.push(b[j].clone());
                j += 1;
            } else {
                // ID相同，值相加后添加，双指针同时移动
                result.push(vec![a[i][0], a[i][1] + b[j][1]]);
                i += 1;
                j += 1;
            }
        }
    }
}

// 1855. 下标对中的最大距离
pub mod n1855 {
    pub fn max_distance(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        let n1 = nums1.len(); // nums1 的长度
        let n2 = nums2.len(); // nums2 的长度
        let mut i = 0; // 遍历 nums1 的指针（初始化为 0）
        let mut res = 0; // 存储最大距离的结果（初始化为 0）

        // 遍历 nums2 的每个元素（j 为当前索引）
        for j in 0..n2 {
            // 移动 i 指针：确保 i < n1 且 nums1[i] > nums2[j] 时，i 后移（满足 nums1[i] ≤ nums2[j] 的前提）
            while i < n1 && nums1[i] > nums2[j] {
                i += 1;
            }
            if i < n1 && i <= j {
                res = res.max(j - i);
            }
        }

        res as _
    }
}

// 1385. 两个数组间的距离值
pub mod n1385 {
    pub fn find_the_distance_value(mut arr1: Vec<i32>, mut arr2: Vec<i32>, d: i32) -> i32 {
        arr1.sort_unstable();
        arr2.sort_unstable();
        let mut ans = 0;
        let mut j = 0;
        for x in arr1 {
            while j < arr2.len() && arr2[j] < x - d {
                j += 1;
            }
            if j == arr2.len() || arr2[j] > x + d {
                ans += 1;
            }
        }
        ans
    }
}

// 925. 长按键入
pub mod n925 {
    pub fn is_long_pressed_name(name: String, typed: String) -> bool {
        let name_chars: Vec<char> = name.chars().collect();
        let typed_chars: Vec<char> = typed.chars().collect();
        let mut i = 0; // 指向name的指针
        let mut j = 0; // 指向typed的指针

        while j < typed_chars.len() {
            if i < name_chars.len() && name_chars[i] == typed_chars[j] {
                // 字符匹配，双指针同时后移
                i += 1;
                j += 1;
            } else if j > 0 && typed_chars[j] == typed_chars[j - 1] {
                // 当前字符是长按重复，仅typed指针后移
                j += 1;
            } else {
                // 字符不匹配且非长按，直接返回false
                return false;
            }
        }

        // 遍历结束后，需确保name已完全匹配（i到达末尾）
        i == name_chars.len()
    }
}

// 809. 情感丰富的文字
pub mod n809 {
    pub fn expressive_words(s: String, words: Vec<String>) -> i32 {
        // 判断 t 是否能通过扩展得到 s
        let is_expressive = |s: &str, t: &str| -> bool {
            let (mut i, mut j) = (0, 0);
            let (s_chars, t_chars) = (s.as_bytes(), t.as_bytes());
            let (s_len, t_len) = (s_chars.len(), t_chars.len());

            while i < s_len && j < t_len {
                // 字符不同直接返回 false
                if s_chars[i] != t_chars[j] {
                    return false;
                }
                let ch = s_chars[i];

                // 统计 s 中当前字符的连续个数
                let mut cnt_s = 0;
                while i < s_len && s_chars[i] == ch {
                    cnt_s += 1;
                    i += 1;
                }

                // 统计 t 中当前字符的连续个数
                let mut cnt_t = 0;
                while j < t_len && t_chars[j] == ch {
                    cnt_t += 1;
                    j += 1;
                }

                // 校验个数规则：s的个数不能少于t，且不相等时s的个数需≥3
                if cnt_s < cnt_t || (cnt_s != cnt_t && cnt_s < 3) {
                    return false;
                }
            }

            // 需确保两个字符串都遍历完（避免一方有剩余字符）
            i == s_len && j == t_len
        };
        words
            .into_iter()
            .filter(|word| is_expressive(&s, word))
            .count() as i32
    }
}

// 2337. 移动片段得到字符串
pub mod n2337 {
    pub fn can_change(start: String, target: String) -> bool {
        let (s, t) = (start.as_bytes(), target.as_bytes());
        let (n, m) = (s.len(), t.len());
        if n != m {
            return false;
        }

        let mut i = 0;
        let mut j = 0;

        while i < n || j < m {
            while i < n && s[i] == b'_' {
                i += 1;
            }
            while j < m && t[j] == b'_' {
                j += 1;
            }
            if i < n && j < m {
                if s[i] != t[j] {
                    return false;
                }
                if s[i] == b'L' && i < j {
                    return false;
                }
                if s[i] == b'R' && i > j {
                    return false;
                }
                i += 1;
                j += 1;
            } else if i < n || j < m {
                return false;
            }
        }
        true
    }
}

// 777. 在 LR 字符串中交换相邻字符
pub mod n777 {
    pub fn can_transform(start: String, target: String) -> bool {
        let (s, t) = (start.as_bytes(), target.as_bytes());
        let (n, m) = (s.len(), t.len());
        if n != m {
            return false;
        }

        let mut i = 0;
        let mut j = 0;

        while i < n || j < m {
            while i < n && s[i] == b'X' {
                i += 1;
            }
            while j < m && t[j] == b'X' {
                j += 1;
            }
            if i < n && j < m {
                if s[i] != t[j] {
                    return false;
                }
                if s[i] == b'L' && i < j {
                    return false;
                }
                if s[i] == b'R' && i > j {
                    return false;
                }
                i += 1;
                j += 1;
            } else if i < n || j < m {
                return false;
            }
        }
        true
    }
}

// 844. 比较含退格的字符串
pub mod n844 {
    pub fn backspace_compare(s: String, t: String) -> bool {
        let (mut i, mut j) = (s.len(), t.len());
        let (mut skip_s, mut skip_t) = (0, 0);
        let s_bytes = s.as_bytes();
        let t_bytes = t.as_bytes();

        while i > 0 || j > 0 {
            // 处理 s 的退格，找到当前有效字符位置
            while i > 0 {
                if s_bytes[i - 1] == b'#' {
                    skip_s += 1;
                    i -= 1;
                } else if skip_s > 0 {
                    skip_s -= 1;
                    i -= 1;
                } else {
                    break;
                }
            }

            // 处理 t 的退格，找到当前有效字符位置
            while j > 0 {
                if t_bytes[j - 1] == b'#' {
                    skip_t += 1;
                    j -= 1;
                } else if skip_t > 0 {
                    skip_t -= 1;
                    j -= 1;
                } else {
                    break;
                }
            }

            // 两个都未遍历结束
            if i > 0 && j > 0 {
                if s_bytes[i - 1] != t_bytes[j - 1] {
                    return false;
                }
            }
            // 只有一个未遍历结束
            else if i > 0 || j > 0 {
                return false;
            }

            // 移动指针，继续向前遍历
            i = i.saturating_sub(1);
            j = j.saturating_sub(1);
        }

        true
    }
}

// 986. 区间列表的交集
pub mod n986 {
    pub fn interval_intersection(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let len_a = a.len();
        let len_b = b.len();

        while i < len_a && j < len_b {
            // 计算交集的起始和结束位置
            let lo = a[i][0].max(b[j][0]);
            let hi = a[i][1].min(b[j][1]);

            // 若存在有效交集，加入结果集
            if lo <= hi {
                result.push(vec![lo, hi]);
            }

            // 移除终点更小的区间（移动对应指针）
            if a[i][1] < b[j][1] {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }
}

// 1537. 最大得分
pub mod n1537 {
    pub fn max_sum(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        const MOD: i64 = 1_000_000_007;
        let (mut sum1, mut sum2) = (0i64, 0i64);
        let (mut i, mut j) = (0usize, 0usize);
        let (len1, len2) = (nums1.len(), nums2.len());

        while i < len1 && j < len2 {
            if nums1[i] < nums2[j] {
                // 小的一方累加并前进（nums1[i] 更小）
                sum1 += nums1[i] as i64;
                i += 1;
            } else if nums1[i] > nums2[j] {
                // 小的一方累加并前进（nums2[j] 更小）
                sum2 += nums2[j] as i64;
                j += 1;
            } else {
                // 遇到公共节点：取两条路径的最大值 + 公共节点值，取余后同步更新
                let current = sum1.max(sum2) + nums1[i] as i64 % MOD;
                sum1 = current;
                sum2 = current;
                i += 1;
                j += 1;
            }
        }

        // 累加剩余元素
        sum1 += nums1[i..].iter().map(|&x| x as i64).sum::<i64>();
        sum2 += nums2[j..].iter().map(|&x| x as i64).sum::<i64>();

        // 取最大值后取余，转为 i32 返回
        (sum1.max(sum2) % MOD) as i32
    }
}

// 392. 判断子序列
pub mod n392 {
    pub fn is_subsequence(s: String, t: String) -> bool {
        let (mut i, mut j) = (0, 0);
        let (s_chars, t_chars) = (s.as_bytes(), t.as_bytes());
        let (n, m) = (s_chars.len(), t_chars.len());

        while i < n && j < m {
            if s_chars[i] == t_chars[j] {
                i += 1;
            }
            j += 1;
        }

        i == n
    }
}
