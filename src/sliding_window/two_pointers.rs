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
