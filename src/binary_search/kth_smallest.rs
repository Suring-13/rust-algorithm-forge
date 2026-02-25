// 668. 乘法表中第k小的数
pub mod n668 {
    pub fn find_kth_number(m: i32, n: i32, k: i32) -> i32 {
        // 转换为u32避免负数/溢出（乘法表数值非负，且m/n/k均为正整数）
        let m = m as u32;
        let n = n as u32;
        let k = k as u32;

        // 检查函数：判断≤x的数字数量是否≥k
        let check = |x: u32| -> bool {
            let mut cnt = 0u32;
            // 遍历每一行（1~m）
            for i in 1..=m {
                // 该行≤x的数量：min(x//i, n)
                let row_cnt = n.min(x / i);
                cnt += row_cnt;
                // 提前终止：数量已达标，无需继续遍历
                if cnt >= k {
                    return true;
                }
            }
            cnt >= k
        };

        let mut left = 0u32;
        let mut right = m * n; // 上界为m*n

        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                right = mid; // 满足条件，尝试更小的x
            } else {
                left = mid + 1; // 不满足，需要更大的x
            }
        }

        right as i32
    }
}

// 378. 有序矩阵中第 K 小的元素
pub mod n378 {
    pub fn kth_smallest(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let n = matrix.len();

        let check = |mx: i32| -> bool {
            let mut cnt = 0; // matrix 中的 <= mx 的元素个数
            let mut i = 0;
            let mut j = n as i32 - 1; // 从右上角开始
            while i < n && j >= 0 && cnt < k {
                if matrix[i][j as usize] > mx {
                    j -= 1; // 排除第 j 列
                } else {
                    cnt += j + 1; // 从 matrix[i][0] 到 matrix[i][j] 都 <= mx
                    i += 1;
                }
            }
            cnt >= k
        };

        let mut left = matrix[0][0];
        let mut right = matrix[n - 1][n - 1];
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        right
    }
}

// 719. 找出第 K 小的数对距离
pub mod n719 {
    pub fn smallest_distance_pair(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable();
        let k = k as usize;
        let n = nums.len();

        // 检查函数：统计间距≤mx的数对数量是否≥k
        let check = |mx: i32| -> bool {
            let mut cnt = 0usize;
            let mut i = 0usize; // 滑动窗口左指针
            for j in 0..n {
                // 移动左指针，直到nums[j] - nums[i] ≤ mx
                while nums[j] - nums[i] > mx {
                    i += 1;
                }
                // 累加当前右指针j对应的有效数对数量
                cnt += j - i;
                // 提前终止：数量已达标，无需继续遍历
                if cnt >= k {
                    return true;
                }
            }
            cnt >= k
        };

        let mut left = 0;
        let mut right = nums[n - 1] - nums[0];

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid; // 满足条件，尝试更小的mx
            } else {
                left = mid + 1; // 不满足，需要更大的mx
            }
        }

        right
    }
}

// 878. 第 N 个神奇数字
pub mod n878 {
    pub fn nth_magical_number(n: i32, a: i32, b: i32) -> i32 {
        const MOD: i64 = 1_000_000_007;

        let n = n as i64;
        let a = a as i64;
        let b = b as i64;

        // 最大公约数 gcd
        fn gcd(x: i64, y: i64) -> i64 {
            if y == 0 { x } else { gcd(y, x % y) }
        }
        let lcm = a / gcd(a, b) * b;

        let mut left = a.min(b) + n - 1;
        let mut right = a.min(b) * n;

        while left < right {
            let mid = left + (right - left) / 2;
            let cnt = mid / a + mid / b - mid / lcm;
            if cnt >= n {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        (right % MOD) as i32
    }
}

// 1201. 丑数 III
pub mod n1201 {
    pub fn nth_ugly_number(n: i32, a: i32, b: i32, c: i32) -> i32 {
        let (n, a, b, c) = (n as i64, a as i64, b as i64, c as i64);

        // 最大公约数
        fn gcd(x: i64, y: i64) -> i64 {
            if y == 0 { x } else { gcd(y, x % y) }
        }
        // 最小公倍数
        fn lcm(x: i64, y: i64) -> i64 {
            x / gcd(x, y) * y
        }

        let lcm_ab = lcm(a, b);
        let lcm_ac = lcm(a, c);
        let lcm_bc = lcm(b, c);
        let lcm_abc = lcm(lcm_ab, c);

        // 计算 <=x 的丑数个数（容斥）
        let check = |x: i64| -> bool {
            let count = x / a + x / b + x / c - x / lcm_ab - x / lcm_ac - x / lcm_bc + x / lcm_abc;
            count >= n
        };

        let mut left = a.min(b).min(c) + n - 1;
        let mut right = a.min(b).min(c) * n;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        right as _
    }
}
